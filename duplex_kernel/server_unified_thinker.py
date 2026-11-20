"""Unified native Thinker decode route."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Mapping, Sequence

import torch

from duplex_kernel.audio_response_protocol import CONTROLLER_GUIDED_SAY_V1
from duplex_kernel.live_say import LiveSayCancelled
from duplex_kernel.session import DuplexSession
from duplex_kernel.unified_native_decoder import (
    DEFAULT_CONTINUATION_GENERATED_BUDGET,
    NativeDecodeSessionStatus,
    UnifiedNativeDecoder,
)
from duplex_kernel.server_unified_templates import _validate_private_transition_chunk


def _controller_handoff_is_valid(controller: Mapping[str, Any]) -> bool:
    if not controller.get("promoted"):
        return False
    handoff = controller.get("handoff_state")
    target = handoff.get("target_branch") if isinstance(handoff, Mapping) else None
    revision_id = str(controller.get("revision_id", ""))
    return bool(
        isinstance(target, dict)
        and target.get("_adopted_native_session_controller") is controller
        and str(target.get("revision_id", "")) == revision_id
        and str(target.get("_streaming_revision_id", revision_id)) == revision_id
    )


def _cached_controller_result(controller: dict[str, Any]) -> dict[str, Any]:
    fatal_error = controller.get("fatal_error")
    if isinstance(fatal_error, BaseException):
        raise fatal_error
    result = controller.get("final_result")
    if not isinstance(result, Mapping):
        raise RuntimeError("native session controller completed without a result")
    cached = dict(result)
    if _controller_handoff_is_valid(controller):
        cached["revision_handoff"] = True
    return cached


def _run_controller_to_completion(
    controller: dict[str, Any],
    *,
    advance_controller: Any,
    finish_decoded: Any,
) -> dict[str, Any]:
    """Complete one controller once; concurrent callers share its result."""
    condition = controller.get("lifecycle_condition")
    if not isinstance(condition, threading.Condition):
        condition = threading.Condition()
        controller["lifecycle_condition"] = condition
    owner = threading.get_ident()
    with condition:
        if controller.get("done"):
            return _cached_controller_result(controller)
        while controller.get("completion_owner") is not None:
            if controller.get("completion_owner") == owner:
                raise RuntimeError("native session controller completion is reentrant")
            condition.wait()
            if controller.get("done"):
                return _cached_controller_result(controller)
        controller["completion_owner"] = owner

    try:
        while True:
            step = advance_controller(lambda _progress: False)
            if step.status is NativeDecodeSessionStatus.FAILED:
                fatal_error = controller.get("fatal_error")
                if isinstance(fatal_error, BaseException):
                    raise fatal_error
                session = controller.get("session")
                fatal_error = getattr(session, "fatal_error", None)
                if isinstance(fatal_error, BaseException):
                    raise fatal_error
                raise RuntimeError("native decoder returned FAILED without its original error")
            if step.status in {
                NativeDecodeSessionStatus.DONE,
                NativeDecodeSessionStatus.CANCELLED,
            }:
                if controller.get("done"):
                    return _cached_controller_result(controller)
                result = finish_decoded(step.result)
                if not isinstance(result, Mapping):
                    raise TypeError("decoded controller result must be a mapping")
                with condition:
                    if not controller.get("done"):
                        controller["final_result"] = (
                            result if isinstance(result, dict) else dict(result)
                        )
                        controller["done"] = True
                return _cached_controller_result(controller)
    except BaseException as exc:
        with condition:
            if not controller.get("done"):
                controller["fatal_error"] = exc
                controller["done"] = True
            condition.notify_all()
        if isinstance(controller.get("fatal_error"), BaseException):
            raise controller["fatal_error"]
        raise
    finally:
        with condition:
            controller["completion_owner"] = None
            condition.notify_all()


def _run_unified_native_thinker(
    engine: Any,
    session: DuplexSession,
    request: dict[str, Any],
    *,
    worker: Any,
    transition_token_ids: tuple[int, ...],
    think_kwargs: dict[str, Any],
    generation_id: int | None,
    branch: dict[str, Any],
) -> dict[str, Any]:
    """Decode the Thinker suffix through the listener's native cache owner."""
    from duplex_kernel.server_speech_ack import _record_unified_native_profile_timestamp
    from duplex_kernel.server_unified import _consume_unified_native_text_append_result
    listener_generation_id = int(branch["listener_generation_id"])
    expected_branch_revision_id = str(branch.get("revision_id", "r0"))
    stop_check = think_kwargs.get("stop_check")
    response_cancel_event = request.get("_response_cancel_event")

    def decode_cancelled() -> bool:
        return bool(
            (response_cancel_event is not None and response_cancel_event.is_set())
            or (callable(stop_check) and stop_check())
            or str(branch.get("revision_id", "r0")) != expected_branch_revision_id
        )

    if decode_cancelled():
        raise asyncio.CancelledError
    def continuation_option(name: str) -> Any:
        value = request.get(name)
        return value if value is not None else think_kwargs.get(name)

    continuation_context_transition_ids = tuple(int(value) for value in transition_token_ids)
    continuation_prefix_value = continuation_option("_continuation_prefix_token_ids")
    continuation_prefix_ids: tuple[int, ...] | None = None
    continuation_logical_offset: int | None = None
    continuation_budget_used: int | None = None
    if continuation_prefix_value is not None:
        if not isinstance(continuation_prefix_value, (list, tuple)):
            raise TypeError("_continuation_prefix_token_ids deve essere una sequenza")
        prefix_ids: list[int] = []
        for value in continuation_prefix_value:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("_continuation_prefix_token_ids contiene un ID non valido")
            prefix_ids.append(int(value))
        continuation_prefix_ids = tuple(prefix_ids)
        logical_value = continuation_option("_logical_source_offset")
        budget_value = continuation_option("_generated_budget_used")
        for name, value in (
            ("_logical_source_offset", logical_value),
            ("_generated_budget_used", budget_value),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} deve essere un intero non negativo")
        continuation_logical_offset = int(logical_value)
        continuation_budget_used = int(budget_value)
        if continuation_budget_used > DEFAULT_CONTINUATION_GENERATED_BUDGET:
            raise ValueError("_generated_budget_used supera il budget globale di 768 token")
        # Reconstruct the response branch from its required owner transition
        # plus the frozen generated prefix. Only the generated prefix belongs
        # to sampler history; replayed context never emits packets.
        transition_token_ids = (
            (*continuation_context_transition_ids, *continuation_prefix_ids)
            if continuation_budget_used < DEFAULT_CONTINUATION_GENERATED_BUDGET
            else ()
        )
    elif any(
        continuation_option(name) is not None
        for name in ("_logical_source_offset", "_generated_budget_used")
    ):
        raise ValueError(
            "_logical_source_offset and _generated_budget_used require a frozen continuation prefix"
        )
    private_handle = branch.get("ephemeral_decode_handle")
    response_handle = branch.get("response_decode_handle")
    response_handle_ref: dict[str, Any] = {"value": response_handle}
    private_transition_ids = tuple(int(value) for value in transition_token_ids)
    private_transition_offset = 0
    private_transition_remaining = (
        len(private_transition_ids) if private_handle is not None else 0
    )
    cache_cursor = int(
        private_handle.source_cache_position
        if private_handle is not None
        else response_handle.source_cache_position
        if response_handle is not None
        else branch["cache_position"]
    )
    decode_budget_reservation: dict[str, Any] | None = None
    reservation_cleanup_state = {"released": False}

    def release_decode_budget_reservation(original_error: BaseException | None = None) -> None:
        reservation = decode_budget_reservation
        if reservation is None or reservation_cleanup_state["released"]:
            return
        try:
            latest = worker.session_state(session.session_id).result(timeout=5.0)
            if int(latest.generation_id) == listener_generation_id:
                release_deadline = time.monotonic() + 2.0
                while True:
                    try:
                        worker.reserve_tail_tokens(
                            session.session_id,
                            expected_generation_id=listener_generation_id,
                            expected_cache_position=int(latest.cache_position),
                            token_count=int(reservation["close_reservation_tokens"]),
                        ).result(timeout=5.0)
                        break
                    except RuntimeError as exc:
                        if str(exc) not in {
                            "tail reservation requires a completed listener frame watermark",
                            "tail reservation cannot overlap a shared listener tick",
                            "unified native operation lost its generation/cache-position compare-and-swap",
                        } or time.monotonic() >= release_deadline:
                            raise
                        latest = worker.session_state(session.session_id).result(timeout=5.0)
                        if int(latest.generation_id) != listener_generation_id:
                            break
                        time.sleep(0.002)
            reservation_cleanup_state["released"] = True
        except BaseException as release_error:
            if original_error is None:
                raise
            if hasattr(original_error, "add_note"):
                original_error.add_note(
                    "response KV reservation cleanup also failed: "
                    f"{type(release_error).__name__}: {release_error}"
                )

    current = worker.session_state(session.session_id).result(timeout=5.0)
    if int(current.generation_id) != listener_generation_id:
        raise RuntimeError("unified Thinker branch lost its listener generation")
    if private_handle is not None:
        if (
            private_handle.parent_session_id != session.session_id
            or int(private_handle.parent_generation_id) != listener_generation_id
            or int(private_handle.source_cache_position) != cache_cursor
        ):
            raise RuntimeError("private Thinker branch identity differs from its listener source")
        branch["cache_position"] = cache_cursor
        branch["turn_start_cache_position"] = cache_cursor
    if response_handle is not None:
        if (
            response_handle.parent_session_id != session.session_id
            or int(response_handle.parent_generation_id) != listener_generation_id
            or str(response_handle.response_id) != str(branch.get("response_id"))
            or int(response_handle.response_generation_id) != int(
                branch.get("response_generation_id", generation_id)
            )
            or str(response_handle.revision_id) != str(branch.get("revision_id"))
            or int(response_handle.source_cache_position) != cache_cursor
        ):
            raise RuntimeError("response decode handle identity differs from its native branch")
        branch["cache_position"] = cache_cursor
        branch["turn_start_cache_position"] = cache_cursor
    elif private_handle is None and int(branch.get("native_token_append_count", 0)) == 0:
        cache_cursor = int(current.cache_position)
        branch["cache_position"] = cache_cursor
        branch["turn_start_cache_position"] = cache_cursor
    branch["transition_token_ids"] = private_transition_ids
    # The native owner can rebase the first append while a listener tick is in
    # flight. Until the transition has actually committed, its suffix cursor
    # is unknown; anchor it from the append result below.
    branch["spoken_prefix_base_cursor"] = cache_cursor
    branch["spoken_prefix_transition_cursor"] = None
    if not tuple(branch.get("client_consumed_segments", ())):
        branch["spoken_prefix_exclusive_cursor"] = int(branch["spoken_prefix_base_cursor"])
        branch["client_consumed_prefix_token_ids"] = ()
        branch["client_consumed_prefix_exclusive_cursor"] = int(cache_cursor)
        branch["client_consumed_prefix_text_parts"] = ()
        branch["client_consumed_prefix_text"] = ""
    transition_tokens_appended = 0
    if private_handle is None and response_handle is None:
        requested_budget = int(think_kwargs["max_new_tokens"])
        requested_extension = int(
            think_kwargs.get("sentence_completion_max_tokens", 0)
            if think_kwargs.get("graceful_sentence_completion") else 0
        )
        if continuation_budget_used is not None:
            response_budget_remaining = (
                DEFAULT_CONTINUATION_GENERATED_BUDGET - continuation_budget_used
            )
            requested_budget = min(requested_budget, response_budget_remaining)
            requested_extension = min(
                requested_extension,
                max(0, response_budget_remaining - requested_budget),
            )
        preflight_deadline = time.monotonic() + 2.0
        while True:
            try:
                decode_budget_reservation = worker.reserve_response_decode_budget(
                    session.session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    requested_tokens=requested_budget,
                    transition_tokens=len(private_transition_ids),
                    sentence_completion_tokens=requested_extension,
                ).result(timeout=5.0)
                break
            except RuntimeError as exc:
                if str(exc) not in {
                    "response decode preflight requires a completed listener frame watermark",
                    "response decode preflight cannot overlap a shared listener tick",
                    "unified native operation lost its generation/cache-position compare-and-swap",
                } or time.monotonic() >= preflight_deadline:
                    raise
                if decode_cancelled():
                    raise asyncio.CancelledError from exc
                latest = worker.session_state(session.session_id).result(timeout=5.0)
                if int(latest.generation_id) != listener_generation_id:
                    raise RuntimeError("unified Thinker preflight lost its listener generation") from exc
                if int(branch.get("native_token_append_count", 0)) != 0:
                    if int(latest.cache_position) != cache_cursor:
                        raise RuntimeError(
                            "unified Thinker preflight observed an untracked cache append"
                        ) from exc
                else:
                    cache_cursor = int(latest.cache_position)
                    branch["cache_position"] = cache_cursor
                    branch["turn_start_cache_position"] = cache_cursor
                    branch["spoken_prefix_base_cursor"] = cache_cursor
                    branch["spoken_prefix_exclusive_cursor"] = cache_cursor
                    branch["client_consumed_prefix_exclusive_cursor"] = cache_cursor
                time.sleep(0.002)
        if decode_cancelled():
            release_decode_budget_reservation()
            raise asyncio.CancelledError
        think_kwargs["max_new_tokens"] = int(decode_budget_reservation["effective_tokens"])
        think_kwargs["sentence_completion_max_tokens"] = int(
            decode_budget_reservation["sentence_completion_tokens"]
        )
        think_kwargs["_token_limit_effective"] = int(
            decode_budget_reservation["effective_tokens"]
        )
        route_cap_reason = think_kwargs.get("_token_limit_cap_reason")
        kv_cap_reason = decode_budget_reservation.get("cap_reason")
        cap_reasons = []
        if isinstance(route_cap_reason, str) and route_cap_reason:
            cap_reasons.append(route_cap_reason)
        if isinstance(kv_cap_reason, str) and kv_cap_reason and kv_cap_reason not in cap_reasons:
            cap_reasons.append(kv_cap_reason)
        think_kwargs["_token_limit_cap_reason"] = (
            "+".join(cap_reasons) if cap_reasons else None
        )
        think_kwargs["_decode_budget_reservation"] = dict(decode_budget_reservation)
    token_timings: dict[str, Any] = {}
    # Unified-native decode submits token work to its owning queue and does not
    # acquire the shared Python Thinker/Gemma model lock in this call chain.
    # Queue/Future wait is a different measurement and is reported separately.
    thinker_metrics: dict[str, Any] = {
        "thinker_decode_lock_wait_ms": 0.0,
        "thinker_decode_lock_acquire_count": 0,
        "thinker_decode_lock_wait_provenance": "native_no_python_decode_lock",
    }
    native_profile: dict[str, Any] = {
        "clock": "server monotonic_ns",
        "first_packet_published_clock": (
            "legacy token0 packet callback completion on server monotonic_ns"
        ),
        "qwen_prefill_clock": "server monotonic_ns",
        "policy_decision_at_ns": branch.get("policy_decision_at_ns"),
        "queue_open_requested_ns": branch.get("queue_open_requested_ns"),
        "queue_open_ready_ns": branch.get("queue_open_ready_ns"),
        "append_steps": [],
        "sampling_steps": [],
    }
    native_profile_lock = threading.Lock()
    branch["unified_native_profile"] = native_profile
    branch["unified_native_profile_lock"] = native_profile_lock
    session_handoff_state: dict[str, Any] = {"target_branch": None}

    def mirror_adopted_branch_frontier() -> None:
        target = session_handoff_state.get("target_branch")
        if not isinstance(target, dict):
            return
        target_lock = target.get("spoken_prefix_lock")
        if target_lock is None:
            raise RuntimeError("adopted response branch lacks its spoken-prefix lock")
        with session._generation_lock:
            with target_lock:
                target["cache_position"] = int(branch["cache_position"])
                target["native_token_append_count"] = int(
                    branch.get("native_token_append_count", 0)
                )
                target["generated_token_count"] = int(
                    branch.get("_generated_budget_used", 0)
                )
                target["_generated_budget_used"] = int(
                    branch.get("_generated_budget_used", 0)
                )
                unappended_terminal_count = int(
                    branch.get("unappended_terminal_token_count", 0)
                )
                target["unappended_terminal_token_count"] = unappended_terminal_count
                target["sampled_generated_token_count"] = int(
                    target["generated_token_count"]
                ) + unappended_terminal_count
                target["response_decode_handle"] = response_handle_ref.get("value")
                target["response_decode_purpose"] = "active_response"

    def adopt_append_cursor(result: Any, token_ids: Sequence[int]) -> None:
        nonlocal cache_cursor, transition_tokens_appended
        token_count = len(token_ids)
        result_cursor = getattr(result, "cache_position", None)
        first_append = int(branch.get("native_token_append_count", 0)) == 0
        if result_cursor is None:
            committed_cursor = cache_cursor + token_count
        else:
            committed_cursor = int(result_cursor)
        if first_append:
            actual_start_cursor = committed_cursor - token_count
            if actual_start_cursor < cache_cursor:
                raise RuntimeError("unified text append returned a cursor before its request")
            rebase_delta = actual_start_cursor - cache_cursor
            if rebase_delta:
                for key in (
                    "turn_start_cache_position",
                    "spoken_prefix_base_cursor",
                    "spoken_prefix_exclusive_cursor",
                ):
                    value = branch.get(key)
                    if isinstance(value, int):
                        branch[key] = value + rebase_delta
        cache_cursor = committed_cursor
        branch["cache_position"] = cache_cursor
        branch["native_token_append_count"] = int(
            branch.get("native_token_append_count", 0)
        ) + token_count
        if transition_tokens_appended < len(private_transition_ids):
            expected_transition = private_transition_ids[
                transition_tokens_appended : transition_tokens_appended + token_count
            ]
            actual_tokens = tuple(int(value) for value in token_ids)
            if actual_tokens != expected_transition:
                raise RuntimeError(
                    "unified native append diverged from the pending Thinker transition: "
                    f"transition_offset={transition_tokens_appended}, "
                    f"expected={expected_transition}, actual={actual_tokens}"
                )
            transition_tokens_appended += token_count
            if transition_tokens_appended == len(private_transition_ids):
                branch["spoken_prefix_base_cursor"] = committed_cursor
                branch["spoken_prefix_transition_cursor"] = committed_cursor
                if not tuple(branch.get("client_consumed_segments", ())):
                    branch["spoken_prefix_exclusive_cursor"] = committed_cursor
                    branch["client_consumed_prefix_exclusive_cursor"] = committed_cursor
        elif continuation_budget_used is not None:
            branch["_generated_budget_used"] = int(
                branch.get("_generated_budget_used", continuation_budget_used)
            ) + token_count
            branch["generated_token_count"] = int(
                branch.get("generated_token_count", 0)
            ) + token_count
        mirror_adopted_branch_frontier()

    def publish_profile() -> None:
        if thinker_metrics:
            token_timings["thinker_runtime_metrics"] = dict(thinker_metrics)
        with native_profile_lock:
            token_timings["unified_native_profile"] = {
                **native_profile,
                "append_steps": [dict(step) for step in native_profile["append_steps"]],
                "sampling_steps": [dict(step) for step in native_profile["sampling_steps"]],
            }
        callback = request.get("_on_thinker_metrics")
        if callable(callback):
            callback(dict(token_timings))

    async def forward_native_tokens(token_ids: list[int]):
        nonlocal cache_cursor, private_transition_remaining, private_transition_offset
        if decode_cancelled():
            raise asyncio.CancelledError
        first_generated_forward = (
            (
                private_transition_remaining == 0
                if private_handle is not None
                else transition_tokens_appended >= len(private_transition_ids)
            )
            and "token0_forward_submit_ns" not in native_profile
        )
        retry_deadline = time.monotonic() + 2.0
        append_call_ns = time.monotonic_ns()
        if private_handle is not None:
            is_private_transition = private_transition_remaining > 0
            next_transition_offset = private_transition_offset
            if is_private_transition:
                next_transition_offset = _validate_private_transition_chunk(
                    private_transition_ids,
                    private_transition_offset,
                    token_ids,
                    source_cache_position=int(private_handle.source_cache_position),
                    current_cache_position=cache_cursor,
                )
                owner_future = worker.append_private_transition_tokens(
                    private_handle,
                    token_ids,
                    expected_cache_position=cache_cursor,
                )
            else:
                if private_transition_offset != len(private_transition_ids):
                    raise RuntimeError("private decode sampled before the original Thinker transition completed")
                token0_submit_ns = time.monotonic_ns()
                if first_generated_forward:
                    with native_profile_lock:
                        native_profile.setdefault("token0_forward_submit_ns", token0_submit_ns)
                        native_profile.setdefault("token0_append_request", {
                            "response_id": branch.get("response_id"),
                            "generation_id": generation_id,
                            "listener_generation_id": listener_generation_id,
                            "q_len": len(token_ids),
                            "expected_cache_position": cache_cursor,
                        })
                owner_future = worker.append_private_tokens(
                    private_handle,
                    token_ids,
                    expected_cache_position=cache_cursor,
                )
            wrapped = asyncio.wrap_future(owner_future)
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                result = await asyncio.shield(wrapped)
                received_ns = time.monotonic_ns()
                cache_cursor = int(result.cache_position)
                branch["cache_position"] = cache_cursor
                if is_private_transition:
                    private_transition_offset = next_transition_offset
                    private_transition_remaining = len(private_transition_ids) - private_transition_offset
                elif first_generated_forward:
                    with native_profile_lock:
                        native_profile.setdefault("token0_forward_return_ns", received_ns)
                        native_profile.setdefault(
                            "token0_forward_owner_future_wait_ns",
                            max(0, received_ns - token0_submit_ns),
                        )
                raise
            cache_cursor = int(result.cache_position)
            branch["cache_position"] = cache_cursor
            if is_private_transition:
                private_transition_offset = next_transition_offset
                private_transition_remaining = len(private_transition_ids) - private_transition_offset
                if private_transition_remaining == 0:
                    branch["spoken_prefix_base_cursor"] = cache_cursor
                    branch["spoken_prefix_transition_cursor"] = cache_cursor
                    if not tuple(branch.get("client_consumed_segments", ())):
                        branch["spoken_prefix_exclusive_cursor"] = cache_cursor
            received_ns = time.monotonic_ns()
            if first_generated_forward and not is_private_transition:
                with native_profile_lock:
                    native_profile.setdefault("token0_forward_return_ns", received_ns)
                    native_profile.setdefault(
                        "token0_forward_owner_future_wait_ns",
                        max(0, received_ns - token0_submit_ns),
                    )
            step_profile: dict[str, Any] = {
                "phase": "private_transition" if is_private_transition else "private_generated_token",
                "logical_token_count": len(token_ids),
                "append_call_ns": append_call_ns,
                "future_result_received_ns": received_ns,
                "owner_future_wait_ns": max(0, received_ns - append_call_ns),
                "private_cache_position": cache_cursor,
            }
            worker_timings = getattr(result, "timings_ns", None)
            if isinstance(worker_timings, Mapping):
                step_profile.update({str(key): int(value) for key, value in worker_timings.items()})
            pair = _consume_unified_native_text_append_result(result, timings_ns=step_profile)
            step_profile["consumer_returned_ns"] = time.monotonic_ns()
            native_profile["append_steps"].append(step_profile)
            return pair
        current_response_handle = response_handle_ref.get("value")
        if current_response_handle is not None:
            is_prompt = transition_tokens_appended < len(private_transition_ids)
            token0_submit_ns = time.monotonic_ns()
            if first_generated_forward:
                with native_profile_lock:
                    native_profile.setdefault("token0_forward_submit_ns", token0_submit_ns)
                    native_profile.setdefault("token0_append_request", {
                        "response_id": branch.get("response_id"),
                        "generation_id": generation_id,
                        "listener_generation_id": listener_generation_id,
                        "revision_id": expected_branch_revision_id,
                        "q_len": len(token_ids),
                        "expected_cache_position": cache_cursor,
                    })
            owner_future = worker.append_response_decode(
                current_response_handle,
                token_ids,
                expected_cache_position=cache_cursor,
                generated=not is_prompt,
            )
            wrapped = asyncio.wrap_future(owner_future)
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                result = await asyncio.shield(wrapped)
                if decode_cancelled():
                    raise
                adopt_append_cursor(result, token_ids)
                raise
            if decode_cancelled():
                raise asyncio.CancelledError
            adopt_append_cursor(result, token_ids)
            received_ns = time.monotonic_ns()
            if first_generated_forward:
                with native_profile_lock:
                    native_profile.setdefault("token0_forward_return_ns", received_ns)
                    native_profile.setdefault(
                        "token0_forward_owner_future_wait_ns",
                        max(0, received_ns - token0_submit_ns),
                    )
            step_profile: dict[str, Any] = {
                "phase": "response_prompt" if is_prompt else "response_generated_token",
                "logical_token_count": len(token_ids),
                "append_call_ns": append_call_ns,
                "future_result_received_ns": received_ns,
                "owner_future_wait_ns": max(0, received_ns - append_call_ns),
                "response_revision_id": expected_branch_revision_id,
            }
            worker_timings = getattr(result, "timings_ns", None)
            if isinstance(worker_timings, Mapping):
                step_profile.update({str(key): int(value) for key, value in worker_timings.items()})
            pair = _consume_unified_native_text_append_result(result, timings_ns=step_profile)
            step_profile["consumer_returned_ns"] = time.monotonic_ns()
            native_profile["append_steps"].append(step_profile)
            return pair
        while True:
            prepared_protocol = None
            if (
                branch.get("response_protocol") == CONTROLLER_GUIDED_SAY_V1
                and int(branch.get("native_token_append_count", 0)) == 0
                and tuple(int(value) for value in token_ids) == private_transition_ids
            ):
                prepared_protocol = CONTROLLER_GUIDED_SAY_V1
            append_kwargs = {
                "expected_generation_id": listener_generation_id,
                "expected_cache_position": cache_cursor,
                "allow_initial_cursor_rebase": (
                    int(branch.get("native_token_append_count", 0)) == 0
                ),
                "append_scope_id": str(branch.get("response_id") or "") or None,
                "response_cancel_event": response_cancel_event,
            }
            if decode_budget_reservation is not None:
                append_kwargs["consume_reserved_tokens"] = len(token_ids)
            if prepared_protocol is not None:
                append_kwargs["prepared_protocol"] = prepared_protocol
            token0_submit_ns = time.monotonic_ns()
            if first_generated_forward:
                with native_profile_lock:
                    native_profile.setdefault("token0_forward_submit_ns", token0_submit_ns)
                    native_profile.setdefault("token0_append_request", {
                        "response_id": branch.get("response_id"),
                        "generation_id": generation_id,
                        "listener_generation_id": listener_generation_id,
                        "q_len": len(token_ids),
                        "expected_cache_position": cache_cursor,
                        "append_scope_id": append_kwargs["append_scope_id"],
                    })
            owner_future = worker.append_text_tokens(
                session.session_id,
                token_ids,
                **append_kwargs,
            )
            wrapped = asyncio.wrap_future(owner_future)
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                if response_cancel_event is not None:
                    response_cancel_event.set()
                try:
                    result = await asyncio.shield(wrapped)
                except RuntimeError as append_error:
                    if str(append_error) == "text append response canceled before native KV mutation":
                        raise asyncio.CancelledError from append_error
                    raise
                adopt_append_cursor(result, token_ids)
                received_ns = time.monotonic_ns()
                if first_generated_forward:
                    with native_profile_lock:
                        native_profile.setdefault("token0_forward_return_ns", received_ns)
                        native_profile.setdefault(
                            "token0_forward_owner_future_wait_ns",
                            max(0, received_ns - token0_submit_ns),
                        )
                raise
            except RuntimeError as exc:
                if str(exc) == "text append response canceled before native KV mutation":
                    raise asyncio.CancelledError from exc
                if str(exc) == "text append requires completed listener frame watermark":
                    if time.monotonic() >= retry_deadline or (
                        decode_cancelled()
                    ):
                        if decode_cancelled():
                            raise asyncio.CancelledError from exc
                        raise
                    latest = worker.session_state(session.session_id).result(timeout=5.0)
                    if int(latest.generation_id) != listener_generation_id:
                        raise RuntimeError(
                            "unified Thinker branch lost its listener generation"
                        ) from exc
                    latest_cursor = int(latest.cache_position)
                    latest_valid_history_tokens = int(
                        getattr(latest, "valid_history_tokens", latest_cursor)
                    )
                    initial_append = int(branch.get("native_token_append_count", 0)) == 0
                    if (
                        latest_cursor < cache_cursor
                        or (not initial_append and latest_cursor != cache_cursor)
                        or latest_valid_history_tokens != latest_cursor
                    ):
                        raise RuntimeError(
                            "unified listener cache cursor changed incoherently while waiting "
                            "for a frame watermark: "
                            f"expected_cache_position={cache_cursor}, "
                            f"actual_cache_position={latest_cursor}, "
                            f"actual_valid_history_tokens={latest_valid_history_tokens}"
                        ) from exc
                    await asyncio.sleep(0.002)
                    continue
                raise
            adopt_append_cursor(result, token_ids)
            received_ns = time.monotonic_ns()
            if first_generated_forward:
                with native_profile_lock:
                    native_profile.setdefault("token0_forward_return_ns", received_ns)
                    native_profile.setdefault(
                        "token0_forward_owner_future_wait_ns",
                        max(0, received_ns - token0_submit_ns),
                    )
            step_profile: dict[str, Any] = {
                "phase": "suffix" if not branch.get("first_logits_ready") else "generated_token",
                "logical_token_count": len(token_ids),
                "append_call_ns": append_call_ns,
                "future_result_received_ns": received_ns,
                "owner_future_wait_ns": max(0, received_ns - append_call_ns),
            }
            worker_timings = getattr(result, "timings_ns", None)
            if isinstance(worker_timings, Mapping):
                step_profile.update({str(key): int(value) for key, value in worker_timings.items()})
                owner_started_ns = worker_timings.get("owner_started_ns")
                if isinstance(owner_started_ns, int):
                    step_profile["owner_queue_wait_ns"] = max(0, owner_started_ns - append_call_ns)
                    if first_generated_forward:
                        with native_profile_lock:
                            native_profile.setdefault(
                                "token0_forward_owner_queue_wait_ns",
                                max(0, owner_started_ns - token0_submit_ns),
                            )
            transition_cache_hit = getattr(result, "prepared_transition_cache_hit", None)
            if transition_cache_hit is not None:
                step_profile["prepared_transition_cache_hit"] = bool(transition_cache_hit)
            pair = _consume_unified_native_text_append_result(
                result,
                timings_ns=step_profile,
            )
            step_profile["consumer_returned_ns"] = time.monotonic_ns()
            native_profile["append_steps"].append(step_profile)
            return pair

    async def on_first_logits_ready(event: Any) -> None:
        branch["first_logits_ready"] = True
        token_timings["thinker_first_logits_ready_at_ns"] = int(event.ready_ns)
        native_profile["first_logits_callback_ns"] = time.monotonic_ns()
        decision_ns = native_profile.get("policy_decision_at_ns")
        if isinstance(decision_ns, int):
            first_logits_ms = max(0.0, (int(event.ready_ns) - decision_ns) / 1_000_000.0)
            token_timings["first_logits_after_decision_ms"] = first_logits_ms
            native_profile["first_logits_after_decision_ms"] = first_logits_ms

    async def on_native_token(
        token_id: int,
        _hidden: torch.Tensor,
        timing: Any,
    ) -> None:
        record = request.get("_on_thinker_token_id")
        if callable(record):
            record(int(token_id))
        if "first_thinker_token_at_ns" not in token_timings:
            token_timings["first_thinker_token_at_ns"] = time.monotonic_ns()
            token_timings["first_thinker_token_hidden_ready_at_ns"] = int(
                timing.token_hidden_ready_ns
            )
            token_timings["first_thinker_token_sampled_at_ns"] = int(timing.sampled_ns)
            token_timings["first_thinker_token_forward_started_at_ns"] = int(
                timing.token_forward_started_ns
            )
            decision_ns = native_profile.get("policy_decision_at_ns")
            if isinstance(decision_ns, int):
                sampled_ms = max(0.0, (int(timing.sampled_ns) - decision_ns) / 1_000_000.0)
                packet_ready_ms = max(
                    0.0,
                    (int(timing.token_hidden_ready_ns) - decision_ns) / 1_000_000.0,
                )
                token_timings["first_token_sampled_ms"] = sampled_ms
                token_timings["first_packet_hidden_ready_ms"] = packet_ready_ms
                native_profile["first_token_sampled_ms"] = sampled_ms
                native_profile["first_packet_hidden_ready_ms"] = packet_ready_ms
                if not bool(think_kwargs.get("anticipate_first_token")):
                    # Legacy TTFT is measured when the first post-forward token
                    # reaches its publication callback.
                    token_timings["ttft_ms"] = packet_ready_ms
                    native_profile["ttft_ms"] = packet_ready_ms
            native_profile["first_token_callback_ns"] = int(
                token_timings["first_thinker_token_at_ns"]
            )
            # This callback exposes the post-forward token to the response
            # owner; the websocket action is emitted later by the protocol.
            native_profile["first_token_model_callback_ns"] = time.monotonic_ns()
            publish_profile()

    async def on_first_token(token: Any) -> None:
        published_ns = getattr(token, "packet_published_ns", None)
        if isinstance(published_ns, int):
            token_timings["first_packet_published_at_ns"] = int(published_ns)
            token_timings["ttft_clock"] = (
                "legacy token0 packet callback completion on server monotonic_ns"
            )
            decision_ns = native_profile.get("policy_decision_at_ns")
            if isinstance(decision_ns, int):
                ttft_ms = max(0.0, (int(published_ns) - decision_ns) / 1_000_000.0)
                token_timings["ttft_ms"] = ttft_ms
                native_profile["first_packet_published_at_ns"] = int(published_ns)
                native_profile["ttft_clock"] = (
                    "legacy token0 packet callback completion on server monotonic_ns"
                )
                native_profile["ttft_ms"] = ttft_ms
        callback = think_kwargs.get("on_first_token")
        if callable(callback):
            result = callback(token)
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                await result

    raw_candidate_guard = think_kwargs.get("on_candidate_token")
    candidate_guard_trace: list[dict[str, Any]] = []
    if callable(raw_candidate_guard):
        def observed_candidate_guard(token_id: int, text: str, terminal: bool) -> bool:
            try:
                accepted = bool(raw_candidate_guard(token_id, text, terminal))
            except BaseException as exc:
                candidate_guard_trace.append({
                    "token_id": int(token_id),
                    "text": str(text)[:120],
                    "terminal": bool(terminal),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise
            candidate_guard_trace.append({
                "token_id": int(token_id),
                "text": str(text)[:120],
                "terminal": bool(terminal),
                "accepted": accepted,
            })
            return accepted

        think_kwargs["on_candidate_token"] = observed_candidate_guard

    try:
        sampler = engine.build_thinker_sampler(
            temperature=float(think_kwargs["temperature"]),
            top_p=float(think_kwargs["top_p"]),
            repetition_penalty=float(request.get("repetition_penalty", 1.15)),
            min_new_tokens=int(think_kwargs["min_new_tokens"]),
            action=None,
            runtime_metrics=thinker_metrics,
            on_generation_start=think_kwargs.get("on_generation_start"),
            on_generated_token=(
                None
                if bool(think_kwargs.get("anticipate_first_token"))
                else think_kwargs.get("on_generated_token")
            ),
            on_decoded_text_flush=think_kwargs.get("on_decoded_text_flush"),
            on_candidate_token=think_kwargs.get("on_candidate_token"),
            on_candidate_fallback=think_kwargs.get("on_candidate_fallback"),
            on_action_logits_probe=think_kwargs.get("on_action_logits_probe"),
            state=session,
            generation_id=(int(generation_id) if generation_id is not None else None),
            generator=think_kwargs.get("_sampler_generator"),
        )
    except BaseException as exc:
        release_decode_budget_reservation(exc)
        raise

    def _append_sampling_profile(
        stage: str,
        started_ns: int,
        *,
        candidate: Any = None,
        history_length: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        finished_ns = time.monotonic_ns()
        record: dict[str, Any] = {
            "stage": stage,
            "started_ns": int(started_ns),
            "finished_ns": finished_ns,
            "duration_ns": max(0, finished_ns - int(started_ns)),
        }
        if history_length is not None:
            record["generated_history_length"] = int(history_length)
        if isinstance(candidate, (tuple, list)) and len(candidate) == 3:
            record["candidate_id"] = int(candidate[1])
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        native_profile["sampling_steps"].append(record)
        if error is not None:
            publish_profile()

    raw_sample_token = sampler.sample_token

    def profiled_sample_token(logits: torch.Tensor, generated_ids: Sequence[int]):
        started_ns = time.monotonic_ns()
        try:
            candidate = raw_sample_token(logits, generated_ids)
        except BaseException as exc:
            _append_sampling_profile(
                "sample_token",
                started_ns,
                history_length=len(generated_ids),
                error=exc,
            )
            raise
        _append_sampling_profile(
            "sample_token",
            started_ns,
            candidate=candidate,
            history_length=len(generated_ids),
        )
        return candidate

    sampler.sample_token = profiled_sample_token
    raw_sample_initial = sampler.sample_initial

    def profiled_sample_initial(logits: torch.Tensor, generated_ids: Sequence[int] = ()):
        started_ns = time.monotonic_ns()
        try:
            candidate = raw_sample_initial(logits, generated_ids)
        except BaseException as exc:
            _append_sampling_profile(
                "sample_initial",
                started_ns,
                history_length=len(generated_ids),
                error=exc,
            )
            raise
        _append_sampling_profile(
            "sample_initial",
            started_ns,
            candidate=candidate,
            history_length=len(generated_ids),
        )
        return candidate

    sampler.sample_initial = profiled_sample_initial
    raw_select_valid_candidate = sampler.select_valid_candidate

    def profiled_select_valid_candidate(candidate: Any, **kwargs: Any):
        candidate_guard_trace.clear()
        started_ns = time.monotonic_ns()
        generated_ids = kwargs.get("generated_ids", ())
        try:
            selected = raw_select_valid_candidate(candidate, **kwargs)
        except BaseException as exc:
            diagnostic: dict[str, Any] = {
                "stage": "select_valid_candidate",
                "started_ns": int(started_ns),
                "finished_ns": time.monotonic_ns(),
                "generated_ids": [int(value) for value in generated_ids],
                "guard_outcomes": list(candidate_guard_trace),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(candidate, (tuple, list)) and len(candidate) == 3:
                diagnostic["sampled_candidate_id"] = int(candidate[1])
                scores = candidate[2]
                if isinstance(scores, torch.Tensor) and scores.numel():
                    count = min(64, int(scores.numel()))
                    top_scores, top_ids = torch.topk(scores.reshape(-1), k=count)
                    diagnostic["top64_ids"] = [int(value) for value in top_ids.detach().cpu().tolist()]
                    diagnostic["top64_scores"] = [
                        float(value) for value in top_scores.detach().float().cpu().tolist()
                    ]
            snapshot_callback = request.get("_audio_response_v2_diagnostic_snapshot")
            if callable(snapshot_callback):
                try:
                    diagnostic["protocol_state"] = snapshot_callback()
                except BaseException as snapshot_error:
                    diagnostic["protocol_state_error"] = (
                        f"{type(snapshot_error).__name__}: {snapshot_error}"
                    )
            native_profile["first_guard_failure"] = diagnostic
            thinker_metrics.update({
                key: value for key, value in sampler.runtime_metrics.items()
                if isinstance(key, str)
            })
            finished_ns = int(diagnostic["finished_ns"])
            diagnostic["duration_ns"] = max(0, finished_ns - started_ns)
            publish_profile()
            raise
        _append_sampling_profile(
            "select_valid_candidate",
            started_ns,
            candidate=selected,
            history_length=len(generated_ids),
        )
        return selected

    sampler.select_valid_candidate = profiled_select_valid_candidate
    try:
        decoder = UnifiedNativeDecoder(
            getattr(engine, "thinker", None),
            forward_tokens=forward_native_tokens,
            sampler=sampler,
            monotonic_ns=time.monotonic_ns,
            eos_token_ids=sampler.eos_token_ids,
        )
    except BaseException as exc:
        release_decode_budget_reservation(exc)
        raise

    token_packet_callback = think_kwargs.get("on_token_packet")
    token_committed_callback = think_kwargs.get("on_token_committed")

    async def timed_token_packet_callback(generated: Any) -> None:
        if (
            response_handle is not None
            and str(branch.get("revision_id", "r0")) != expected_branch_revision_id
        ):
            raise asyncio.CancelledError
        token_index = int(getattr(generated, "token_index", -1))
        if token_index == 0:
            _record_unified_native_profile_timestamp(
                branch, "token0_packet_callback_entry_ns", time.monotonic_ns()
            )
        try:
            if callable(token_packet_callback):
                result = token_packet_callback(generated)
                if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                    await result
        finally:
            if token_index == 0:
                _record_unified_native_profile_timestamp(
                    branch, "token0_packet_callback_exit_ns", time.monotonic_ns()
                )

    async def timed_token_committed_callback(generated: Any) -> None:
        if (
            response_handle is not None
            and str(branch.get("revision_id", "r0")) != expected_branch_revision_id
        ):
            raise asyncio.CancelledError
        token_index = int(getattr(generated, "token_index", -1))
        if token_index == 0:
            _record_unified_native_profile_timestamp(
                branch, "token0_committed_callback_entry_ns", time.monotonic_ns()
            )
        try:
            if callable(token_committed_callback):
                result = token_committed_callback(generated)
                if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                    await result
        finally:
            if token_index == 0:
                _record_unified_native_profile_timestamp(
                    branch, "token0_committed_callback_exit_ns", time.monotonic_ns()
                )

    text_flush_callback = think_kwargs.get("on_decoded_text_flush")

    async def timed_text_flush_callback(suffix: str, generated: Any) -> None:
        if (
            response_handle is not None
            and str(branch.get("revision_id", "r0")) != expected_branch_revision_id
        ):
            return
        if callable(text_flush_callback):
            result = text_flush_callback(str(suffix), generated)
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                await result

    decode_error: BaseException | None = None
    try:
        decode_options = {
            "stop_check": think_kwargs.get("stop_check"),
            "first_logits_ready": on_first_logits_ready,
            "on_token": on_native_token,
            "first_token": on_first_token,
            "on_token_packet": (
                timed_token_packet_callback if callable(token_packet_callback) else None
            ),
            "on_token_committed": (
                timed_token_committed_callback
                if callable(token_committed_callback) else None
            ),
            "anticipate_first_token": bool(think_kwargs.get("anticipate_first_token")),
            "graceful_sentence_completion": bool(
                think_kwargs.get("graceful_sentence_completion")
            ),
            "sentence_completion_max_tokens": int(
                think_kwargs.get("sentence_completion_max_tokens", 0)
            ),
            "producer_limit_check": think_kwargs.get("producer_limit_check"),
            "on_text_flush": (
                timed_text_flush_callback if callable(text_flush_callback) else None
            ),
        }
        def finish_decoded(decoded: Any) -> dict[str, Any]:
            publish_profile()
            if decoded.termination_reason == "eos" and decoded.terminal_token_id is not None:
                record = request.get("_on_thinker_token_id")
                if callable(record):
                    record(int(decoded.terminal_token_id))
            generated_ids = [int(item.token_id) for item in decoded.generated_tokens]
            generated_hidden = [item.hidden for item in decoded.generated_tokens]
            if generated_hidden:
                hidden = torch.cat(generated_hidden, dim=1)
            else:
                thinker_config = getattr(getattr(engine, "thinker", None), "config", None)
                text_config = (
                    thinker_config.get_text_config(decoder=True)
                    if callable(getattr(thinker_config, "get_text_config", None))
                    else thinker_config
                )
                hidden_size = int(getattr(text_config, "hidden_size", 1536))
                hidden = torch.empty(
                    (1, 0, hidden_size),
                    device=getattr(engine, "device", "cpu"),
                    dtype=getattr(engine, "dtype", torch.float32),
                )
            suffix_text = str(engine.tokenizer.decode(generated_ids, skip_special_tokens=True))
            response_ids = (
                [*continuation_prefix_ids, *generated_ids]
                if continuation_prefix_ids is not None else generated_ids
            )
            protocol_text = str(engine.tokenizer.decode(response_ids, skip_special_tokens=True))
            terminal_unappended_count = int(
                decoded.termination_reason == "eos" and decoded.terminal_token_id is not None
            )
            branch["generated_token_count"] = int(decoded.committed_count)
            branch["sampled_generated_token_count"] = (
                int(decoded.committed_count) + terminal_unappended_count
            )
            branch["unappended_terminal_token_count"] = terminal_unappended_count
            branch["terminal_reason"] = str(decoded.termination_reason)
            result = {
                "text": protocol_text,
                "protocol_text": protocol_text,
                "suffix_text": suffix_text,
                "gen_ids": generated_ids,
                "hidden": hidden,
                "hidden_prompt_len": 0,
                "prompt_len": 0,
                "past_key_values": None,
                "terminal_reason": decoded.termination_reason,
                "unappended_terminal_token_count": terminal_unappended_count,
                "cancelled": decoded.termination_reason == "cancelled",
                "runtime_metrics": {
                    **thinker_metrics,
                    "unified_native_profile": {
                        **native_profile,
                        "append_steps": [dict(step) for step in native_profile["append_steps"]],
                        "sampling_steps": [dict(step) for step in native_profile["sampling_steps"]],
                    },
                    "unified_native_cache": {
                        "listener_generation_id": listener_generation_id,
                        "turn_start_cache_position": int(branch["turn_start_cache_position"]),
                        "cache_position_after_decode": cache_cursor,
                        "transition_token_count": int(decoded.transition_token_count),
                        "committed_generated_tokens": int(decoded.committed_count),
                        "sampled_generated_tokens": int(decoded.sampled_count),
                        "unappended_terminal_tokens": terminal_unappended_count,
                        "terminal_reason": decoded.termination_reason,
                        "owner_action": decoded.owner_action,
                    },
                },
            }
            target = session_handoff_state.get("target_branch")
            promoted_controller = session_handoff_state.get("controller")
            if isinstance(target, dict):
                with session._generation_lock:
                    target_lock = target.get("spoken_prefix_lock")
                    if target_lock is not None:
                        with target_lock:
                            target["terminal_reason"] = str(decoded.termination_reason)
                            target["generated_token_count"] = int(
                                target.get("_generated_budget_used", 0)
                            )
                            target["unappended_terminal_token_count"] = terminal_unappended_count
                            target["sampled_generated_token_count"] = (
                                int(target["generated_token_count"])
                                + terminal_unappended_count
                            )
                            target["unappended_terminal_token_count"] = terminal_unappended_count
                            if (
                                isinstance(promoted_controller, dict)
                                and promoted_controller.get("promoted")
                                and target.get("_adopted_native_session_controller")
                                is promoted_controller
                                and str(target.get("revision_id", ""))
                                == str(promoted_controller.get("revision_id", ""))
                            ):
                                target["_last_adopted_think"] = result
            return result

        revisioned_decode = bool(
            response_handle is not None
            and branch.get("revisioned_live_say")
            and branch.get("response_protocol") == CONTROLLER_GUIDED_SAY_V1
        )
        candidate_ready_when = think_kwargs.get("_revision_candidate_ready_when")
        if revisioned_decode:
            if continuation_prefix_ids is not None:
                assert continuation_logical_offset is not None
                assert continuation_budget_used is not None
            session_decoder = decoder.start_session(
                transition_token_ids=transition_token_ids,
                max_new_tokens=int(think_kwargs["max_new_tokens"]),
                first_logits_ready=on_first_logits_ready,
                first_token=on_first_token,
                on_token=on_native_token,
                on_token_packet=(
                    timed_token_packet_callback if callable(token_packet_callback) else None
                ),
                on_token_committed=(
                    timed_token_committed_callback
                    if callable(token_committed_callback) else None
                ),
                on_text_flush=(
                    timed_text_flush_callback if callable(text_flush_callback) else None
                ),
                anticipate_first_token=bool(think_kwargs.get("anticipate_first_token")),
                graceful_sentence_completion=bool(
                    think_kwargs.get("graceful_sentence_completion")
                ),
                sentence_completion_max_tokens=int(
                    think_kwargs.get("sentence_completion_max_tokens", 0)
                ),
                continuation_prefix_token_ids=continuation_prefix_ids or (),
                logical_source_offset=(continuation_logical_offset or 0),
                native_handle=response_handle,
                native_cursor=cache_cursor,
            )
            controller: dict[str, Any] = {
                "session": session_decoder,
                "handle_ref": response_handle_ref,
                "handoff_state": session_handoff_state,
                "promoted": False,
                "source_token_index_offset": 0,
                "done": False,
                "final_result": None,
                "fatal_error": None,
                "cancel_error": None,
                "last_step": None,
                "lifecycle_condition": threading.Condition(),
                "advance_owner": None,
                "completion_owner": None,
                "revision_id": str(branch.get("revision_id", "r0")),
            }
            session_handoff_state["controller"] = controller
            controller_sink = think_kwargs.get("_revision_native_session_controller_sink")
            if isinstance(controller_sink, dict):
                controller_sink["value"] = controller

            def advance_controller(ready_when: Any) -> Any:
                condition = controller["lifecycle_condition"]
                owner = threading.get_ident()
                with condition:
                    while controller.get("advance_owner") is not None:
                        if controller.get("advance_owner") == owner:
                            raise RuntimeError("native session controller advance is reentrant")
                        condition.wait()
                    while (
                        controller.get("completion_owner") is not None
                        and controller.get("completion_owner") != owner
                    ):
                        condition.wait()
                    fatal_error = controller.get("fatal_error")
                    if isinstance(fatal_error, BaseException):
                        raise fatal_error
                    if controller.get("done") and controller.get("last_step") is not None:
                        return controller["last_step"]
                    controller["advance_owner"] = owner
                try:
                    step = asyncio.run(session_decoder.advance(
                        max_committed_tokens=8,
                        ready_when=ready_when,
                        stop_check=decode_options["stop_check"],
                        producer_limit_check=decode_options["producer_limit_check"],
                    ))
                    controller["last_step"] = step
                    if step.status in {
                        NativeDecodeSessionStatus.DONE,
                        NativeDecodeSessionStatus.CANCELLED,
                    }:
                        result = finish_decoded(step.result)
                        if not isinstance(result, Mapping):
                            raise TypeError("decoded controller result must be a mapping")
                        with condition:
                            if not controller.get("done"):
                                controller["final_result"] = (
                                    result if isinstance(result, dict) else dict(result)
                                )
                                controller["done"] = True
                    return step
                except BaseException as exc:
                    terminal_status = getattr(session_decoder, "status", None)
                    if terminal_status is NativeDecodeSessionStatus.CANCELLED:
                        try:
                            terminal_step = asyncio.run(session_decoder.advance(
                                max_committed_tokens=8,
                                ready_when=lambda _progress: False,
                                stop_check=decode_options["stop_check"],
                                producer_limit_check=decode_options["producer_limit_check"],
                            ))
                            result = finish_decoded(terminal_step.result)
                            if not isinstance(result, Mapping):
                                raise TypeError("decoded controller result must be a mapping")
                            controller["last_step"] = terminal_step
                            with condition:
                                if not controller.get("done"):
                                    controller["final_result"] = (
                                        result if isinstance(result, dict) else dict(result)
                                    )
                                    controller["done"] = True
                                    controller["cancel_error"] = exc
                        except BaseException as finish_error:
                            with condition:
                                if not controller.get("done"):
                                    controller["fatal_error"] = (
                                        getattr(session_decoder, "fatal_error", None) or exc
                                    )
                                    controller["cleanup_error"] = finish_error
                                    controller["done"] = True
                    else:
                        fatal_error = getattr(session_decoder, "fatal_error", None) or exc
                        with condition:
                            if not controller.get("done"):
                                controller["fatal_error"] = fatal_error
                                controller["done"] = True
                    raise
                finally:
                    with condition:
                        controller["advance_owner"] = None
                        condition.notify_all()

            def run_controller_to_completion() -> dict[str, Any]:
                return _run_controller_to_completion(
                    controller,
                    advance_controller=advance_controller,
                    finish_decoded=finish_decoded,
                )

            controller["advance"] = advance_controller
            controller["run_to_completion"] = run_controller_to_completion
            controller["close"] = session_decoder.close

            if callable(candidate_ready_when):
                while True:
                    step = advance_controller(candidate_ready_when)
                    if step.status is NativeDecodeSessionStatus.READY:
                        progress = step.progress
                        controller["checkpoint_step"] = step
                        return {
                            "text": str(progress.stable_text),
                            "protocol_text": str(progress.stable_text),
                            "suffix_text": str(progress.generated_text),
                            "gen_ids": list(session_decoder.generated_token_ids),
                            "terminal_reason": None,
                            "cancelled": False,
                            "spoken_prefix_base_cursor": int(
                                branch.get("spoken_prefix_base_cursor", cache_cursor)
                            ),
                            "cache_position": int(branch.get("cache_position", cache_cursor)),
                            "transition_token_ids": tuple(
                                continuation_context_transition_ids
                            ),
                            "native_token_append_count": int(
                                branch.get("native_token_append_count", 0)
                            ),
                            "generated_token_count": int(
                                branch.get("generated_token_count", 0)
                            ),
                            "_native_session_controller": controller,
                        }
                    if step.status in {
                        NativeDecodeSessionStatus.DONE,
                        NativeDecodeSessionStatus.CANCELLED,
                    }:
                        return run_controller_to_completion()

            def handoff_adopted_session(
                error: BaseException | None = None,
            ) -> dict[str, Any] | None:
                if str(branch.get("revision_id", "r0")) == expected_branch_revision_id:
                    return None
                adopted_controller = branch.get("_adopted_native_session_controller")
                current_revision_id = str(branch.get("revision_id", "r0"))
                is_expected_cancel = isinstance(
                    error,
                    (asyncio.CancelledError, LiveSayCancelled),
                )
                if error is not None and not is_expected_cancel:
                    session_decoder.close()
                    raise error

                adopted_session = (
                    adopted_controller.get("session")
                    if isinstance(adopted_controller, dict) else None
                )
                adopted_is_distinct = bool(
                    isinstance(adopted_controller, dict)
                    and adopted_controller is not controller
                    and adopted_session is not session_decoder
                )
                adopted_is_current = bool(
                    adopted_is_distinct
                    and str(adopted_controller.get("revision_id", ""))
                    == current_revision_id
                    and _controller_handoff_is_valid(adopted_controller)
                )
                session_decoder.close()

                if adopted_is_current:
                    adopted_error = adopted_controller.get("fatal_error")
                    adopted_status = getattr(adopted_session, "status", None)
                    adopted_closed = bool(getattr(adopted_session, "_closed", False))
                    if (
                        isinstance(adopted_error, BaseException)
                        or adopted_status is NativeDecodeSessionStatus.FAILED
                    ):
                        if isinstance(adopted_error, BaseException):
                            raise adopted_error
                        adopted_fatal = getattr(adopted_session, "fatal_error", None)
                        if isinstance(adopted_fatal, BaseException):
                            raise adopted_fatal
                        if isinstance(error, BaseException):
                            raise error
                        raise RuntimeError("adopted native session failed without its original error")

                    if adopted_controller.get("done"):
                        result = _cached_controller_result(adopted_controller)
                    elif adopted_closed:
                        result = None
                    else:
                        resume = adopted_controller.get("run_to_completion")
                        result = resume() if callable(resume) else None
                    if isinstance(result, Mapping) and (
                        str(branch.get("revision_id", "r0")) == current_revision_id
                        and _controller_handoff_is_valid(adopted_controller)
                    ):
                        handoff_result = dict(result)
                        handoff_result["revision_handoff"] = True
                        if is_expected_cancel and error is not None:
                            handoff_result["active_decode_cancelled_by_revision"] = (
                                f"{type(error).__name__}: {error}"[:160]
                            )
                        return handoff_result

                if not isinstance(adopted_controller, dict):
                    adopted = branch.get("_last_adopted_think")
                    if (
                        isinstance(adopted, Mapping)
                        and str(branch.get("_streaming_revision_id", current_revision_id))
                        == current_revision_id
                    ):
                        result = dict(adopted)
                        result["revision_handoff"] = True
                        return result

                if is_expected_cancel:
                    final_result = controller.get("final_result")
                    if isinstance(final_result, Mapping):
                        return dict(final_result)
                    return None
                if error is not None:
                    raise error
                return None

            if not callable(candidate_ready_when):
                try:
                    while True:
                        step = advance_controller(lambda _progress: False)
                        handoff = handoff_adopted_session()
                        if handoff is not None:
                            return handoff
                        if step.status in {
                            NativeDecodeSessionStatus.DONE,
                            NativeDecodeSessionStatus.CANCELLED,
                        }:
                            return finish_decoded(step.result)
                except BaseException as exc:
                    handoff = handoff_adopted_session(exc)
                    if handoff is not None:
                        return handoff
                    raise
            raise RuntimeError("revision candidate session ended without a ready or terminal state")

        if continuation_prefix_ids is not None:
            assert continuation_logical_offset is not None
            assert continuation_budget_used is not None
            decoded = asyncio.run(decoder.decode_continuation(
                frozen_prefix_token_ids=continuation_prefix_ids,
                context_transition_token_ids=continuation_context_transition_ids,
                logical_source_offset=continuation_logical_offset,
                generated_budget_used=continuation_budget_used,
                total_generated_budget=DEFAULT_CONTINUATION_GENERATED_BUDGET,
                max_new_tokens=int(think_kwargs["max_new_tokens"]),
                **decode_options,
            ))
        else:
            decoded = asyncio.run(decoder.decode(
                transition_token_ids=transition_token_ids,
                max_new_tokens=int(think_kwargs["max_new_tokens"]),
                **decode_options,
            ))
    except BaseException as exc:
        decode_error = exc
        if (
            response_handle is not None
            and str(branch.get("revision_id", "r0")) != expected_branch_revision_id
        ):
            adopted = branch.get("_last_adopted_think")
            if isinstance(adopted, Mapping):
                # The coordinator can commit while this ACTIVE decoder is
                # between owner appends. Its handle is then fenced or already
                # released. Return the adopted cumulative result instead of
                # turning a successful revision into a failed response.
                publish_profile()
                handoff = dict(adopted)
                handoff["revision_handoff"] = True
                handoff["active_decode_cancelled_by_revision"] = (
                    f"{type(exc).__name__}: {exc}"[:160]
                )
                return handoff
        raise
    finally:
        release_decode_budget_reservation(decode_error)
    if (
        response_handle is not None
        and str(branch.get("revision_id", "r0")) != expected_branch_revision_id
        and decoded.termination_reason == "cancelled"
    ):
        adopted = branch.get("_last_adopted_think")
        if isinstance(adopted, Mapping):
            handoff = dict(adopted)
            handoff["revision_handoff"] = True
            return handoff
    return finish_decoded(decoded)
