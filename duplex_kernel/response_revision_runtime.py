"""Native decode integration for revisioned controller-guided V1 speech."""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Mapping

import torch

from duplex_kernel.audio_response_protocol import CONTROLLER_GUIDED_SAY_V1
from duplex_kernel.listener_microbatch_worker import (
    ListenerWatermark,
    ResponseDecodePurpose,
    StaleResponseRevisionError,
)
from duplex_kernel.response_revision import (
    ResponseRevisionCandidate,
    ResponseRevisionCoordinator,
    ResponseRevisionRequest,
    StaleResponseRevision,
    install_response_revision_transaction,
)
from duplex_kernel.session import SpokenTokenPacket
from duplex_kernel.server_unified_thinker import _run_unified_native_thinker
from duplex_kernel.turn_policy import ControlAction


MAX_CANDIDATE_PRE_ADOPTION_TOKENS = 8


def _candidate_ready_for_adoption(progress: Any, *, qwen_text_ready: bool) -> bool:
    """Permit source adoption once text is stable or eight Gemma IDs are committed."""
    if qwen_text_ready:
        return True
    return len(tuple(getattr(progress, "generated_token_ids", ()))) >= (
        MAX_CANDIDATE_PRE_ADOPTION_TOKENS
    )


def create_live_say_revision_coordinator(
    *,
    engine: Any,
    session: Any,
    request: Mapping[str, Any],
    worker: Any,
    branch: dict[str, Any],
    stream: Any,
    ledger: Any,
    transition_token_ids: tuple[int, ...],
    think_kwargs: Mapping[str, Any],
    response_protocol: str,
    generation_id: int,
    response_id: str,
    update_metrics: Callable[[str, Any], Any] | None = None,
) -> ResponseRevisionCoordinator:
    """Create one response-local candidate worker and atomic adoption seam."""
    if response_protocol != CONTROLLER_GUIDED_SAY_V1:
        raise ValueError("response revisions are only supported for controller-guided V1")
    if stream is None or ledger is None:
        raise RuntimeError("revisioned V1 response requires its Qwen stream and source ledger")

    cache_epoch_id = stream.cache_epoch
    branch["cache_epoch_id"] = cache_epoch_id
    branch.setdefault("_streaming_revision_id", str(branch.get("revision_id", "r0")))
    set_context = getattr(ledger, "set_context", None)
    if callable(set_context):
        set_context(
            revision_id=str(branch.get("revision_id", "r0")),
            cache_epoch_id=None if cache_epoch_id is None else str(cache_epoch_id),
        )

    def snapshot_with_applied_history(
        cancel_event: threading.Event,
    ) -> tuple[Any, int]:
        """Wait briefly for the canonical owner to apply the current prompt.

        The worker snapshot is read without holding any application lock. Its
        cache/history identity is then compared with the mirror that compaction
        publishes under the session generation lock. Adoption repeats that
        comparison inside the same lock, which closes the snapshot-to-CAS race.
        """
        deadline = time.monotonic() + 2.0
        last_snapshot: Any | None = None
        while not cancel_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                last_snapshot = worker.session_state(session.session_id).result(
                    timeout=min(0.25, remaining)
                )
            except TimeoutError:
                if cancel_event.wait(min(0.01, max(0.0, deadline - time.monotonic()))):
                    break
                continue
            with session._generation_lock:
                if (
                    not session.is_generation_current(generation_id=generation_id)
                    or str(session.response_id) != response_id
                    or session.stop_requested
                ):
                    return None, -1
                history_revision = int(session.semantic_context_revision)
                listener_state = getattr(session, "listener_state", None)
                if listener_state is None:
                    raise RuntimeError("revisioned SPEAK lacks the listener cache identity mirror")
                mirror_cache_epoch = getattr(listener_state, "cache_epoch_id", None)
                mirror_native_revision = getattr(
                    listener_state, "native_semantic_context_revision", None
                )
                mirror_history_revision = getattr(
                    listener_state, "applied_history_revision", None
                )
                snapshot_cache_epoch = getattr(last_snapshot, "cache_epoch_id", None)
                snapshot_native_revision = getattr(
                    last_snapshot, "semantic_context_revision", None
                )
                snapshot_history_revision = getattr(
                    last_snapshot, "applied_history_revision", None
                )
                if (
                    isinstance(mirror_cache_epoch, int)
                    and not isinstance(mirror_cache_epoch, bool)
                    and mirror_cache_epoch >= 0
                    and mirror_cache_epoch == snapshot_cache_epoch
                    and mirror_native_revision == snapshot_native_revision
                    and mirror_history_revision == history_revision
                    and snapshot_history_revision == history_revision
                ):
                    return last_snapshot, history_revision
            if cancel_event.wait(min(0.01, max(0.0, deadline - time.monotonic()))):
                break
        if cancel_event.is_set():
            return None, -1
        raise StaleResponseRevision(
            "canonical listener has not applied the current history revision"
        )

    def run_candidate(
        revision_request: ResponseRevisionRequest,
        cancel_event: threading.Event,
    ) -> ResponseRevisionCandidate | None:
        if cancel_event.is_set():
            return None
        with session._generation_lock:
            if (
                not session.is_generation_current(generation_id=generation_id)
                or str(session.response_id) != response_id
            ):
                return None
            with branch["spoken_prefix_lock"]:
                with stream._condition:
                    with ledger._lock:
                        if (
                            str(branch.get("response_id")) != response_id
                            or int(branch.get("session_generation_id", -1)) != generation_id
                        ):
                            raise StaleResponseRevision("response generation changed")
                        stream_frontier = stream._source_frontier_locked()
                        ledger_snapshot = ledger.source_frontier_snapshot()
                        if (
                            int(ledger_snapshot["frozen_source_sequence"])
                            != int(stream_frontier.frozen_source_sequence)
                            or str(ledger_snapshot["revision_id"])
                            != str(branch.get("revision_id", "r0"))
                            or ledger_snapshot["cache_epoch_id"]
                            != (
                                None if stream_frontier.cache_epoch is None
                                else str(stream_frontier.cache_epoch)
                            )
                        ):
                            raise StaleResponseRevision("stream and ledger source frontiers differ")
                        base_revision_id = str(branch.get("revision_id", "r0"))
                        frozen_frontier = int(stream_frontier.frozen_source_sequence)
                        prefix_rows = tuple(
                            dict(ledger._source_tokens[sequence])
                            for sequence in sorted(ledger._source_tokens)
                            if int(sequence) <= frozen_frontier
                        )
                        frozen_prefix_token_ids = tuple(
                            int(row["token_id"]) for row in prefix_rows
                        )
                        logical_source_offset = int(ledger_snapshot["next_source_sequence"])
                        revision_number = int(branch.get("_next_revision_number", 1))
                        branch["_next_revision_number"] = revision_number + 1
                        revision_id = f"r{revision_number}"

        remaining_budget = 768 - len(frozen_prefix_token_ids)
        if remaining_budget <= 0:
            return None
        listener_snapshot, history_revision = snapshot_with_applied_history(cancel_event)
        if listener_snapshot is None:
            return None
        if (
            int(listener_snapshot.generation_id)
            != int(branch.get("listener_generation_id", -1))
            or int(listener_snapshot.next_frame_id) - 1 < int(revision_request.frame_id)
        ):
            raise StaleResponseRevision("requested policy watermark is not completed yet")
        effective_cache_epoch = int(listener_snapshot.cache_epoch_id)
        source_native_revision = listener_snapshot.semantic_context_revision
        watermark = ListenerWatermark(
            session_id=str(session.session_id),
            generation_id=int(listener_snapshot.generation_id),
            state_epoch_id=str(listener_snapshot.state_epoch_id),
            frame_id=int(listener_snapshot.next_frame_id),
            state_version=int(listener_snapshot.state_version),
            sequence_position=int(listener_snapshot.sequence_position),
            cache_position=int(listener_snapshot.cache_position),
            cache_epoch_id=effective_cache_epoch,
            applied_history_revision=history_revision,
        )
        handle = worker.begin_response_decode(
            session.session_id,
            expected_generation_id=int(listener_snapshot.generation_id),
            response_id=response_id,
            response_generation_id=generation_id,
            revision_id=revision_id,
            purpose=ResponseDecodePurpose.for_action(ControlAction.REPLAN),
            source_watermark=watermark,
            max_tokens=max(1, min(768, remaining_budget)),
            prompt_token_budget=774,
        ).result(timeout=30.0)
        if (
            int(handle.source_cache_epoch_id) != effective_cache_epoch
            or int(handle.source_history_revision) != history_revision
            or int(handle.source_frame_id) != int(listener_snapshot.next_frame_id)
        ):
            worker.end_response_decode(handle).result(timeout=30.0)
            raise StaleResponseRevision("candidate fork did not preserve its applied cache watermark")
        candidate_branch: dict[str, Any] = {
            "worker": worker,
            "response_protocol": response_protocol,
            "response_action": ControlAction.SPEAK.value,
            "revisioned_live_say": True,
            "response_id": response_id,
            "response_generation_id": generation_id,
            "session_generation_id": generation_id,
            "listener_generation_id": int(handle.parent_generation_id),
            "listener_state_epoch_id": str(handle.parent_state_epoch_id),
            "source_frame_id": int(handle.source_frame_id),
            "source_state_version": int(handle.source_state_version),
            "source_sequence_position": int(handle.source_sequence_position),
            "cache_position": int(handle.source_cache_position),
            "turn_start_cache_position": int(handle.source_cache_position),
            "spoken_prefix_base_cursor": int(handle.source_cache_position),
            "response_decode_handle": handle,
            "response_decode_purpose": ResponseDecodePurpose.CANDIDATE.value,
            "revision_id": revision_id,
            "cache_epoch_id": effective_cache_epoch,
            "transition_token_ids": transition_token_ids,
            "say_token_packets": [],
            "segment_token_packets": [],
            "spoken_prefix_lock": threading.RLock(),
            "client_consumed_segments": (),
            "client_consumed_pcm_sequence": -1,
            "response_event_ledger": ledger,
            "generated_token_count": 0,
            "native_token_append_count": 0,
            "_generated_budget_used": len(frozen_prefix_token_ids),
            "close_ids": (),
        }
        candidate_rows: list[dict[str, Any]] = []
        candidate_packets: list[SpokenTokenPacket] = []
        candidate_packet_records: list[dict[str, Any]] = []
        candidate_branch["say_token_packets"] = candidate_packets
        candidate_branch["segment_token_packets"] = candidate_packet_records
        candidate_handoff: dict[str, Any] = {
            "promoted": False,
            "target_branch": None,
            "token_index_offset": len(frozen_prefix_token_ids),
        }

        class _CandidateCancelView:
            def is_set(self) -> bool:
                return not candidate_handoff["promoted"] and cancel_event.is_set()

            def set(self) -> None:
                cancel_event.set()

            def wait(self, timeout: float | None = None) -> bool:
                if candidate_handoff["promoted"]:
                    return False
                return cancel_event.wait(timeout)

        candidate_cancel_view = _CandidateCancelView()

        def candidate_token_id(token_id: int) -> None:
            if candidate_handoff["promoted"]:
                callback = request.get("_on_thinker_token_id")
                if callable(callback):
                    callback(int(token_id))

        def candidate_runtime_metrics(values: Any) -> None:
            if candidate_handoff["promoted"]:
                callback = request.get("_on_thinker_metrics")
                if callable(callback):
                    callback(values)

        candidate_request = dict(request)
        candidate_request.update({
            "_response_cancel_event": candidate_cancel_view,
            "_continuation_prefix_token_ids": frozen_prefix_token_ids,
            "_logical_source_offset": logical_source_offset,
            "_generated_budget_used": len(frozen_prefix_token_ids),
            "_cache_epoch_id": effective_cache_epoch,
            "_on_thinker_token_id": candidate_token_id,
            "_on_thinker_metrics": candidate_runtime_metrics,
        })

        def candidate_stop_check() -> bool:
            return bool(
                (cancel_event.is_set() and not candidate_handoff["promoted"])
                or session.stop_requested
                or not session.is_generation_current(generation_id=generation_id)
                or (request.get("_stop_check") and request["_stop_check"]())
                or stream.eos_consumed
            )

        def collect_candidate_packet(generated: Any) -> None:
            if cancel_event.is_set():
                raise RuntimeError("candidate revision was cancelled")
            token_index = int(generated.token_index)
            source_sequence = int(generated.logical_source_sequence)
            cursor = int(candidate_branch["spoken_prefix_base_cursor"]) + token_index + 1
            token_id = int(generated.token_id)
            text_piece = str(generated.token_text)
            hidden = generated.hidden
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("candidate packet lacks its post-forward Gemma hidden")
            if hidden.ndim == 3 and tuple(hidden.shape[:2]) == (1, 1):
                hidden_1536 = hidden[:, 0, :]
            elif hidden.ndim == 2:
                hidden_1536 = hidden
            else:
                raise ValueError("candidate hidden must have shape (1,1,1536) or (1,1536)")
            if tuple(hidden_1536.shape) != (1, 1536):
                raise ValueError("candidate hidden width must be 1536")
            packet = SpokenTokenPacket(
                token_id=token_id,
                text_piece=text_piece,
                gemma_hidden_1536=hidden_1536,
                exclusive_kv_cursor=cursor,
                hidden_provenance=str(generated.hidden_provenance),
                kv_committed=True,
            )
            provenance = {
                "revision_id": revision_id,
                "cache_epoch_id": effective_cache_epoch,
                "physical_kv_cursor": cursor,
            }
            row = {
                "source_token_seq": source_sequence,
                "token_id": token_id,
                "text_piece": text_piece,
                # The stream suffix plan rebuilds the tokenization and FiLM
                # source index from Gemma's post-forward state.  Keep this
                # with the logical source row; ACK packet records intentionally
                # contain only public text and physical provenance.
                "hidden": hidden_1536,
                "exclusive_kv_cursor": cursor,
                "physical_kv_cursor": cursor,
                "revision_id": revision_id,
                "cache_epoch_id": effective_cache_epoch,
                "physical_kv_provenance": [provenance],
            }
            record = {
                **row,
                "token_index": token_index,
                "hidden_provenance": str(generated.hidden_provenance),
                "kv_committed": True,
            }
            candidate_rows.append(row)
            candidate_packets.append(packet)
            candidate_packet_records.append(record)

        def update_candidate_final_text(_suffix: str, generated: Any) -> None:
            if candidate_handoff["promoted"]:
                callback = think_kwargs.get("on_decoded_text_flush")
                if callable(callback):
                    offset = int(candidate_handoff["token_index_offset"])
                    callback(
                        str(_suffix),
                        replace(generated, token_index=offset + int(generated.token_index)),
                    )
                return
            source_sequence = int(generated.logical_source_sequence)
            for index in range(len(candidate_rows) - 1, -1, -1):
                if int(candidate_rows[index]["source_token_seq"]) == source_sequence:
                    text_piece = str(generated.token_text)
                    candidate_rows[index]["text_piece"] = text_piece
                    candidate_packets[index] = replace(
                        candidate_packets[index], text_piece=text_piece
                    )
                    candidate_packet_records[index]["text_piece"] = text_piece
                    return
            raise RuntimeError("candidate final text flush lost its source row")

        def candidate_or_active_packet(generated: Any) -> None:
            if not candidate_handoff["promoted"]:
                collect_candidate_packet(generated)
                return
            callback = think_kwargs.get("on_token_packet")
            if callable(callback):
                offset = int(candidate_handoff["token_index_offset"])
                callback(replace(
                    generated,
                    token_index=offset + int(generated.token_index),
                ))

        def candidate_or_active_commit(generated: Any) -> None:
            if not candidate_handoff["promoted"]:
                return
            callback = think_kwargs.get("on_token_committed")
            if callable(callback):
                offset = int(candidate_handoff["token_index_offset"])
                callback(replace(
                    generated,
                    token_index=offset + int(generated.token_index),
                ))

        def candidate_qwen_ready(_progress: Any) -> bool:
            if cancel_event.is_set() or not candidate_rows:
                return False
            try:
                return bool(stream.has_unconsumed_qwen_tokens_for_suffix(
                    candidate_rows,
                    expected_frontier=stream_frontier,
                    frozen_prefix_token_ids=frozen_prefix_token_ids,
                    revision_id=revision_id,
                    cache_epoch=effective_cache_epoch,
                ))
            except RuntimeError as exc:
                raise StaleResponseRevision(str(exc)) from exc

        def candidate_ready_for_adoption(progress: Any) -> bool:
            # Bound pre-adoption work globally, not just per advance call. If
            # Gemma is still completing a byte/Unicode fragment, install its
            # source rows after eight committed IDs while retaining the old
            # Qwen queue; the same decoder session will publish when text is
            # stable after handoff.
            return _candidate_ready_for_adoption(
                progress,
                qwen_text_ready=candidate_qwen_ready(progress),
            )

        candidate_kwargs = dict(think_kwargs)
        candidate_kwargs.update({
            "max_new_tokens": int(remaining_budget),
            "min_new_tokens": 0,
            "stop_check": candidate_stop_check,
            "on_text_chunk": None,
            "on_token_id": None,
            "on_generation_start": None,
            "on_token_packet": candidate_or_active_packet,
            "on_token_committed": candidate_or_active_commit,
            "on_generated_token": None,
            "on_decoded_text_flush": update_candidate_final_text,
            "anticipate_first_token": False,
            "sentence_completion_max_tokens": 0,
            "graceful_sentence_completion": False,
            "producer_limit_check": None,
            "_revision_candidate_ready_when": candidate_ready_for_adoption,
        })
        candidate_controller_sink: dict[str, Any] = {"value": None}
        candidate_kwargs["_revision_native_session_controller_sink"] = candidate_controller_sink
        seed_bytes = hashlib.blake2b(
            f"{response_id}:{revision_id}:{revision_request.frame_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        private_generator = torch.Generator(device=getattr(engine, "device", "cpu"))
        private_generator.manual_seed(int.from_bytes(seed_bytes, "little") & 0x7FFF_FFFF)
        candidate_kwargs["_sampler_generator"] = private_generator
        ownership_state: dict[str, Any] = {"promoted": False, "released": False}
        candidate_controller: dict[str, Any] | None = None

        def cleanup_candidate() -> None:
            if ownership_state["promoted"] or ownership_state["released"]:
                return
            ownership_state["released"] = True
            controller_to_close = (
                candidate_controller
                if isinstance(candidate_controller, dict)
                else candidate_controller_sink.get("value")
            )
            if isinstance(controller_to_close, dict):
                close = controller_to_close.get("close")
                if callable(close):
                    close()
            worker.end_response_decode(handle).result()

        candidate_returned = False
        try:
            candidate_think = _run_unified_native_thinker(
                engine,
                session,
                candidate_request,
                worker=worker,
                transition_token_ids=transition_token_ids,
                think_kwargs=candidate_kwargs,
                generation_id=generation_id,
                branch=candidate_branch,
            )
            candidate_controller = candidate_think.get("_native_session_controller")
            if not isinstance(candidate_controller, dict):
                candidate_controller = candidate_controller_sink.get("value")
            if cancel_event.is_set() or candidate_think.get("cancelled"):
                return None
            if isinstance(candidate_controller, dict) and not candidate_controller.get("done"):
                checkpoint = candidate_controller.get("checkpoint_step")
                progress = getattr(checkpoint, "progress", None)
                if progress is None or not candidate_ready_for_adoption(progress):
                    return None
            elif candidate_rows and not candidate_qwen_ready(None):
                return None
            latest_snapshot = worker.session_state(session.session_id).result(timeout=30.0)
            if (
                int(latest_snapshot.generation_id) != int(handle.parent_generation_id)
                or str(latest_snapshot.state_epoch_id) != str(handle.parent_state_epoch_id)
                or int(latest_snapshot.cache_epoch_id) != effective_cache_epoch
                or latest_snapshot.semantic_context_revision != source_native_revision
                or latest_snapshot.applied_history_revision != history_revision
            ):
                raise StaleResponseRevision("listener cache/history epoch changed during candidate decode")
            candidate_think["spoken_prefix_base_cursor"] = int(
                candidate_branch.get("spoken_prefix_base_cursor", handle.source_cache_position)
            )
            candidate_think["cache_position"] = int(
                candidate_branch.get("cache_position", handle.source_cache_position)
            )
            candidate_think["transition_token_ids"] = tuple(
                int(value) for value in transition_token_ids
            )
            candidate_think["_revision_handoff_state"] = candidate_handoff
            candidate_think["_native_ownership_state"] = ownership_state
            candidate = ResponseRevisionCandidate(
                session_id=str(session.session_id),
                response_id=response_id,
                generation_id=generation_id,
                base_revision_id=base_revision_id,
                revision_id=revision_id,
                history_revision=history_revision,
                cache_epoch_id=effective_cache_epoch,
                listener_watermark=ListenerWatermark(
                    session_id=str(session.session_id),
                    generation_id=int(handle.parent_generation_id),
                    state_epoch_id=str(handle.parent_state_epoch_id),
                    frame_id=int(handle.source_frame_id),
                    state_version=int(handle.source_state_version),
                    sequence_position=int(handle.source_sequence_position),
                    cache_position=int(handle.source_cache_position),
                    cache_epoch_id=effective_cache_epoch,
                    applied_history_revision=history_revision,
                ),
                latest_listener_snapshot=latest_snapshot,
                requested_frame_id=int(revision_request.frame_id),
                stream_frontier=stream_frontier,
                ledger_snapshot=ledger_snapshot,
                frozen_prefix_token_ids=frozen_prefix_token_ids,
                source_rows=tuple(candidate_rows),
                packets=tuple(candidate_packets),
                packet_records=tuple(candidate_packet_records),
                cumulative_text=str(candidate_think.get("text", "")),
                think_result=candidate_think,
                cache_position=int(candidate_branch.get("cache_position", handle.source_cache_position)),
                handle=handle,
                cancel_event=cancel_event,
                cleanup_callback=cleanup_candidate,
            )
            candidate_returned = True
            return candidate
        finally:
            if not candidate_returned:
                cleanup_candidate()

    def promote_candidate_handle(candidate: ResponseRevisionCandidate) -> Any:
        active_handle = branch.get("response_decode_handle")
        expected_active_revision = candidate.base_revision_id if active_handle is not None else None
        try:
            return worker.promote_response_decode(
                candidate.handle,
                active_handle,
                expected_active_revision,
            ).result()
        except StaleResponseRevisionError as exc:
            raise StaleResponseRevision(str(exc)) from exc

    def commit_candidate(candidate: ResponseRevisionCandidate) -> bool:
        if not install_response_revision_transaction(
            session=session,
            branch=branch,
            stream=stream,
            ledger=ledger,
            candidate=candidate,
            promote_candidate=promote_candidate_handle,
        ):
            return False
        if callable(update_metrics):
            update_metrics("response_revision", {
                **dict(branch.get("revision_metrics", {})),
                **dict(coordinator.metrics_snapshot()),
            })
        return True

    def refresh_request(previous: ResponseRevisionRequest) -> ResponseRevisionRequest | None:
        if (
            session.stop_requested
            or not session.is_generation_current(generation_id=generation_id)
            or stream.eos_consumed
        ):
            return None
        return replace(previous, requested_at_ns=time.monotonic_ns())

    coordinator = ResponseRevisionCoordinator(
        session_id=str(session.session_id),
        response_id=response_id,
        generation_id=generation_id,
        run_candidate=run_candidate,
        commit_candidate=commit_candidate,
        refresh_request=refresh_request,
        max_stale_retries=2,
    )
    branch["revision_coordinator"] = coordinator
    branch["revision_metrics"] = coordinator.metrics_snapshot()
    return coordinator


__all__ = ["create_live_say_revision_coordinator"]
