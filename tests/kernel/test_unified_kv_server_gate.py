from __future__ import annotations

import asyncio
import json
import numpy as np
import pytest
import torch
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import duplex_kernel.server_unified_thinker as unified_thinker_module
import duplex_kernel.server_metrics as server_metrics_module
from duplex_kernel.server_metrics import _compact_unified_native_terminal_metrics

from duplex_kernel.server import (
    _acquire_shared_transition_lock_with_timing,
    _prewarm_unified_v1_transition_after_import,
    _controller_guided_say_generation_options,
    _build_unified_audio_response_suffix,
    _build_unified_backchannel_response_suffix,
    _build_unified_listener_prefix_token_ids,
    _build_unified_next_user_header,
    _consume_unified_native_text_append_result,
    _complete_client_pcm_segment,
    _finalize_unified_native_cache_branch,
    _record_emitted_pcm_packet,
    _record_client_consumed_pcm,
    _record_turn_local_vocalized_prefix,
    _require_closed_tts_codec_eos,
    _publish_thinker_metrics_event,
    _record_server_ack_lock_wait_ms,
    _unified_end_of_turn_token_id,
    _run_unified_native_thinker,
    _refresh_unified_prefix_for_instruction_update,
    _require_unified_stable_audio_protocol,
    _strict_env_switch,
    _native_control6_debug_input_row,
    _write_native_control6_debug_input_sidecar,
    _unified_listener_frame_contract,
    _unified_response_owns_shared_audio_queue,
    _shared_listener_response_branch,
    _shared_listener_frame_queue_contract,
    _shared_listener_queue_voice_activity,
    _attach_shared_listener_response_branches,
    _clear_shared_listener_response_branches,
    _validate_private_transition_chunk,
    _validate_unified_say_packet_frontier,
    _register_response_speech_ack_streams,
    _make_finish_audio_segment_callback,
    _ack_response_speech_packet,
    _retire_response_speech_ack_streams,
    _backchannel_budget_without_eos,
    _publish_backchannel_tokens_after_eos,
    _snapshot_speaking_branch_context,
    _unified_native_finalizer_event,
    _send_unified_native_finalizer_event,
    _filter_unified_native_event_metrics,
    scheduler_tick_trace_metadata,
)
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
    CONTROLLER_GUIDED_SAY_V1,
)
from duplex_kernel.server_response_runtime import (
    _unified_direct_audio_cache_branch_is_live,
    should_stream_thinker_text,
)
from duplex_kernel.scheduler import DuplexTick
from duplex_kernel.unified_native_decoder import NativeDecodeSessionStatus


@pytest.mark.parametrize(
    ("stream_requested", "unified_native", "tts_callback", "expected"),
    [
        (True, True, lambda *_args: None, False),
        (True, False, lambda *_args: None, True),
        (True, True, None, True),
        (False, False, lambda *_args: None, False),
    ],
)
def test_unified_tts_consumption_is_the_only_thinker_text_publisher(
    stream_requested, unified_native, tts_callback, expected
):
    assert should_stream_thinker_text(
        stream_requested=stream_requested,
        unified_native=unified_native,
        tts_text_consumed_callback=tts_callback,
    ) is expected


class _DecodeBudgetWorkerMixin:
    """Supply the no-capacity-limit reservation contract to native Thinker fakes."""

    def reserve_response_decode_budget(
        self,
        _session_id,
        *,
        requested_tokens,
        sentence_completion_tokens=0,
        **_kwargs,
    ):
        future = Future()
        future.set_result({
            "effective_tokens": int(requested_tokens),
            "sentence_completion_tokens": int(sentence_completion_tokens),
            "cap_reason": None,
            "close_reservation_tokens": 0,
        })
        return future

    def reserve_tail_tokens(self, _session_id, *, token_count, **_kwargs):
        future = Future()
        future.set_result(int(token_count))
        return future


def test_revisioned_v1_decode_handle_is_a_live_direct_audio_cache_branch() -> None:
    branch = {
        "queue_open": False,
        "revisioned_live_say": True,
        "response_decode_handle": object(),
    }

    assert _unified_direct_audio_cache_branch_is_live(
        branch, CONTROLLER_GUIDED_SAY_V1
    )
    assert _unified_direct_audio_cache_branch_is_live(
        {"queue_open": True}, AUDIO_RESPONSE_PROTOCOL_V2
    )
    assert not _unified_direct_audio_cache_branch_is_live(
        {"queue_open": False, "response_decode_handle": object()},
        CONTROLLER_GUIDED_SAY_V1,
    )
    assert not _unified_direct_audio_cache_branch_is_live(
        branch, AUDIO_RESPONSE_PROTOCOL_V2
    )


def test_unified_native_preflight_retry_observes_cancellation_before_retry() -> None:
    class Worker:
        reserve_attempts = 0

        def session_state(self, _session_id):
            future = Future()
            future.set_result(SimpleNamespace(generation_id=12, cache_position=20))
            return future

        def reserve_response_decode_budget(self, *_args, **_kwargs):
            self.reserve_attempts += 1
            raise RuntimeError(
                "response decode preflight requires a completed listener frame watermark"
            )

    checks = 0

    def stop_check():
        nonlocal checks
        checks += 1
        return checks >= 2

    worker = Worker()
    with pytest.raises(asyncio.CancelledError):
        _run_unified_native_thinker(
            SimpleNamespace(),
            SimpleNamespace(session_id="session"),
            {},
            worker=worker,
            transition_token_ids=(1, 2),
            think_kwargs={"max_new_tokens": 1, "stop_check": stop_check},
            generation_id=4,
            branch={
                "listener_generation_id": 12,
                "cache_position": 20,
                "turn_start_cache_position": 20,
                "native_token_append_count": 0,
                "client_consumed_segments": (),
            },
        )

    assert worker.reserve_attempts == 1
    assert checks == 2


def test_unified_native_stop_check_precedes_worker_preflight_state_read() -> None:
    class Worker:
        state_reads = 0
        reserve_attempts = 0

        def session_state(self, _session_id):
            self.state_reads += 1
            raise AssertionError("cancelled decode must stop before native preflight")

        def reserve_response_decode_budget(self, *_args, **_kwargs):
            self.reserve_attempts += 1
            raise AssertionError("cancelled decode must not reserve KV")

    worker = Worker()
    with pytest.raises(asyncio.CancelledError):
        _run_unified_native_thinker(
            SimpleNamespace(),
            SimpleNamespace(session_id="session"),
            {},
            worker=worker,
            transition_token_ids=(1,),
            think_kwargs={"max_new_tokens": 1, "stop_check": lambda: True},
            generation_id=4,
            branch={"listener_generation_id": 12, "cache_position": 20},
        )

    assert worker.state_reads == worker.reserve_attempts == 0


def test_unified_native_cancellation_after_reservation_releases_close_tail_once() -> None:
    class Worker:
        state_reads = 0
        reservation_calls = 0
        released_tail_counts: list[int] = []

        def session_state(self, _session_id):
            self.state_reads += 1
            future = Future()
            future.set_result(SimpleNamespace(generation_id=12, cache_position=20))
            return future

        def reserve_response_decode_budget(self, *_args, **_kwargs):
            self.reservation_calls += 1
            future = Future()
            future.set_result({
                "effective_tokens": 1,
                "sentence_completion_tokens": 0,
                "cap_reason": None,
                "close_reservation_tokens": 3,
            })
            return future

        def reserve_tail_tokens(self, _session_id, *, token_count, **_kwargs):
            self.released_tail_counts.append(int(token_count))
            future = Future()
            future.set_result(int(token_count))
            return future

    checks = 0

    def stop_check():
        nonlocal checks
        checks += 1
        return checks >= 2

    worker = Worker()
    with pytest.raises(asyncio.CancelledError):
        _run_unified_native_thinker(
            SimpleNamespace(),
            SimpleNamespace(session_id="session"),
            {},
            worker=worker,
            transition_token_ids=(1,),
            think_kwargs={"max_new_tokens": 1, "stop_check": stop_check},
            generation_id=4,
            branch={
                "listener_generation_id": 12,
                "cache_position": 20,
                "turn_start_cache_position": 20,
                "native_token_append_count": 0,
                "client_consumed_segments": (),
            },
        )

    assert worker.reservation_calls == 1
    assert worker.released_tail_counts == [3]
    assert worker.state_reads == 2  # Initial frontier and cleanup frontier.


def test_active_response_returns_adopted_revision_when_its_handle_is_fenced() -> None:
    adopted = {
        "text": "Ciao, aggiornamento.",
        "protocol_text": "Ciao, aggiornamento.",
        "suffix_text": " aggiornamento.",
        "gen_ids": [51, 52],
        "terminal_reason": "eos",
        "cancelled": False,
    }
    branch = {
        "listener_generation_id": 12,
        "response_id": "response-a",
        "response_generation_id": 4,
        "response_decode_handle": SimpleNamespace(
            parent_session_id="session",
            parent_generation_id=12,
            source_cache_position=20,
            response_id="response-a",
            response_generation_id=4,
            revision_id="r0",
        ),
        "revision_id": "r0",
        "cache_position": 20,
        "turn_start_cache_position": 20,
        "native_token_append_count": 0,
        "client_consumed_segments": (),
    }

    class Worker:
        def session_state(self, _session_id):
            future = Future()
            future.set_result(SimpleNamespace(
                generation_id=12,
                cache_position=20,
                valid_history_tokens=20,
            ))
            return future

        def append_response_decode(self, *_args, **_kwargs):
            branch["revision_id"] = "r1"
            branch["_last_adopted_think"] = adopted
            future = Future()
            future.set_exception(RuntimeError("active response fork was released after adoption"))
            return future

    class Sampler:
        eos_token_ids = {99}
        runtime_metrics = {}
        sample_token = staticmethod(lambda *_args: pytest.fail("sample must not run after fenced append"))
        sample_initial = staticmethod(lambda *_args: pytest.fail("sample must not run after fenced append"))
        select_valid_candidate = staticmethod(
            lambda *_args, **_kwargs: pytest.fail("candidate selection must not run after fenced append")
        )

    engine = SimpleNamespace(
        thinker=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ",".join(map(str, ids))),
        device="cpu",
        dtype=torch.float32,
        _thinker_wrapper=SimpleNamespace(processor=object()),
        audio_tower=object(),
        build_thinker_sampler=lambda **_kwargs: Sampler(),
    )
    result = _run_unified_native_thinker(
        engine,
        SimpleNamespace(session_id="session"),
        {},
        worker=Worker(),
        transition_token_ids=(1, 2),
        think_kwargs={
            "temperature": 0.0,
            "top_p": 1.0,
            "min_new_tokens": 0,
            "max_new_tokens": 2,
        },
        generation_id=4,
        branch=branch,
    )

    assert result["revision_handoff"] is True
    assert result["text"] == adopted["text"]
    assert result["gen_ids"] == adopted["gen_ids"]
    assert "active_decode_cancelled_by_revision" in result


def test_promoted_native_checkpoint_resumes_and_publishes_terminal_result(monkeypatch) -> None:
    response_handle = SimpleNamespace(
        handle_id="candidate-handle",
        purpose="candidate_response",
        parent_session_id="session",
        parent_generation_id=12,
        response_id="response-a",
        response_generation_id=4,
        revision_id="r1",
        source_cache_position=20,
    )
    promoted_handle = SimpleNamespace(
        **{
            **vars(response_handle),
            "purpose": "active_response",
        }
    )
    session = SimpleNamespace(
        session_id="session",
        _generation_lock=threading.RLock(),
    )
    candidate_branch = {
        "listener_generation_id": 12,
        "response_id": "response-a",
        "response_generation_id": 4,
        "response_decode_handle": response_handle,
        "revision_id": "r1",
        "response_protocol": "controller_guided_say_v1",
        "revisioned_live_say": True,
        "cache_position": 20,
        "turn_start_cache_position": 20,
        "spoken_prefix_base_cursor": 20,
        "native_token_append_count": 0,
        "client_consumed_segments": (),
    }

    class Worker:
        def __init__(self):
            self.cursor = 20
            self.appended = []
            self.handles = []

        @staticmethod
        def _done(value):
            future = Future()
            future.set_result(value)
            return future

        def session_state(self, _session_id):
            return self._done(SimpleNamespace(
                generation_id=12,
                cache_position=self.cursor,
            ))

        def append_response_decode(
            self, handle, token_ids, *, expected_cache_position, generated,
        ):
            assert expected_cache_position == self.cursor
            self.handles.append((handle.handle_id, handle.purpose, bool(generated)))
            token_ids = tuple(int(value) for value in token_ids)
            self.appended.extend(token_ids)
            self.cursor += len(token_ids)
            return self._done(SimpleNamespace(
                cache_position=self.cursor,
                token_count=len(token_ids),
                logits=torch.zeros((1, 100)),
                final_norm_hidden=torch.zeros((1, 1, 4)),
                ready_event=None,
                timings_ns={},
            ))

    worker = Worker()

    class Sampler:
        eos_token_ids = {99}
        runtime_metrics = {}

        def sample_token(self, *_args):
            raise AssertionError("the fake session does not sample")

        def sample_initial(self, *_args):
            raise AssertionError("the fake session does not sample")

        def select_valid_candidate(self, *_args, **_kwargs):
            raise AssertionError("the fake session does not select candidates")

    class FakeSession:
        def __init__(self, forward_tokens, **options):
            self.forward_tokens = forward_tokens
            self.options = options
            self.handle = options["native_handle"]
            self.calls = 0
            self.generated = []
            self.native_cursor = options["native_cursor"]

        @property
        def native_handle(self):
            return self.handle

        @property
        def generated_token_ids(self):
            return tuple(int(token.token_id) for token in self.generated)

        def validate_native_handle_adoption(self, handle):
            assert handle.handle_id == self.handle.handle_id

        def adopt_native_handle(self, handle):
            self.validate_native_handle_adoption(handle)
            self.handle = handle

        def close(self):
            return None

        @staticmethod
        def _token(token_id, index):
            return SimpleNamespace(
                token_id=token_id,
                token_index=index,
                token_text="A" if token_id == 11 else "B",
                hidden=torch.full((1, 1, 4), float(token_id)),
                packet_published_ns=100 + index,
                kv_committed=True,
                text_source_sequences=(),
            )

        async def _call(self, name, *args):
            callback = self.options.get(name)
            if callable(callback):
                result = callback(*args)
                if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                    await result

        async def advance(
            self, *, max_committed_tokens, ready_when, stop_check,
            producer_limit_check,
        ):
            assert max_committed_tokens == 8
            assert stop_check is None or not stop_check()
            if self.calls == 0:
                await self.forward_tokens([5, 6, 17])
                token = self._token(11, 0)
                await self.forward_tokens([11])
                self.generated.append(token)
                await self._call("on_token_packet", token)
                await self._call("on_token_committed", token)
                progress = SimpleNamespace(
                    generated_token_ids=(11,),
                    generated_text="A",
                    stable_text="A",
                    source_tokens=(),
                    source_offset=2,
                    native_cursor=24,
                )
                assert ready_when(progress)
                self.calls += 1
                self.native_cursor = 24
                return SimpleNamespace(
                    status=NativeDecodeSessionStatus.READY,
                    generated_tokens=(token,),
                    progress=progress,
                    native_handle=self.handle,
                    native_cursor=self.native_cursor,
                    source_offset=2,
                    result=None,
                )

            assert self.handle.purpose == "active_response"
            token = self._token(12, 1)
            await self.forward_tokens([12])
            self.generated.append(token)
            await self._call("on_token_packet", token)
            await self._call("on_token_committed", token)
            self.calls += 1
            self.native_cursor = 25
            result = SimpleNamespace(
                generated_tokens=tuple(self.generated),
                termination_reason="eos",
                terminal_token_id=99,
                generated_count=3,
                committed_count=2,
                sampled_count=3,
                transition_token_count=3,
                first_logits_ready=None,
                first_token=self.generated[0],
                pending_token_id=None,
                owner_action="continue_generation",
            )
            return SimpleNamespace(
                status=NativeDecodeSessionStatus.DONE,
                generated_tokens=(token,),
                progress=SimpleNamespace(
                    generated_token_ids=(11, 12),
                    generated_text="AB",
                    stable_text="AB",
                    source_tokens=(),
                    source_offset=3,
                    native_cursor=25,
                ),
                native_handle=self.handle,
                native_cursor=self.native_cursor,
                source_offset=3,
                result=result,
            )

    class FakeDecoder:
        def __init__(self, _thinker, *, forward_tokens, sampler, **_kwargs):
            assert sampler is not None
            self.forward_tokens = forward_tokens

        def start_session(self, **options):
            return FakeSession(self.forward_tokens, **options)

    monkeypatch.setattr(unified_thinker_module, "UnifiedNativeDecoder", FakeDecoder)
    post_adoption_packet_indexes = []
    callback_handoff = {"promoted": False, "offset": 1}

    def packet_callback(token):
        if callback_handoff["promoted"]:
            post_adoption_packet_indexes.append(
                callback_handoff["offset"] + int(token.token_index)
            )

    engine = SimpleNamespace(
        thinker=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=SimpleNamespace(
            decode=lambda ids, skip_special_tokens=True: "".join(
                {17: "P", 11: "A", 12: "B"}[int(value)] for value in ids
            ),
        ),
        device="cpu",
        dtype=torch.float32,
        _thinker_wrapper=SimpleNamespace(processor=object()),
        audio_tower=object(),
        build_thinker_sampler=lambda **_kwargs: Sampler(),
    )
    checkpoint = _run_unified_native_thinker(
        engine,
        session,
        {
            "_continuation_prefix_token_ids": (17,),
            "_logical_source_offset": 1,
            "_generated_budget_used": 1,
        },
        worker=worker,
        transition_token_ids=(5, 6),
        think_kwargs={
            "temperature": 0.0,
            "top_p": 1.0,
            "min_new_tokens": 0,
            "max_new_tokens": 767,
            "on_token_packet": packet_callback,
            "on_token_committed": lambda _token: None,
            "_revision_candidate_ready_when": lambda progress: len(
                progress.generated_token_ids
            ) >= 1,
        },
        generation_id=4,
        branch=candidate_branch,
    )
    controller = checkpoint["_native_session_controller"]
    assert checkpoint["text"] == "A"
    assert checkpoint["gen_ids"] == [11]
    assert worker.appended == [5, 6, 17, 11]

    target_branch = {
        "revision_id": "r1",
        "spoken_prefix_lock": threading.RLock(),
        "_generated_budget_used": 2,
        "generated_token_count": 2,
        "sampled_generated_token_count": 2,
        "unappended_terminal_token_count": 0,
        "native_token_append_count": 4,
        "cache_position": 24,
        "_last_adopted_think": checkpoint,
        "_adopted_native_session_controller": controller,
    }
    handoff = controller["handoff_state"]
    handoff.update({
        "promoted": True,
        "target_branch": target_branch,
        "token_index_offset": 1,
    })
    controller["promoted"] = True
    controller["source_token_index_offset"] = 1
    controller["handle_ref"]["value"] = promoted_handle
    controller["session"].adopt_native_handle(promoted_handle)
    callback_handoff["promoted"] = True

    completed = controller["run_to_completion"]()

    assert completed["terminal_reason"] == "eos"
    assert completed["text"] == completed["protocol_text"] == "PAB"
    assert completed["gen_ids"] == [11, 12]
    assert completed["unappended_terminal_token_count"] == 1
    assert target_branch["_last_adopted_think"] is controller["final_result"]
    assert target_branch["_last_adopted_think"]["text"] == "PAB"
    assert target_branch["_generated_budget_used"] == 3
    assert target_branch["generated_token_count"] == 3
    assert target_branch["sampled_generated_token_count"] == 4
    assert target_branch["unappended_terminal_token_count"] == 1
    assert target_branch["cache_position"] == 25
    assert post_adoption_packet_indexes == [2]
    assert worker.handles[-1] == ("candidate-handle", "active_response", True)


def test_unified_native_preflight_retry_rejects_replaced_listener_generation() -> None:
    class Worker:
        reserve_attempts = 0
        state_reads = 0

        def session_state(self, _session_id):
            self.state_reads += 1
            future = Future()
            generation_id = 12 if self.state_reads == 1 else 13
            future.set_result(SimpleNamespace(generation_id=generation_id, cache_position=20))
            return future

        def reserve_response_decode_budget(self, *_args, **_kwargs):
            self.reserve_attempts += 1
            raise RuntimeError(
                "response decode preflight requires a completed listener frame watermark"
            )

    worker = Worker()
    with pytest.raises(RuntimeError, match="preflight lost its listener generation"):
        _run_unified_native_thinker(
            SimpleNamespace(),
            SimpleNamespace(session_id="session"),
            {},
            worker=worker,
            transition_token_ids=(1, 2),
            think_kwargs={"max_new_tokens": 1, "stop_check": lambda: False},
            generation_id=4,
            branch={
                "listener_generation_id": 12,
                "cache_position": 20,
                "turn_start_cache_position": 20,
                "native_token_append_count": 0,
                "client_consumed_segments": (),
            },
        )

    assert worker.reserve_attempts == 1
    assert worker.state_reads == 2


def test_live_film_callback_configuration_disables_anticipated_transition_hidden() -> None:
    from duplex_kernel.server_response_runtime import _configure_native_live_say_callbacks

    packets = []
    committed = []
    packet_callback = packets.append
    committed_callback = committed.append
    generated = lambda *_args: pytest.fail("legacy token callback must not replace packet publication")
    think_kwargs = {"anticipate_first_token": True, "on_generated_token": generated}
    _configure_native_live_say_callbacks(
        think_kwargs,
        frame_residual_active=True,
        packet_callback=packet_callback,
        committed_callback=committed_callback,
    )

    assert think_kwargs["anticipate_first_token"] is False
    assert think_kwargs["on_token_packet"] is packet_callback
    assert think_kwargs["on_token_committed"] is committed_callback
    assert think_kwargs["on_generated_token"] is None


def test_ordinary_live_say_keeps_first_token_anticipation() -> None:
    from duplex_kernel.server_response_runtime import _configure_native_live_say_callbacks

    think_kwargs = {}
    _configure_native_live_say_callbacks(
        think_kwargs,
        frame_residual_active=False,
        packet_callback=lambda _packet: None,
        committed_callback=lambda _packet: None,
    )

    assert think_kwargs["anticipate_first_token"] is True


def test_unified_v1_prewarm_waits_for_true_prefix_import_before_policy_dispatch() -> None:
    events: list[tuple] = []
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    class Worker:
        imported = False
        outstanding = 1

        def prepare_text_token_inputs(self, token_ids, *, protocol):
            events.append(("prepare", tuple(token_ids), protocol))
            future = Future()
            future.set_result({"logical_q_len": 6, "physical_q_len": 6, "cache_size": 1, "cache_misses": 1})
            return future

        def begin_ephemeral_decode(self, session_id, *, expected_generation_id, expected_cache_position):
            if not self.imported or self.outstanding != 0:
                raise RuntimeError("unified native operation requires an imported true chat prefix")
            events.append(("begin", session_id, expected_generation_id, expected_cache_position))
            future = Future()
            future.set_result(SimpleNamespace(cache_position=expected_cache_position, handle_id="private-q6"))
            return future

        def append_private_transition_tokens(self, private_handle, token_ids, *, expected_cache_position):
            events.append(("native_q6", private_handle.handle_id, tuple(token_ids), expected_cache_position))
            future = Future()
            future.set_result(SimpleNamespace(cache_position=expected_cache_position + len(token_ids)))
            return future

        def end_ephemeral_decode(self, private_handle):
            events.append(("end", private_handle.handle_id))
            future = Future()
            future.set_result(None)
            return future

    worker = Worker()
    binding = {
        "event_loop": loop,
        "unified_v1_transition_prewarm_key": None,
    }
    output = SimpleNamespace(
        generation_id=7,
        state_epoch_id="epoch-q6",
        semantic_context_revision=0,
        semantic_prefix_len=42,
        cache_position=42,
    )
    transition_ids = (100, 101, 102, 103, 104, 105)
    try:
        # Session setup must complete before the lazy true-prefix import; a
        # session-open Q6 CAS would fail with the same guard as begin() above.
        events.append(("session.created",))
        worker.imported = True
        worker.outstanding = 0
        events.append(("tick_complete_true_prefix_imported", 42, 0))
        result = _prewarm_unified_v1_transition_after_import(
            worker,
            binding=binding,
            output=output,
            session_id="session-q6",
            transition_ids=transition_ids,
        )
        events.append(("policy_dispatch",))

        # A normal next audio frame advances only the canonical KV cursor;
        # it must not trigger a second Q6 roundtrip for the same prefix.
        second_frame_output = SimpleNamespace(
            generation_id=7,
            state_epoch_id="epoch-q6",
            semantic_context_revision=0,
            semantic_prefix_len=42,
            cache_position=44,
        )
        assert _prewarm_unified_v1_transition_after_import(
            worker,
            binding=binding,
            output=second_frame_output,
            session_id="session-q6",
            transition_ids=transition_ids,
        ) is None

        # A refreshed prefix revision owns a new prewarm, still using the
        # current canonical cursor supplied by its completed import tick.
        refreshed_output = SimpleNamespace(
            generation_id=7,
            state_epoch_id="epoch-q6",
            semantic_context_revision=1,
            semantic_prefix_len=50,
            cache_position=50,
        )
        events.append(("prefix_refresh_imported", 50, 0))
        refreshed = _prewarm_unified_v1_transition_after_import(
            worker,
            binding=binding,
            output=refreshed_output,
            session_id="session-q6",
            transition_ids=transition_ids,
        )
        events.append(("policy_dispatch_after_refresh",))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)
        loop.close()

    assert events == [
        ("session.created",),
        ("tick_complete_true_prefix_imported", 42, 0),
        ("prepare", transition_ids, "controller_guided_say_v1"),
        ("begin", "session-q6", 7, 42),
        ("native_q6", "private-q6", transition_ids, 42),
        ("end", "private-q6"),
        ("policy_dispatch",),
        ("prefix_refresh_imported", 50, 0),
        ("prepare", transition_ids, "controller_guided_say_v1"),
        ("begin", "session-q6", 7, 50),
        ("native_q6", "private-q6", transition_ids, 50),
        ("end", "private-q6"),
        ("policy_dispatch_after_refresh",),
    ]
    assert result["logical_q_len"] == result["physical_q_len"] == 6
    assert result["canonical_cache_position"] == 42
    assert result["private_cache_position"] == 48
    assert result["prepare_ms"] >= 0.0
    assert result["native_roundtrip_ms"] >= 0.0
    assert refreshed["canonical_cache_position"] == 50
    assert refreshed["private_cache_position"] == 56


def test_cancelled_late_native_token_metrics_are_kept_locally_and_not_sent() -> None:
    cancellation_event = threading.Event()
    current_generation = {"value": True}
    local_profiles: list[tuple[str, str, dict]] = []
    websocket_events: list[dict] = []
    fifo = [{"token_id": 7, "kv_committed": True}]

    def update_metrics(response_id: str, phase: str, values: dict) -> None:
        local_profiles.append((response_id, phase, dict(values)))

    def send_event(event: dict) -> bool:
        websocket_events.append(event)
        return False

    cancellation_event.set()  # The token0 forward completed after its response was fenced.
    fifo_before_callback = [dict(packet) for packet in fifo]
    result = _publish_thinker_metrics_event(
        {"unified_native_profile": {"token0_forward_return_ns": 123}},
        response_id="resp-stale",
        policy_decision_at_ns=100,
        session=SimpleNamespace(
            is_generation_current=lambda _token: current_generation["value"]
        ),
        token=object(),
        cancellation_event=cancellation_event,
        update_runtime_metrics=update_metrics,
        send_event=send_event,
    )

    assert result is False
    assert local_profiles == [(
        "resp-stale",
        "thinker_metrics",
        {
            "unified_native_profile": {"token0_forward_return_ns": 123},
            "policy_decision_at_ns": 100,
        },
    )]
    assert websocket_events == []
    assert fifo == fifo_before_callback


def test_thinker_metrics_websocket_event_projects_large_native_profile() -> None:
    append_steps = [
        {"phase": "generated_token", "append_call_ns": 10_000 + index}
        for index in range(501)
    ]
    sampling_steps = [
        {
            "stage": "sample_token",
            "started_ns": 20_000 + index,
            "finished_ns": 20_100 + index,
            "candidate_id": index,
        }
        for index in range(1_001)
    ]
    per_token_latencies = [float(index) / 10 for index in range(1_001)]
    values = {
        "policy_decision_at_ns": 123_456,
        "thinker_first_token_at_ns": 123_789,
        "first_logits_after_decision_ms": 3.25,
        "thinker_runtime_metrics": {
            "thinker_decode_lock_wait_ms": 0.0,
            "generated_token_count": 1_001,
            "summary": {"terminal_reason": "eos", "sampled_token_count": 1_001},
            "per_token_latencies_ms": per_token_latencies,
        },
        "unified_native_profile": {
            "clock": "server monotonic_ns",
            "token0_forward_submit_ns": 123_500,
            "token0_forward_return_ns": 123_600,
            "token0_append_request": {
                "generation_id": 7,
                "expected_cache_position": 41,
                "append_scope_id": "resp-large-profile",
            },
            "first_packet_published_clock": "server monotonic_ns",
            "append_steps": append_steps,
            "sampling_steps": sampling_steps,
        },
    }
    local_profiles: list[dict] = []
    websocket_events: list[dict] = []

    result = _publish_thinker_metrics_event(
        values,
        response_id="resp-large-profile",
        policy_decision_at_ns=None,
        session=SimpleNamespace(is_generation_current=lambda _token: True),
        token=object(),
        cancellation_event=threading.Event(),
        update_runtime_metrics=lambda _rid, _phase, metric_values: local_profiles.append(
            dict(metric_values)
        ),
        send_event=lambda event: websocket_events.append(event) or True,
    )

    assert result is True
    assert local_profiles == [values]
    assert local_profiles[0]["unified_native_profile"]["append_steps"] == append_steps
    assert local_profiles[0]["unified_native_profile"]["sampling_steps"] == sampling_steps
    assert local_profiles[0]["thinker_runtime_metrics"]["per_token_latencies_ms"] == per_token_latencies
    assert len(websocket_events) == 1

    event = websocket_events[0]
    compact_utf8 = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(compact_utf8) < 8 * 1024
    assert event["policy_decision_at_ns"] == 123_456
    assert event["thinker_first_token_at_ns"] == 123_789
    assert event["first_logits_after_decision_ms"] == 3.25
    assert event["thinker_runtime_metrics"]["generated_token_count"] == 1_001
    assert event["unified_native_profile"]["token0_forward_submit_ns"] == 123_500
    assert event["unified_native_profile"]["token0_forward_return_ns"] == 123_600
    assert event["unified_native_profile"]["token0_append_request"] == {
        "generation_id": 7,
        "expected_cache_position": 41,
        "append_scope_id": "resp-large-profile",
    }
    assert event["unified_native_profile"]["append_steps_count"] == 501
    assert event["unified_native_profile"]["sampling_steps_count"] == 1_001
    assert "append_steps" not in event["unified_native_profile"]
    assert "sampling_steps" not in event["unified_native_profile"]
    assert event["thinker_runtime_metrics"]["summary"] == {
        "terminal_reason": "eos",
        "sampled_token_count": 1_001,
    }
    assert "per_token_latencies_ms" not in event["thinker_runtime_metrics"]


def test_server_ack_lock_wait_is_additive_to_existing_queue_timing() -> None:
    event = {
        "type": "input_audio_buffer.consumed",
        "server_ack_queue_ms": 4.5,
    }

    _record_server_ack_lock_wait_ms(
        event,
        lock_requested_ns=1_000_000,
        lock_acquired_ns=2_234_567,
    )

    assert event["server_ack_queue_ms"] == 4.5
    assert event["server_ack_lock_wait_ms"] == 1.235


def test_cancelled_large_thinker_metrics_profile_is_not_sent() -> None:
    values = {
        "policy_decision_at_ns": 123_456,
        "unified_native_profile": {
            "token0_forward_return_ns": 123_600,
            "append_steps": [{"append_call_ns": index} for index in range(501)],
            "sampling_steps": [{"sampled_ns": index} for index in range(1_001)],
        },
    }
    cancellation_event = threading.Event()
    cancellation_event.set()
    local_profiles: list[dict] = []
    websocket_events: list[dict] = []

    result = _publish_thinker_metrics_event(
        values,
        response_id="resp-cancelled-large-profile",
        policy_decision_at_ns=None,
        session=SimpleNamespace(is_generation_current=lambda _token: True),
        token=object(),
        cancellation_event=cancellation_event,
        update_runtime_metrics=lambda _rid, _phase, metrics: local_profiles.append(
            dict(metrics)
        ),
        send_event=lambda event: websocket_events.append(event) or True,
    )

    assert result is False
    assert len(local_profiles) == 1
    assert len(local_profiles[0]["unified_native_profile"]["append_steps"]) == 501
    assert len(local_profiles[0]["unified_native_profile"]["sampling_steps"]) == 1_001
    assert websocket_events == []


def test_speaking_branch_snapshots_direct_audio_and_acoustic_state_independently() -> None:
    direct_pcm = np.arange(8, dtype=np.float32)
    user_frame = np.full((4,), 0.25, dtype=np.float32)
    assistant_frame = np.full((4,), -0.5, dtype=np.float32)
    state = SimpleNamespace(
        _lock=threading.RLock(),
        state_epoch_id="epoch-a",
        listener_state_version=12,
        next_frame_id=14,
        gemma_lm_context_start_frame=3,
        gemma_lm_source_frame_ids=[3, 4],
        gemma_lm_last_source_frame_id=4,
        user_pcm_history=[user_frame],
        assistant_pcm_history=[assistant_frame],
        user_mel_history=[],
        assistant_mel_history=[],
        user_pcm_tail=user_frame,
        assistant_pcm_tail=assistant_frame,
    )
    direct, acoustic = _snapshot_speaking_branch_context(
        {"_direct_audio_pcm": direct_pcm, "_direct_audio_window": {"start": 3}},
        state,
    )
    assert direct is not None and acoustic["state_epoch_id"] == "epoch-a"
    assert direct["_direct_audio_pcm"] is not direct_pcm
    assert acoustic["user_pcm_history"][0] is not user_frame
    assert acoustic["assistant_pcm_tail"] is not assistant_frame
    direct_pcm[0] = -9.0
    user_frame[0] = 9.0
    assistant_frame[0] = 9.0
    assert direct["_direct_audio_pcm"][0] == 0.0
    assert acoustic["user_pcm_history"][0][0] == 0.25
    assert acoustic["assistant_pcm_tail"][0] == -0.5


@pytest.mark.parametrize(
    ("token_index", "cursor_offset", "kv_committed", "frontier_offset"),
    [
        (0, 1, False, 0),
        (1, 2, True, 2),
    ],
)
def test_unified_say_cursor_uses_actual_transition_suffix_and_commit_state(
    token_index: int,
    cursor_offset: int,
    kv_committed: bool,
    frontier_offset: int,
) -> None:
    # The six-ID transition starts at an audio endpoint that the listener
    # rebases before appending. The packet anchor must be the returned native
    # suffix cursor, not endpoint + a hard-coded template length.
    actual_suffix_cursor = 128
    branch = {
        "spoken_prefix_base_cursor": actual_suffix_cursor,
        "spoken_prefix_transition_cursor": actual_suffix_cursor,
        "cache_position": actual_suffix_cursor + frontier_offset,
        "transition_token_ids": (11, 12, 13, 14, 15, 16),
        "listener_generation_id": 7,
        "session_generation_id": 7,
    }

    assert _validate_unified_say_packet_frontier(
        branch,
        token_index,
        actual_suffix_cursor + cursor_offset,
        kv_committed=kv_committed,
    ) == actual_suffix_cursor + cursor_offset


def test_unified_say_cursor_mismatch_reports_frontier_provenance() -> None:
    branch = {
        "spoken_prefix_base_cursor": 128,
        "spoken_prefix_transition_cursor": 128,
        "cache_position": 129,
        "transition_token_ids": (11, 12, 13, 14, 15, 16),
        "listener_generation_id": 7,
        "session_generation_id": 7,
    }

    with pytest.raises(RuntimeError, match=r"token_index=1.*expected_frontier=130.*actual_frontier=129"):
        _validate_unified_say_packet_frontier(
            branch, 1, 130, kv_committed=True
        )


def test_native_control6_debug_input_capture_is_bounded_binary_and_opt_in(tmp_path) -> None:
    policy_input = SimpleNamespace(
        native_control_hidden=torch.arange(1536, dtype=torch.float32).reshape(1, 1536),
        native_control_phase_t=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        frame_id=55,
        generation_id=7,
        speaker_active=True,
    )
    decision = SimpleNamespace(
        action=SimpleNamespace(value="YIELD"),
        diagnostics={"native_control6_user_activity_detected": True},
    )
    row = _native_control6_debug_input_row(
        policy_input=policy_input,
        decision=decision,
        accepted=True,
        phase_profile="native1540",
    )
    assert row is not None
    assert _write_native_control6_debug_input_sidecar(None, session_id="s1", rows=[row]) is None
    path = _write_native_control6_debug_input_sidecar(
        tmp_path,
        session_id="s1",
        rows=[row],
    )
    assert path is not None
    with np.load(path, allow_pickle=False) as saved:
        assert saved["frame_ids"].tolist() == [55]
        assert saved["generation_ids"].tolist() == [7]
        assert saved["hidden"].shape == (1, 1536)
        assert saved["phase_t"].tolist() == [[1.0, 2.0, 3.0, 4.0]]
        assert saved["action"].tolist() == ["YIELD"]
        assert saved["accepted"].tolist() == [True]
        assert saved["user_activity_detected"].tolist() == [True]
        assert saved["phase_profile"].tolist() == ["native1540"]


def test_native_control6_debug_capture_uses_worker_scored_hidden_not_prehead_snapshot() -> None:
    policy_input = SimpleNamespace(
        native_control_hidden=None,
        native_control_phase_t=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        frame_id=12,
        generation_id=4,
        speaker_active=False,
    )
    hidden = np.linspace(-1.0, 1.0, 1536, dtype=np.float32)
    phase = np.asarray([0.125, -0.25, 0.5, -0.75], dtype=np.float32)
    diagnostics = {
        "native_control6_user_activity_detected": True,
        "_native_control6_debug_input": {
            "hidden": hidden,
            "phase_t": phase,
            "frame_id": 12,
            "generation_id": 4,
            "phase_profile": "native1540",
        },
    }
    decision = SimpleNamespace(action=SimpleNamespace(value="SPEAK"), diagnostics=diagnostics)

    row = _native_control6_debug_input_row(
        policy_input=policy_input,
        decision=decision,
        accepted=True,
        phase_profile="native1540",
    )

    assert row is not None
    assert np.array_equal(row["hidden"], hidden)
    assert np.array_equal(row["phase_t"], phase)
    assert row["frame_id"] == 12
    assert row["user_activity_detected"] is True
    assert "_native_control6_debug_input" not in diagnostics


def _unified_instruction_refresh_fixture(instruction: str = ""):
    target_instruction = "I" * 37

    class Tokenizer:
        audio_token = "<|audio|>"

        def __init__(self):
            self.messages = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            self.messages = messages
            return "serialized"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == self.audio_token:
                return [900]
            included = any(
                target_instruction in str(message.get("content", ""))
                for message in self.messages
            )
            before_marker = 189 + (37 if included else 0)
            return [1] + [22] * (before_marker - 1) + [900, 77]

    class Worker:
        def __init__(self, *, fail=False):
            self.calls = []
            self.fail = fail

        def refresh_unified_prefix_before_audio(
            self,
            session_id,
            prefix_token_ids,
            *,
            new_revision,
            expected_generation_id,
            expected_cache_position,
            prefix_import,
        ):
            self.calls.append((
                session_id,
                tuple(prefix_token_ids),
                new_revision,
                expected_generation_id,
                expected_cache_position,
                prefix_import,
            ))
            future = Future()
            if self.fail:
                future.set_exception(RuntimeError("injected owner refresh failure"))
            else:
                future.set_result(None)
            return future

    class Session:
        def __init__(self):
            self.session_id = "session"
            self.instructions = instruction
            self.semantic_context_revision = 0
            self._generation_lock = threading.RLock()
            self.active_response = False
            self.policy_response_pending = False
            self.history = []
            self.listener = SimpleNamespace(backend=SimpleNamespace(boa_token_id=901))
            self.listener_state = SimpleNamespace(
                _lock=threading.RLock(),
                next_frame_id=0,
                listener_input_frames=0,
                listener_state_version=0,
                gemma_lm_audio_frames=0,
                semantic_audio_soft_history=[],
            )
            self.audio_ingress = SimpleNamespace(
                stats=lambda: {"input_samples_received": 0, "input_frames_enqueued": 0},
                qsize=0,
                pending_samples=0,
            )
            self.history_context_notifications = 0
            self.history_context_notification_lock_owned = []

        def update_instructions(self, value, *, notify_callback=True):
            del notify_callback
            value = str(value)
            with self._generation_lock:
                if value == self.instructions:
                    return False
                self.instructions = value
                self.semantic_context_revision += 1
                return True

        def notify_history_context_changed(self):
            self.history_context_notifications += 1
            self.history_context_notification_lock_owned.append(
                self._generation_lock._is_owned()
            )

    class Dispatcher:
        def has_pending_work(self, _session_id):
            return False

    return Tokenizer(), Session(), Worker(), Dispatcher()


def test_unified_session_update_refreshes_190_token_prefix_to_227_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, session, worker, dispatcher = _unified_instruction_refresh_fixture()
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", "heard_action_say_v2")
    engine = SimpleNamespace(
        tokenizer=tokenizer,
        _thinker_wrapper=SimpleNamespace(
            processor=SimpleNamespace(audio_token="<|audio|>")
        ),
    )
    imports = []

    def make_import(_backend, token_ids, *, semantic_context_revision=0):
        importer = SimpleNamespace(
            token_ids=tuple(token_ids),
            semantic_context_revision=semantic_context_revision,
        )
        imports.append(importer)
        return importer

    monkeypatch.setattr("duplex_kernel.server.make_unified_prefix_import", make_import)
    initial_ids = _build_unified_listener_prefix_token_ids(engine, session)
    assert len(initial_ids) == 190
    binding = {
        "generation": 7,
        "batch_cache_position": len(initial_ids),
        "semantic_prefix_importer": make_import(session.listener.backend, initial_ids),
    }

    result = asyncio.run(_refresh_unified_prefix_for_instruction_update(
        engine,
        session,
        instructions="I" * 37,
        worker=worker,
        binding=binding,
        dispatcher=dispatcher,
    ))

    assert result == {"status": "refreshed", "prefix_len": 227, "revision": 1}
    assert session.instructions == "I" * 37
    assert session.semantic_context_revision == 1
    assert len(worker.calls) == 1
    call = worker.calls[0]
    assert call[:5] == ("session", imports[-1].token_ids, 1, 7, 190)
    assert imports[-1].semantic_context_revision == 1
    assert len(call[1]) == binding["batch_cache_position"] == 227
    assert binding["semantic_prefix_importer"] is imports[-1]
    assert session.history_context_notifications == 1
    assert session.history_context_notification_lock_owned == [False]

    # Repeating the same instruction is harmless even after audio begins; a
    # changed instruction is rejected without changing the committed session.
    session.listener_state.next_frame_id = 1
    session.listener_state.listener_input_frames = 1
    no_op = asyncio.run(_refresh_unified_prefix_for_instruction_update(
        engine,
        session,
        instructions="I" * 37,
        worker=worker,
        binding=binding,
        dispatcher=dispatcher,
    ))
    assert no_op["status"] == "unchanged"
    with pytest.raises(ValueError, match="before the first audio frame"):
        asyncio.run(_refresh_unified_prefix_for_instruction_update(
            engine,
            session,
            instructions="different instructions",
            worker=worker,
            binding=binding,
            dispatcher=dispatcher,
        ))


def test_backchannel_generation_budget_caps_real_tokenizer_ids_at_six() -> None:
    assert _controller_guided_say_generation_options(
        action="BACKCHANNEL",
        controller_guided_say=False,
        direct_audio_candidate=False,
        response_protocol=None,
    )["max_tokens"] == 6
    assert _controller_guided_say_generation_options(
        action="BACKCHANNEL",
        controller_guided_say=True,
        direct_audio_candidate=True,
        response_protocol="heard_action_say_v2",
    )["max_tokens"] == AUDIO_RESPONSE_V2_MAX_NEW_TOKENS


def test_unified_instruction_refresh_failure_rolls_back_instruction_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, session, worker, dispatcher = _unified_instruction_refresh_fixture()
    worker.fail = True
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", "heard_action_say_v2")
    engine = SimpleNamespace(
        tokenizer=tokenizer,
        _thinker_wrapper=SimpleNamespace(
            processor=SimpleNamespace(audio_token="<|audio|>")
        ),
    )
    monkeypatch.setattr(
        "duplex_kernel.server.make_unified_prefix_import",
        lambda _backend, token_ids, *, semantic_context_revision=0: SimpleNamespace(
            token_ids=tuple(token_ids),
            semantic_context_revision=semantic_context_revision,
        ),
    )
    binding = {
        "generation": 7,
        "batch_cache_position": 190,
        "semantic_prefix_importer": object(),
    }
    old_importer = binding["semantic_prefix_importer"]

    with pytest.raises(RuntimeError, match="injected owner refresh failure"):
        asyncio.run(_refresh_unified_prefix_for_instruction_update(
            engine,
            session,
            instructions="I" * 37,
            worker=worker,
            binding=binding,
            dispatcher=dispatcher,
        ))

    assert session.instructions == ""
    assert session.semantic_context_revision == 0
    assert binding["batch_cache_position"] == 190
    assert binding["semantic_prefix_importer"] is old_importer
    assert session.history_context_notifications == 0


def test_unified_native_result_waits_and_records_before_forward_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = []

    class FakeTensor:
        device = torch.device("cuda:0")
        is_cuda = True

        def __init__(self, name: str) -> None:
            self.name = name

        def record_stream(self, stream) -> None:
            assert stream is consumer
            operations.append(f"record:{self.name}")

    class FakeStream:
        def wait_event(self, event) -> None:
            assert event is ready_event
            operations.append("wait")

    ready_event = object()
    consumer = FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *, device: consumer)
    logits = FakeTensor("logits")
    hidden = FakeTensor("hidden")
    result = SimpleNamespace(
        logits=logits,
        final_norm_hidden=hidden,
        ready_event=ready_event,
    )

    # The returned pair is what forward_native_tokens exposes to the decoder;
    # all stream dependency and allocator-lifetime actions must precede it.
    returned = _consume_unified_native_text_append_result(result)
    assert returned == (logits, hidden)
    assert operations == ["wait", "record:logits", "record:hidden"]


def test_unified_cuda_result_without_ready_event_fails_closed() -> None:
    class FakeTensor:
        device = torch.device("cuda:0")
        is_cuda = True

        def record_stream(self, _stream) -> None:
            raise AssertionError("a missing ready event must fail before tensor use")

    output = FakeTensor()
    with pytest.raises(RuntimeError, match="missing its ready_event"):
        _consume_unified_native_text_append_result(SimpleNamespace(
            logits=output,
            final_norm_hidden=output,
            ready_event=None,
        ))


def test_unified_kv_switch_is_strict_0_or_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUPLEX_UNIFIED_KV", raising=False)
    assert _strict_env_switch("DUPLEX_UNIFIED_KV") is False
    monkeypatch.setenv("DUPLEX_UNIFIED_KV", "1")
    assert _strict_env_switch("DUPLEX_UNIFIED_KV") is True
    monkeypatch.setenv("DUPLEX_UNIFIED_KV", "0")
    assert _strict_env_switch("DUPLEX_UNIFIED_KV") is False
    monkeypatch.setenv("DUPLEX_UNIFIED_KV", "true")
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        _strict_env_switch("DUPLEX_UNIFIED_KV")


def test_unified_frame_logical_q_preserves_pending_user_audio() -> None:
    quiet = np.zeros(1280, dtype=np.float32)
    quiet_pending = _unified_listener_frame_contract(
        quiet,
        assistant_audio_queue_open=False,
        user_input_pending=True,
    )
    assert quiet_pending == (2, None, False)

    voiced = np.full(1280, 0.02, dtype=np.float32)
    assert _unified_listener_frame_contract(
        voiced,
        assistant_audio_queue_open=False,
        user_input_pending=False,
    ) == (2, None, False)


def test_unified_frame_logical_q_keeps_silence_and_assistant_playback_continuous() -> None:
    quiet = np.zeros(1280, dtype=np.float32)
    assert _unified_listener_frame_contract(
        quiet,
        assistant_audio_queue_open=False,
        user_input_pending=False,
    ) == (2, None, False)

    voiced = np.full(1280, 0.02, dtype=np.float32)
    assert _unified_listener_frame_contract(
        voiced,
        assistant_audio_queue_open=True,
        user_input_pending=False,
    ) == (2, None, False)


def test_assistant_queue_retention_includes_current_endpoint_but_not_pending_silence() -> None:
    quiet = np.zeros(1280, dtype=np.float32)
    pending_context = SimpleNamespace(
        policy_input=SimpleNamespace(peer_turn_active=False),
        speech_endpoint=False,
    )
    pending_voice = _shared_listener_queue_voice_activity(
        pending_context,
        lambda: True,
    )
    assert pending_voice is False
    assert _unified_listener_frame_contract(
        quiet,
        assistant_audio_queue_open=True,
        user_input_pending=True,
        voice_active=pending_voice,
    ) == (2, None, False)

    endpoint_context = SimpleNamespace(
        policy_input=SimpleNamespace(peer_turn_active=False),
        speech_endpoint=True,
    )
    endpoint_voice = _shared_listener_queue_voice_activity(
        endpoint_context,
        lambda: False,
    )
    assert endpoint_voice is True
    # The endpoint event is a retention signal only; it does not rewrite the
    # policy's peer_turn_active state.
    assert endpoint_context.policy_input.peer_turn_active is False
    assert _unified_listener_frame_contract(
        quiet,
        assistant_audio_queue_open=True,
        user_input_pending=False,
        voice_active=endpoint_voice,
    ) == (2, None, False)


def test_unified_listener_q0_requires_owner_queue_and_backchannel_keeps_causal_q2() -> None:
    voiced = np.full(1280, 0.02, dtype=np.float32)
    backchannel_branch = {"queue_open": False, "ephemeral_private": True}
    assistant_branch = {"queue_open": True, "ephemeral_private": False}

    assert not _unified_response_owns_shared_audio_queue(backchannel_branch)
    backchannel_frame = _unified_listener_frame_contract(
        voiced,
        assistant_audio_queue_open=_unified_response_owns_shared_audio_queue(
            backchannel_branch
        ),
        user_input_pending=True,
    )
    assert backchannel_frame == (2, None, False)

    assert _unified_response_owns_shared_audio_queue(assistant_branch)
    assistant_frame = _unified_listener_frame_contract(
        voiced,
        assistant_audio_queue_open=_unified_response_owns_shared_audio_queue(
            assistant_branch
        ),
        user_input_pending=False,
    )
    assert assistant_frame == (2, None, False)

    assert _unified_response_owns_shared_audio_queue(assistant_branch)


def test_shared_listener_response_branch_uses_session_binding_registry() -> None:
    branch = {"queue_open": True, "response_id": "r1"}
    binding = {"playback_ack_branches": {"r1": branch}}

    assert _shared_listener_response_branch(binding, "r1") is branch
    assert _shared_listener_response_branch(binding, "missing") is None
    assert _shared_listener_response_branch({}, "r1") is None


def test_response_branch_registry_attaches_after_session_registration() -> None:
    branches = {"r1": {"queue_open": True, "response_id": "r1"}}
    # The websocket creates its registry before shared-listener registration;
    # attach it only after registration produces the binding dictionary.
    binding = {"session": object()}

    attached = _attach_shared_listener_response_branches(binding, branches)

    assert attached is binding
    assert binding["playback_ack_branches"] is branches
    assert _shared_listener_response_branch(binding, "r1") is branches["r1"]


def test_shared_listener_frame_callback_keeps_private_backchannel_q2_and_cleans_up() -> None:
    voiced = np.full(1280, 0.02, dtype=np.float32)
    branches = {"bc1": {"queue_open": False, "ephemeral_private": True}}
    binding = {"playback_ack_branches": branches}

    assert _shared_listener_frame_queue_contract(
        binding,
        voiced,
        response_id="bc1",
        user_input_pending=True,
    ) == (2, None, False)
    assert _shared_listener_frame_queue_contract(
        binding,
        voiced,
        response_id="missing",
        user_input_pending=False,
    ) == (2, None, False)

    _clear_shared_listener_response_branches(binding, branches)
    assert "playback_ack_branches" not in binding


def test_shared_queue_ownership_is_independent_of_active_response_flag() -> None:
    voiced = np.full(1280, 0.02, dtype=np.float32)
    branches = {
        "shared": {"queue_open": True, "ephemeral_private": False},
        "private_bc": {"queue_open": True, "ephemeral_private": True},
        "closed": {"queue_open": False, "ephemeral_private": False},
    }
    binding = {"playback_ack_branches": branches}

    # Legacy queue metadata cannot suppress canonical Q2, including during prefill.
    assert _shared_listener_frame_queue_contract(
        binding,
        voiced,
        response_id="shared",
        user_input_pending=True,
    ) == (2, None, False)
    # A private BACKCHANNEL never owns canonical Q0, regardless of its queue flag.
    assert _shared_listener_frame_queue_contract(
        binding,
        voiced,
        response_id="private_bc",
        user_input_pending=True,
    ) == (2, None, False)
    # A stale or closed shared branch cannot suppress causal Q2.
    assert _shared_listener_frame_queue_contract(
        binding,
        voiced,
        response_id="closed",
        user_input_pending=True,
    ) == (2, None, False)


def test_private_transition_chunks_match_original_prompt_and_private_cursor() -> None:
    prompt = tuple(range(11, 21))
    offset = 0
    for chunk in (prompt[:5], prompt[5:8], prompt[8:]):
        offset = _validate_private_transition_chunk(
            prompt,
            offset,
            chunk,
            source_cache_position=190,
            current_cache_position=190 + offset,
        )
    assert offset == len(prompt)
    with pytest.raises(RuntimeError, match="original Thinker transition"):
        _validate_private_transition_chunk(
            prompt,
            5,
            (prompt[5], 99),
            source_cache_position=190,
            current_cache_position=195,
        )
    with pytest.raises(RuntimeError, match="original Thinker transition"):
        _validate_private_transition_chunk(
            prompt,
            0,
            tuple(range(11, 20)),
            source_cache_position=190,
            current_cache_position=190,
        )
    with pytest.raises(RuntimeError, match="private transition cursor"):
        _validate_private_transition_chunk(
            prompt,
            5,
            prompt[5:8],
            source_cache_position=190,
            current_cache_position=196,
        )


def test_unified_prefix_uses_text_template_and_replaces_only_final_audio_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        audio_token = "<|audio|>"

        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            self.messages = messages
            return "rendered chat with <|audio|> marker and trailing turn end"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == self.audio_token:
                return [900]
            return [11, 12, 900, 13]

    class Processor:
        audio_token = "<|audio|>"

        def __call__(self, **_kwargs):
            raise AssertionError("prefix construction must not call the audio processor")

    class Backend:
        boa_token_id = 901

    tokenizer = Tokenizer()
    engine = type("Engine", (), {
        "tokenizer": tokenizer,
        "_thinker_wrapper": type("Wrapper", (), {"processor": Processor()})(),
    })()
    session = type("Session", (), {
        "listener": type("Listener", (), {"backend": Backend()})(),
        "language": "Italian",
        "instructions": "Parla di Torino.",
        "history": [{"role": "user", "content": "turno precedente"}],
    })()
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", "heard_action_say_v2")

    prefix = _build_unified_listener_prefix_token_ids(engine, session)

    assert prefix == (11, 12, 901)
    assert tokenizer.messages[0]["role"] == "system"
    assert "Parla di Torino." in tokenizer.messages[0]["content"]
    assert tokenizer.messages[-2:] == [
        {"role": "user", "content": "turno precedente"},
        {"role": "user", "content": "<|audio|>"},
    ]


def test_unified_mode_accepts_stable_controller_guided_say_and_v2_protocols() -> None:
    assert _require_unified_stable_audio_protocol("heard_action_say_v2") == "heard_action_say_v2"
    assert _require_unified_stable_audio_protocol("controller_guided_say_v1") == "controller_guided_say_v1"


def test_unified_v1_prefix_is_stable_say_only_and_has_no_late_action_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, session, _worker, _dispatcher = _unified_instruction_refresh_fixture(
        "Parla di Torino."
    )
    session.language = "Italian"
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", "controller_guided_say_v1")
    engine = SimpleNamespace(
        tokenizer=tokenizer,
        _thinker_wrapper=SimpleNamespace(
            processor=SimpleNamespace(audio_token="<|audio|>")
        ),
    )

    prefix = _build_unified_listener_prefix_token_ids(engine, session)

    system = tokenizer.messages[0]["content"]
    assert "Parla di Torino." in system
    assert "soltanto il testo da pronunciare" in system
    assert "JSON" in system
    assert "SPEAK" not in system and "BACKCHANNEL" not in system
    assert prefix[-1] == session.listener.backend.boa_token_id


def test_unified_v1_backchannel_suffix_reuses_generic_say_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        audio_token = "<|audio|>"
        eot_token_id = 102

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert messages == [{"role": "user", "content": "<|audio|>"}]
            return "template with marker and turn end" + (" assistant prefix" if add_generation_prompt else "")

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == self.audio_token:
                return [900]
            return [100, 101, 900, 102, 103] if text.endswith(" assistant prefix") else [100, 101, 900, 102]

    class Backend:
        boa_token_id = 901
        eoa_token_id = 902
        bos_token_id = 100

    engine = SimpleNamespace(
        tokenizer=Tokenizer(),
        _thinker_wrapper=SimpleNamespace(processor=SimpleNamespace(audio_token="<|audio|>")),
    )
    session = SimpleNamespace(listener=SimpleNamespace(backend=Backend()))
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", "controller_guided_say_v1")

    assert _build_unified_backchannel_response_suffix(engine, session) == (
        _build_unified_audio_response_suffix(engine, session)
    )


def test_unified_suffix_closes_audio_user_turn_and_starts_assistant_without_processor() -> None:
    class Tokenizer:
        audio_token = "<|audio|>"
        eot_token_id = 102

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert messages == [{"role": "user", "content": "<|audio|>"}]
            return "template with marker and turn end" + (" assistant prefix" if add_generation_prompt else "")

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == "<|audio|>":
                return [900]
            if text.endswith(" assistant prefix"):
                return [100, 101, 900, 102, 103]
            return [100, 101, 900, 102]

    class Backend:
        boa_token_id = 901
        eoa_token_id = 902
        bos_token_id = 100

    tokenizer = Tokenizer()
    engine = type("Engine", (), {
        "tokenizer": tokenizer,
        "_thinker_wrapper": type("Wrapper", (), {"processor": type("Processor", (), {"audio_token": "<|audio|>"})()})(),
    })()
    session = type("Session", (), {
        "listener": type("Listener", (), {"backend": Backend()})(),
    })()

    assert _build_unified_audio_response_suffix(engine, session) == (902, 102, 103)
    assert _build_unified_next_user_header(engine, session) == (101, 901)
    assert _unified_end_of_turn_token_id(engine, session) == 102


def test_unified_template_preserves_newline_and_model_generation_prompt_after_eot() -> None:
    class Tokenizer:
        audio_token = "<|audio|>"
        eot_token = "<turn|>"
        eot_token_id = 106

        def apply_chat_template(self, _messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            tail = "<turn|>\n"
            if add_generation_prompt:
                tail += "<|turn>model\n"
            return "<bos><|turn>user\n<|audio|>" + tail

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == self.audio_token:
                return [258881]
            if text.endswith("<|turn>model\n"):
                return [1, 105, 258881, 106, 107, 105, 4368, 107]
            return [1, 105, 258881, 106, 107]

    class Backend:
        boa_token_id = 258882
        eoa_token_id = 258883
        bos_token_id = 1

    engine = SimpleNamespace(
        tokenizer=Tokenizer(),
        _thinker_wrapper=SimpleNamespace(processor=SimpleNamespace(audio_token="<|audio|>")),
    )
    session = SimpleNamespace(listener=SimpleNamespace(backend=Backend()))

    assert _unified_end_of_turn_token_id(engine, session) == 106
    assert _build_unified_audio_response_suffix(engine, session) == (
        258883, 106, 107, 105, 4368, 107
    )


def test_unified_template_rejects_duplicate_eot_and_incompatible_audio_marker() -> None:
    class Tokenizer:
        audio_token = "<|audio|>"
        eot_token_id = 106

        def __init__(self, *, duplicate_eot=False, incompatible_marker=False):
            self.duplicate_eot = duplicate_eot
            self.incompatible_marker = incompatible_marker

        def apply_chat_template(self, _messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            tail = "<turn|><turn|>\n" if self.duplicate_eot else "<turn|>\n"
            return "<bos><|turn>user\n<|audio|>" + tail

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == self.audio_token:
                return [258881, 258882] if self.incompatible_marker else [258881]
            if self.duplicate_eot:
                return [1, 105, 258881, 106, 106, 107]
            return [1, 105, 258881, 106, 107]

    class Backend:
        boa_token_id = 258882
        eoa_token_id = 258883
        bos_token_id = 1

    def engine_for(tokenizer):
        return SimpleNamespace(
            tokenizer=tokenizer,
            _thinker_wrapper=SimpleNamespace(processor=SimpleNamespace(audio_token="<|audio|>")),
        )

    session = SimpleNamespace(listener=SimpleNamespace(backend=Backend()))
    with pytest.raises(RuntimeError, match="exactly one configured EOT token"):
        _unified_end_of_turn_token_id(engine_for(Tokenizer(duplicate_eot=True)), session)
    with pytest.raises(RuntimeError, match="audio marker must map to exactly one tokenizer token"):
        _unified_end_of_turn_token_id(engine_for(Tokenizer(incompatible_marker=True)), session)


def test_unified_thinker_uses_owner_rebased_cursor_only_for_first_append() -> None:
    owner_calls = []
    callback_order = []

    class Worker(_DecodeBudgetWorkerMixin):
        def __init__(self) -> None:
            self.cursor = 20
            self.cache_identity = object()
            self.append_attempts = []
            self.append_positions = []
            self.append_rebase_flags = []
            self.append_scope_ids = []
            self.committed_tokens = []

        def session_state(self, _session_id):
            future = Future()
            future.set_result(SimpleNamespace(
                generation_id=12,
                cache_position=self.cursor,
                valid_history_tokens=self.cursor,
            ))
            owner_calls.append(("state", self.cache_identity))
            return future

        def append_text_tokens(
            self, session_id, token_ids, *, expected_generation_id,
            expected_cache_position, allow_initial_cursor_rebase=False,
            append_scope_id=None, response_cancel_event=None,
            consume_reserved_tokens=None,
        ):
            callback_order.append(("append", tuple(token_ids)))
            assert session_id == "session"
            assert expected_generation_id == 12
            self.append_attempts.append(tuple(token_ids))
            self.append_positions.append(expected_cache_position)
            self.append_rebase_flags.append(allow_initial_cursor_rebase)
            self.append_scope_ids.append(append_scope_id)
            assert response_cancel_event is None
            if len(self.append_attempts) == 1:
                assert allow_initial_cursor_rebase is True
                assert expected_cache_position == 20
                self.cursor += 2  # A listener Q2 append wins the race before text commit.
            else:
                assert allow_initial_cursor_rebase is False
                assert expected_cache_position == self.cursor
            assert expected_cache_position <= self.cursor
            self.committed_tokens.extend(token_ids)
            owner_calls.append(("append", self.cache_identity, tuple(token_ids)))
            self.cursor += len(token_ids)
            future = Future()
            future.set_result(SimpleNamespace(
                logits=torch.zeros((1, 100), dtype=torch.float32),
                final_norm_hidden=torch.full(
                    (1, 1, 4), float(token_ids[-1]), dtype=torch.float32
                ),
                ready_event=None,
                cache_position=self.cursor,
            ))
            return future

    class ForbiddenAudioPath:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("unified decode must not call processor/audio tower")

    class Sampler:
        eos_token_ids = {99}

        def __init__(self) -> None:
            self.next_tokens = iter((7, 8, 99))

        def sample_initial(self, logits, _generated_ids):
            callback_order.append(("sample_initial",))
            token_id = next(self.next_tokens)
            return torch.tensor([[token_id]]), token_id, logits

        def sample_token(self, logits, _generated_ids):
            token_id = next(self.next_tokens)
            return torch.tensor([[token_id]]), token_id, logits

        def select_valid_candidate(self, candidate, *, source_logits, generated_ids):
            del source_logits, generated_ids
            return candidate

    sampler = Sampler()
    worker = Worker()
    sampler_kwargs = {}
    captured_metrics = []
    recorded_tokens = []
    first_packet_timestamps = []

    def on_thinker_metrics(metrics):
        snapshot = dict(metrics)
        captured_metrics.append(snapshot)
        callback_order.append(("profile_publish",))
        profile = snapshot.get("unified_native_profile", {})
        published_ns = profile.get("first_packet_published_at_ns")
        if isinstance(published_ns, int):
            callback_order.append(("profile_packet_published", published_ns))

    def build_sampler(**kwargs):
        sampler_kwargs.update(kwargs)
        return sampler

    engine = SimpleNamespace(
        thinker=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ",".join(map(str, ids))),
        device="cpu",
        dtype=torch.float32,
        _thinker_wrapper=SimpleNamespace(processor=ForbiddenAudioPath()),
        audio_tower=ForbiddenAudioPath(),
        think=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unified mode must not fall back to engine.think")
        ),
        build_thinker_sampler=build_sampler,
    )
    session = SimpleNamespace(session_id="session")
    branch = {
        "listener_generation_id": 12,
        "response_id": "resp-test",
        "cache_position": 20,
        "turn_start_cache_position": 20,
        "native_token_append_count": 0,
        "policy_decision_at_ns": time.monotonic_ns() - 10_000_000,
    }

    def on_token_packet(packet):
        packet_cursor = _validate_unified_say_packet_frontier(
            branch,
            int(packet.token_index),
            int(branch["spoken_prefix_base_cursor"]) + int(packet.token_index) + 1,
            kv_committed=bool(packet.kv_committed),
        )
        callback_order.append((
            "packet", packet.token_index, branch["cache_position"], packet_cursor
        ))
        assert packet.hidden_provenance == ("transition" if packet.token_index == 0 else "token")
        assert packet.kv_committed is (packet.token_index != 0)
        assert packet.packet_published_ns is None

    def on_token_committed(packet):
        _validate_unified_say_packet_frontier(
            branch,
            int(packet.token_index),
            int(branch["spoken_prefix_base_cursor"]) + int(packet.token_index) + 1,
            kv_committed=True,
        )
        callback_order.append(("commit", packet.token_index, branch["cache_position"]))
        assert packet.kv_committed is True
        assert packet.hidden_provenance == "transition"

    def on_first_token(packet):
        callback_order.append(("first", packet.token_index, packet.packet_published_ns))
        assert packet.packet_published_ns is not None
        assert packet.kv_committed is False
        first_packet_timestamps.append(packet.packet_published_ns)

    def run():
        return _run_unified_native_thinker(
            engine,
            session,
            {
                "_on_thinker_metrics": on_thinker_metrics,
                "_on_thinker_token_id": recorded_tokens.append,
            },
            worker=worker,
            transition_token_ids=(3, 4, 5, 6, 7, 8),
            think_kwargs={
                "temperature": 0.0,
                "top_p": 1.0,
                "min_new_tokens": 0,
                "max_new_tokens": 4,
                "anticipate_first_token": True,
                "on_token_packet": on_token_packet,
                "on_token_committed": on_token_committed,
                "on_first_token": on_first_token,
                "on_generated_token": lambda *_args: pytest.fail(
                    "legacy sampler callback must be disabled for anticipated V1 packets"
                ),
            },
            generation_id=4,
            branch=branch,
        )
    think = run()
    assert worker.append_attempts == [(3, 4, 5, 6, 7, 8), (7,), (8,)]
    assert worker.append_positions == [20, 28, 29]
    assert worker.append_rebase_flags == [True, False, False]
    assert worker.append_scope_ids == ["resp-test", "resp-test", "resp-test"]
    assert worker.committed_tokens == [3, 4, 5, 6, 7, 8, 7, 8]
    assert worker.cursor == branch["cache_position"] == 30
    assert branch["turn_start_cache_position"] == 22
    assert branch["spoken_prefix_base_cursor"] == 28
    assert branch["spoken_prefix_transition_cursor"] == 28
    assert branch["spoken_prefix_exclusive_cursor"] == 28
    assert sampler_kwargs["on_generated_token"] is None
    assert callback_order.index(("packet", 0, 28, 29)) < callback_order.index(("append", (7,)))
    assert callback_order.index(("append", (7,))) < callback_order.index(("commit", 0, 29))
    assert callback_order.index(("append", (8,))) < callback_order.index(("packet", 1, 30, 30))
    assert callback_order[-1][0] == "first" or any(item[0] == "first" for item in callback_order)
    timestamped_profiles = [
        item for item in callback_order if item[0] == "profile_packet_published"
    ]
    assert timestamped_profiles
    assert callback_order.index(timestamped_profiles[0]) > callback_order.index(
        ("commit", 0, 29)
    )
    # First-logits timing is retained for final metrics, but publishing those
    # metrics must not block sampling the anticipated first token.
    assert callback_order.index(("sample_initial",)) < callback_order.index(
        ("profile_publish",)
    )

    assert all(call[1] is worker.cache_identity for call in owner_calls)
    successful_appends = [call for call in owner_calls if call[0] == "append"]
    assert [call[2] for call in successful_appends] == [(3, 4, 5, 6, 7, 8), (7,), (8,)]
    assert recorded_tokens == [7, 8, 99]
    assert think["gen_ids"] == [7, 8]
    assert think["hidden"].shape == (1, 2, 4)
    assert think["runtime_metrics"]["unified_native_cache"]["owner_action"] == "handle_terminal_eos"
    assert captured_metrics[-1]["first_thinker_token_at_ns"] >= captured_metrics[-1][
        "first_thinker_token_hidden_ready_at_ns"
    ]
    assert captured_metrics[-1]["ttft_ms"] < captured_metrics[-1][
        "first_packet_hidden_ready_ms"
    ]
    final_profile = captured_metrics[-1]["unified_native_profile"]
    assert final_profile["first_packet_published_at_ns"] == first_packet_timestamps[0]
    packet_entry_ns = final_profile["token0_packet_callback_entry_ns"]
    packet_exit_ns = final_profile["token0_packet_callback_exit_ns"]
    token0_submit_ns = final_profile["token0_forward_submit_ns"]
    token0_return_ns = final_profile["token0_forward_return_ns"]
    commit_entry_ns = final_profile["token0_committed_callback_entry_ns"]
    commit_exit_ns = final_profile["token0_committed_callback_exit_ns"]
    assert packet_entry_ns <= packet_exit_ns <= token0_submit_ns
    assert token0_submit_ns <= token0_return_ns <= commit_entry_ns <= commit_exit_ns
    assert final_profile["token0_append_request"] == {
        "response_id": "resp-test",
        "generation_id": 4,
        "listener_generation_id": 12,
        "q_len": 1,
        "expected_cache_position": 28,
        "append_scope_id": "resp-test",
    }


def test_cancel_during_token0_forward_keeps_late_metrics_local_only() -> None:
    cancellation_event = threading.Event()
    local_profiles: list[dict] = []
    websocket_events: list[dict] = []
    fifo_records: list[dict] = []

    class Worker(_DecodeBudgetWorkerMixin):
        def __init__(self) -> None:
            self.cursor = 20

        def session_state(self, _session_id):
            future = Future()
            future.set_result(SimpleNamespace(
                generation_id=12,
                cache_position=self.cursor,
                valid_history_tokens=self.cursor,
            ))
            return future

        def append_text_tokens(
            self,
            _session_id,
            token_ids,
            *,
            expected_generation_id,
            expected_cache_position,
            allow_initial_cursor_rebase=False,
            append_scope_id=None,
            response_cancel_event=None,
            consume_reserved_tokens=None,
        ):
            token_ids = tuple(int(token_id) for token_id in token_ids)
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            assert response_cancel_event is cancellation_event
            if token_ids == (7,):
                cancellation_event.set()
            self.cursor += len(token_ids)
            future = Future()
            future.set_result(SimpleNamespace(
                logits=torch.zeros((1, 100), dtype=torch.float32),
                final_norm_hidden=torch.full((1, 1, 4), float(token_ids[-1])),
                ready_event=None,
                cache_position=self.cursor,
            ))
            return future

    class Sampler:
        eos_token_ids = {99}

        def __init__(self) -> None:
            self.next_tokens = iter((7, 99))

        def sample_initial(self, logits, _generated_ids):
            token_id = next(self.next_tokens)
            return torch.tensor([[token_id]]), token_id, logits

        def sample_token(self, logits, _generated_ids):
            token_id = next(self.next_tokens)
            return torch.tensor([[token_id]]), token_id, logits

        def select_valid_candidate(self, candidate, *, source_logits, generated_ids):
            del source_logits, generated_ids
            return candidate

    worker = Worker()
    sampler = Sampler()
    engine = SimpleNamespace(
        thinker=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ",".join(map(str, ids))),
        device="cpu",
        dtype=torch.float32,
        _thinker_wrapper=SimpleNamespace(processor=object()),
        audio_tower=object(),
        build_thinker_sampler=lambda **_kwargs: sampler,
    )
    session = SimpleNamespace(
        session_id="session",
        is_generation_current=lambda _token: not cancellation_event.is_set(),
    )

    def publish_metrics(values: dict) -> None:
        before = [dict(packet) for packet in fifo_records]
        _publish_thinker_metrics_event(
            values,
            response_id="resp-cancelled",
            policy_decision_at_ns=None,
            session=session,
            token=object(),
            cancellation_event=cancellation_event,
            update_runtime_metrics=lambda _rid, _phase, metric_values: local_profiles.append(
                dict(metric_values)
            ),
            send_event=lambda event: websocket_events.append(event) or False,
        )
        assert fifo_records == before

    result = _run_unified_native_thinker(
        engine,
        session,
        {
            "_response_cancel_event": cancellation_event,
            "_on_thinker_metrics": publish_metrics,
        },
        worker=worker,
        transition_token_ids=(1, 2, 3, 4, 5, 6),
        think_kwargs={
            "temperature": 0.0,
            "top_p": 1.0,
            "min_new_tokens": 0,
            "max_new_tokens": 1,
            "anticipate_first_token": True,
            "on_token_packet": lambda packet: fifo_records.append({
                "token_id": int(packet.token_id),
                "kv_committed": bool(packet.kv_committed),
            }),
            "on_token_committed": lambda packet: fifo_records.__setitem__(
                int(packet.token_index),
                {"token_id": int(packet.token_id), "kv_committed": True},
            ),
        },
        generation_id=4,
        branch={
            "listener_generation_id": 12,
            "response_id": "resp-cancelled",
            "cache_position": 20,
            "turn_start_cache_position": 20,
            "native_token_append_count": 0,
            "policy_decision_at_ns": time.monotonic_ns() - 1_000_000,
        },
    )

    assert cancellation_event.is_set()
    assert result["gen_ids"] == [7]
    assert fifo_records == [{"token_id": 7, "kv_committed": True}]
    assert local_profiles and all(
        "unified_native_profile" in values for values in local_profiles
    )
    assert websocket_events == []


def test_failed_early_say_packet_rolls_back_before_the_next_response() -> None:
    class Worker(_DecodeBudgetWorkerMixin):
        def __init__(self) -> None:
            self.cursor = 20
            self.append_calls: list[tuple[int, ...]] = []
            self.rollback_targets: list[int] = []

        @staticmethod
        def done(value=None):
            future = Future()
            future.set_result(value)
            return future

        def session_state(self, _session_id):
            return self.done(SimpleNamespace(
                generation_id=12,
                cache_position=self.cursor,
                valid_history_tokens=self.cursor,
            ))

        def append_text_tokens(
            self,
            _session_id,
            token_ids,
            *,
            expected_generation_id,
            expected_cache_position,
            allow_initial_cursor_rebase=False,
            append_scope_id=None,
            response_cancel_event=None,
            consume_reserved_tokens=None,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            assert append_scope_id == "response"
            assert response_cancel_event is None
            token_ids = tuple(int(value) for value in token_ids)
            self.append_calls.append(token_ids)
            self.cursor += len(token_ids)
            return self.done(SimpleNamespace(
                logits=torch.zeros((1, 100), dtype=torch.float32),
                final_norm_hidden=torch.zeros((1, 1, 4), dtype=torch.float32),
                ready_event=None,
                cache_position=self.cursor,
            ))

        def rollback_assistant_branch(
            self,
            _session_id,
            *,
            expected_generation_id,
            expected_cache_position,
            target_cache_position,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            self.rollback_targets.append(int(target_cache_position))
            self.cursor = int(target_cache_position)
            return self.done(SimpleNamespace(cache_position=self.cursor))

        def drain_queued_audio(
            self,
            _session_id,
            *,
            expected_generation_id,
            expected_cache_position,
            user_turn_already_open,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            assert user_turn_already_open is True
            return self.done(SimpleNamespace(cache_position=self.cursor))

    class Sampler:
        eos_token_ids = {99}

        def __init__(self) -> None:
            self.next_tokens = iter((7, 99))

        def sample_initial(self, logits, _generated_ids):
            token_id = next(self.next_tokens)
            return torch.tensor([[token_id]]), token_id, logits

        def sample_token(self, logits, _generated_ids):
            token_id = next(self.next_tokens)
            return torch.tensor([[token_id]]), token_id, logits

        def select_valid_candidate(self, candidate, *, source_logits, generated_ids):
            del source_logits, generated_ids
            return candidate

    worker = Worker()
    engine = SimpleNamespace(
        thinker=SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            tokenizer=SimpleNamespace(
                decode=lambda ids, skip_special_tokens=True: ",".join(map(str, ids))
            ),
        ),
        tokenizer=SimpleNamespace(
            decode=lambda ids, skip_special_tokens=True: ",".join(map(str, ids))
        ),
        device="cpu",
        dtype=torch.float32,
        _thinker_wrapper=SimpleNamespace(processor=object()),
        audio_tower=object(),
        think=lambda *_args, **_kwargs: pytest.fail("unexpected legacy Thinker path"),
        build_thinker_sampler=lambda **_kwargs: Sampler(),
    )
    session = SimpleNamespace(session_id="session")

    def branch_for_response():
        return {
            "listener_generation_id": 12,
            "response_id": "response",
            "cache_position": worker.cursor,
            "turn_start_cache_position": worker.cursor,
            "native_token_append_count": 0,
            "queue_open": True,
            "finalized": False,
            "spoken_prefix_exclusive_cursor": worker.cursor,
        }

    first = branch_for_response()
    with pytest.raises(RuntimeError, match="injected early packet failure"):
        _run_unified_native_thinker(
            engine,
            session,
            {},
            worker=worker,
            transition_token_ids=(1, 2, 3, 4, 5, 6),
            think_kwargs={
                "temperature": 0.0,
                "top_p": 1.0,
                "min_new_tokens": 0,
                "max_new_tokens": 2,
                "anticipate_first_token": True,
                "on_token_packet": lambda _packet: (_ for _ in ()).throw(
                    RuntimeError("injected early packet failure")
                ),
            },
            generation_id=4,
            branch=first,
        )

    assert first["cache_position"] == 26  # Transition suffix committed before callback failure.
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        first,
        history_committed=False,
        close_ids=(90, 91),
    ))
    assert worker.rollback_targets == [20]
    assert worker.cursor == 20
    assert first["finalized"] is True

    second = branch_for_response()
    second_packets = []
    result = _run_unified_native_thinker(
        engine,
        session,
        {},
        worker=worker,
        transition_token_ids=(11, 12, 13, 14, 15, 16),
        think_kwargs={
            "temperature": 0.0,
            "top_p": 1.0,
            "min_new_tokens": 0,
            "max_new_tokens": 2,
            "anticipate_first_token": True,
            "on_token_packet": lambda packet: (
                _validate_unified_say_packet_frontier(
                    second,
                    int(packet.token_index),
                    int(second["spoken_prefix_base_cursor"]) + 1,
                    kv_committed=bool(packet.kv_committed),
                ),
                second_packets.append(packet),
            ),
            "on_token_committed": lambda packet: _validate_unified_say_packet_frontier(
                second,
                int(packet.token_index),
                int(second["spoken_prefix_base_cursor"]) + 1,
                kv_committed=True,
            ),
        },
        generation_id=5,
        branch=second,
    )
    assert result["gen_ids"] == [7]
    assert len(second_packets) == 1 and second_packets[0].kv_committed is False
    assert second["spoken_prefix_base_cursor"] == 26
    assert worker.cursor == second["cache_position"] == 27
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        second,
        history_committed=False,
        close_ids=(90, 91),
    ))
    assert worker.rollback_targets == [20, 20]
    assert worker.cursor == 20


class _FutureWorker:
    def __init__(self, *, cursor: int, turn_start: int, capacity: int, queued_tokens: int = 0,
                 fail_append_at: int | None = None, reserve_watermark_failures: int = 0) -> None:
        self.cursor = cursor
        self.turn_start = turn_start
        self.capacity = capacity
        self.queued_tokens = queued_tokens
        self.fail_append_at = fail_append_at
        self.reserve_watermark_failures = reserve_watermark_failures
        self.append_calls: list[tuple[tuple[int, ...], int]] = []
        self.rollback_targets: list[int] = []
        self.drain_open_modes: list[bool] = []

    @staticmethod
    def _done(value=None, error: BaseException | None = None):
        future = Future()
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)
        return future

    def session_state(self, _session_id):
        return self._done(SimpleNamespace(generation_id=12, cache_position=self.cursor))

    def reserve_tail_tokens(
        self, _session_id, *, expected_generation_id, expected_cache_position, token_count
    ):
        assert expected_generation_id == 12
        assert expected_cache_position == self.cursor
        if self.reserve_watermark_failures:
            self.reserve_watermark_failures -= 1
            return self._done(error=RuntimeError(
                "tail reservation requires a completed listener frame watermark"
            ))
        if self.cursor + self.queued_tokens + token_count > self.capacity:
            return self._done(error=RuntimeError("native listener tail reservation exceeds capacity"))
        return self._done(SimpleNamespace(cache_position=self.cursor))

    def append_text_tokens(
        self, _session_id, token_ids, *, expected_generation_id, expected_cache_position,
        consume_reserved_tokens,
    ):
        assert expected_generation_id == 12
        assert expected_cache_position == self.cursor
        token_ids = tuple(token_ids)
        self.append_calls.append((token_ids, consume_reserved_tokens))
        self.cursor += len(token_ids)
        if self.fail_append_at == len(self.append_calls):
            return self._done(error=RuntimeError("injected native close append failure"))
        return self._done(SimpleNamespace(
            logits=torch.zeros((1, 8)),
            final_norm_hidden=torch.zeros((1, 1, 4)),
            ready_event=None,
        ))

    def rollback_assistant_branch(
        self, _session_id, *, expected_generation_id, expected_cache_position,
        target_cache_position,
    ):
        assert expected_generation_id == 12
        assert expected_cache_position == self.cursor
        self.rollback_targets.append(target_cache_position)
        self.cursor = target_cache_position
        return self._done(SimpleNamespace(cache_position=self.cursor))

    def drain_queued_audio(
        self, _session_id, *, expected_generation_id, expected_cache_position,
        user_turn_already_open,
    ):
        assert expected_generation_id == 12
        assert expected_cache_position == self.cursor
        self.drain_open_modes.append(user_turn_already_open)
        self.cursor += self.queued_tokens
        self.queued_tokens = 0
        branch = getattr(self, "queue_branch", None)
        if branch is not None:
            self.drain_queue_boundary_observation = (
                branch["queue_open"],
                branch["queue_epoch_state"]["value"],
                branch.get("queue_close_epoch"),
            )
            lock = branch.get("transition_lock")
            if lock is not None:
                acquired = lock.acquire(blocking=False)
                self.drain_was_submitted_under_transition_lock = not acquired
                if acquired:
                    lock.release()
        return self._done(SimpleNamespace(
            cache_position=self.cursor,
            queued_audio_drain_metrics=getattr(self, "drain_metrics", None),
        ))


def _unified_branch(cursor: int, turn_start: int) -> dict[str, object]:
    return {
        "listener_generation_id": 12,
        "cache_position": cursor,
        "turn_start_cache_position": turn_start,
        "queue_open": True,
        "finalized": False,
    }


def test_unified_close_capacity_preflight_rolls_back_before_any_close_append() -> None:
    worker = _FutureWorker(cursor=95, turn_start=80, capacity=100, queued_tokens=4)
    branch = _unified_branch(cursor=95, turn_start=80)

    with pytest.raises(RuntimeError, match="assistant branch was rolled back"):
        asyncio.run(_finalize_unified_native_cache_branch(
            worker,
            "session",
            branch,
            history_committed=True,
            close_ids=(31, 32, 33),
        ))

    assert worker.append_calls == []
    assert worker.rollback_targets == [80]
    assert worker.drain_open_modes == [True]
    assert worker.cursor == 84
    assert branch["finalized"] is True
    assert branch["queue_open"] is False


def test_unified_finalizer_captures_worker_drain_metrics_without_changing_cursor() -> None:
    worker = _FutureWorker(cursor=70, turn_start=60, capacity=100)
    worker.drain_metrics = SimpleNamespace(
        elapsed_ms=17.25,
        frame_count=3,
        token_count=6,
        retained_context_source="assistant_phase_q2_queue",
        source_frame_ids=(41, 42, 43),
        queue_response_id="response-finalized",
        queue_epoch="epoch-2",
        queue_first_frame_id=41,
    )
    branch = _unified_branch(cursor=70, turn_start=60)

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=True,
        close_ids=(71, 72),
    ))

    assert worker.cursor == 72
    flattened_metrics = branch["queued_audio_drain_metrics"]
    assert flattened_metrics["drain_transition_lock_wait_ms"] >= 0.0
    assert flattened_metrics["drain_completion_wait_ms"] >= 0.0
    assert {
        key: value for key, value in flattened_metrics.items()
        if key not in {"drain_transition_lock_wait_ms", "drain_completion_wait_ms"}
    } == {
        "queued_audio_drain_ms": 17.25,
        "queued_audio_drain_frames": 3,
        "queued_audio_drain_tokens": 6,
        "queued_audio_retained_context_source": "assistant_phase_q2_queue",
        "queued_audio_source_frame_ids": [41, 42, 43],
        "queued_audio_response_id": "response-finalized",
        "queued_audio_epoch": "epoch-2",
        "queued_audio_first_frame_id": 41,
        "drain_queue_wait_ms": 0.0,
        "owner_wait_ms": 0.0,
        "drain_replay_forward_ms": 0.0,
        "queued_audio_source_queue_epochs": [],
        "retained_frames_count": 3,
        "drain_total_elapsed_ms": 17.25,
        "queued_audio_queue_open_epoch": None,
        "queued_audio_queue_close_epoch": 1,
    }


def test_unified_queue_close_epoch_and_drain_submission_share_transition_lock() -> None:
    worker = _FutureWorker(cursor=70, turn_start=60, capacity=100)
    transition_lock = threading.Lock()
    branch = _unified_branch(cursor=70, turn_start=60)
    branch.update({
        "transition_lock": transition_lock,
        "queue_epoch_state": {"value": 9},
        "queue_open_epoch": 9,
        "queue_boundary_epoch": 9,
    })
    worker.queue_branch = branch

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=True,
        close_ids=(71, 72),
    ))

    assert worker.drain_queue_boundary_observation == (False, 10, 10)
    assert worker.drain_was_submitted_under_transition_lock is True
    assert branch["queue_open"] is False
    assert branch["queue_close_epoch"] > branch["queue_open_epoch"]
    assert isinstance(branch["queue_closed_at_ns"], int)
    closed_binding = {"playback_ack_branches": {"response": branch}}
    assert _shared_listener_frame_queue_contract(
        closed_binding,
        np.full(1280, 0.02, dtype=np.float32),
        response_id="response",
        user_input_pending=False,
    ) == (2, None, False)


def test_unified_close_append_failure_rolls_back_partial_suffix_and_drains_open_turn() -> None:
    worker = _FutureWorker(
        cursor=70,
        turn_start=60,
        capacity=100,
        queued_tokens=2,
        fail_append_at=2,
    )
    branch = _unified_branch(cursor=70, turn_start=60)

    with pytest.raises(RuntimeError, match="assistant branch was rolled back"):
        asyncio.run(_finalize_unified_native_cache_branch(
            worker,
            "session",
            branch,
            history_committed=True,
            close_ids=(41, 42, 43, 44, 45),
        ))

    assert worker.append_calls == [((41, 42), 2), ((43, 44), 2)]
    assert worker.rollback_targets == [60]
    assert worker.drain_open_modes == [True]
    assert worker.cursor == 62
    assert branch["finalized"] is True
    assert branch["queue_open"] is False


def test_unified_success_consumes_reserved_close_suffix_then_drains_new_user_turn() -> None:
    worker = _FutureWorker(cursor=70, turn_start=60, capacity=100, queued_tokens=2)
    branch = _unified_branch(cursor=70, turn_start=60)

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=True,
        close_ids=(51, 52, 53),
    ))

    assert worker.append_calls == [((51, 52), 2), ((53,), 1)]
    assert worker.rollback_targets == []
    assert worker.drain_open_modes == [False]
    assert worker.cursor == 75
    assert branch["finalized"] is True
    assert branch["history_committed"] is True


def test_private_unified_finalizer_releases_only_ephemeral_slot() -> None:
    class Worker:
        def __init__(self):
            self.released = []

        @staticmethod
        def _done(value=None):
            future = Future()
            future.set_result(value)
            return future

        def end_ephemeral_decode(self, handle):
            self.released.append(handle)
            return self._done(None)

        def session_state(self, *_args, **_kwargs):
            raise AssertionError("private cleanup must not inspect canonical session state")

        def rollback_assistant_branch(self, *_args, **_kwargs):
            raise AssertionError("private cleanup must not rewind canonical cache")

        def drain_queued_audio(self, *_args, **_kwargs):
            raise AssertionError("private cleanup must not touch the canonical audio queue")

    worker = Worker()
    handle = SimpleNamespace(cache_position=117)
    branch = {
        "ephemeral_private": True,
        "ephemeral_decode_handle": handle,
        "cache_position": 115,
        "finalized": False,
    }

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=True,
        close_ids=(1, 2),
    ))

    assert worker.released == [handle]
    assert branch["cache_position"] == 117
    assert branch["finalized"] is True
    assert branch["private_slot_released"] is True
    assert branch["history_committed"] is False


def test_response_scoped_speech_ack_uses_real_release_then_ack_path_and_rejects_stale() -> None:
    class Stream:
        def __init__(self, generation_id: int, segment_id: int) -> None:
            self.generation_id = generation_id
            self.segment_id = segment_id
            self.acked: list[tuple[int, int]] = []

        def ack_emitted_audio_frame(self, frame_id: int, *, generation_id: int, segment_id: int) -> None:
            self.acked.append((int(frame_id), int(generation_id)))

    first = {
        "response_id": "resp-one",
        "session_generation_id": 11,
        "speaker_segment_id": "seg-one",
        "spoken_prefix_lock": threading.Lock(),
    }
    second = {
        "response_id": "resp-two",
        "session_generation_id": 12,
        "speaker_segment_id": "seg-two",
        "spoken_prefix_lock": threading.Lock(),
    }
    first_stream = Stream(11, 1)
    second_stream = Stream(12, 2)
    assert _register_response_speech_ack_streams(
        first,
        response_id="resp-one",
        generation_id=11,
        response_segment_id="seg-one",
        live_stream=first_stream,
        closed_unit_streams={},
        closed_unit_events={},
        required=True,
    )
    assert _register_response_speech_ack_streams(
        second,
        response_id="resp-two",
        generation_id=12,
        response_segment_id="seg-two",
        live_stream=second_stream,
        closed_unit_streams={},
        closed_unit_events={},
        required=True,
    )

    released: list[str] = []
    packet_one = SimpleNamespace(
        generation_id=11, speaker_segment_id="seg-one", first_frame_id=3, last_frame_id=4
    )
    packet_two = SimpleNamespace(
        generation_id=12, speaker_segment_id="seg-two", first_frame_id=8, last_frame_id=8
    )

    async def release_and_ack(packet, branch, response_id, generation_id, segment_id):
        from duplex_kernel.server import _release_pcm_packet_then_ack

        async def release(_value):
            released.append(response_id)
            return True

        return await _release_pcm_packet_then_ack(
            packet,
            validate=lambda value: value is packet,
            release=release,
            acknowledge=lambda value: _ack_response_speech_packet(
                branch,
                value,
                response_id=response_id,
                generation_id=generation_id,
                response_segment_id=segment_id,
            ),
        )

    assert asyncio.run(release_and_ack(packet_one, first, "resp-one", 11, "seg-one"))
    assert asyncio.run(release_and_ack(packet_two, second, "resp-two", 12, "seg-two"))
    assert first_stream.acked == [(3, 11), (4, 11)]


    assert second_stream.acked == [(8, 12)]
    assert released == ["resp-one", "resp-two"]

    stale_packet = SimpleNamespace(
        generation_id=11, speaker_segment_id="seg-one", first_frame_id=5, last_frame_id=5
    )
    assert not _ack_response_speech_packet(
        first,
        stale_packet,
        response_id="resp-two",
        generation_id=11,
        response_segment_id="seg-one",
    )
    assert first_stream.acked == [(3, 11), (4, 11)]

    missing_owner = {
        "response_id": "resp-missing",
        "session_generation_id": 13,
        "speaker_segment_id": "seg-missing",
        "speech_ack_required": True,
    }
    assert not _ack_response_speech_packet(
        missing_owner,
        SimpleNamespace(
            generation_id=13, speaker_segment_id="seg-missing",
            first_frame_id=0, last_frame_id=0,
        ),
        response_id="resp-missing",
        generation_id=13,
        response_segment_id="seg-missing",
    )

    _retire_response_speech_ack_streams(first)
    assert not _ack_response_speech_packet(
        first,
        stale_packet,
        response_id="resp-one",
        generation_id=11,
        response_segment_id="seg-one",
    )
    assert first_stream.acked == [(3, 11), (4, 11)]


def test_finish_audio_segment_flushes_packets_then_enqueues_closed_marker() -> None:
    class Playout:
        def __init__(self) -> None:
            self.flushed: list[tuple[int, str]] = []

        def flush_segment(self, *, generation_id: int, speaker_segment_id: str):
            self.flushed.append((generation_id, speaker_segment_id))
            return ["frame"]

    class Packetizer:
        def __init__(self) -> None:
            self.flushed: list[tuple[int, str]] = []

        def flush_segment(self, generation_id: int, segment_id: str):
            self.flushed.append((generation_id, segment_id))
            return ["packetizer-tail"]

    response_id = "resp-closed-unit"
    generation_id = 23
    response_segment_id = "seg-response"
    branch = {
        "response_id": response_id,
        "session_generation_id": generation_id,
        "speaker_segment_id": response_segment_id,
    }
    events: dict[str, threading.Event] = {}
    assert _register_response_speech_ack_streams(
        branch,
        response_id=response_id,
        generation_id=generation_id,
        response_segment_id=response_segment_id,
        live_stream=None,
        closed_unit_streams={},
        closed_unit_events=events,
        required=True,
    )
    playout = Playout()
    packetizer = Packetizer()
    audio_queue = object()
    queued: list[tuple[object, str, int]] = []

    def packetize(frames):
        assert frames == ["frame"]
        return ["playout-packet"]

    def enqueue(item, *, queue_kind, output_queue, burst_items):
        assert output_queue is audio_queue
        queued.append((item, queue_kind, burst_items))
        return True

    finish_audio_segment = _make_finish_audio_segment_callback(
        active_output_segment_id={"value": "seg-tts"},
        playout=playout,
        packetizer=packetizer,
        ack_resources_branch=branch,
        response_id=response_id,
        generation_id=generation_id,
        response_segment_id=response_segment_id,
        packetize_frames_as_packets=packetize,
        queue_response_output=enqueue,
        audio_queue=audio_queue,
    )
    source_tokens = [{"token_index": 4, "text": "Hello."}]

    finish_audio_segment("seg-tts", source_tokens)

    assert playout.flushed == [(generation_id, "seg-tts")]
    assert packetizer.flushed == [(generation_id, "seg-tts")]
    assert events["seg-tts"].is_set() is False
    assert len(queued) == 3
    assert queued[0] == ("playout-packet", "audio", 3)
    assert queued[1] == ("packetizer-tail", "audio", 3)
    assert queued[2][0].speaker_segment_id == "seg-tts"
    assert queued[2][0].source_tokens == ({"token_index": 4, "text": "Hello."},)
    assert queued[2][1:] == ("audio", 3)


def test_private_budget_exhaustion_discards_backchannel_and_releases_slot() -> None:
    from duplex_kernel.session import DuplexSession

    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=True, controller_guided=True,
        terminal_reason="budget_exhausted", cancelled=False
    ) == "backchannel_without_eos"
    assert _backchannel_budget_without_eos(
        "SPEAK", private=True, controller_guided=True,
        terminal_reason="budget_exhausted", cancelled=False
    ) is None
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=True, controller_guided=True,
        terminal_reason="cancelled", cancelled=True
    ) == "cancelled"
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=False, controller_guided=True,
        terminal_reason="budget_exhausted", cancelled=False
    ) == "backchannel_without_eos"
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=False, controller_guided=True,
        terminal_reason="eos", cancelled=False
    ) is None

    published: list[tuple[int, int, str, object]] = []
    buffered = [(0, 7, "sì", object())]
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=True, controller_guided=True,
        terminal_reason="budget_exhausted", cancelled=False,
        buffered_tokens=buffered,
    ) == "backchannel_without_eos"
    assert not _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason="budget_exhausted",
        publish_batch=published.append,
    )
    assert published == []
    assert _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason="eos",
        publish_batch=published.append,
    )
    assert published == [tuple(buffered)]

    class Worker:
        def __init__(self) -> None:
            self.released: list[object] = []

        @staticmethod
        def _done(value=None):
            future = Future()
            future.set_result(value)
            return future

        def end_ephemeral_decode(self, handle):
            self.released.append(handle)
            return self._done()

        def rollback_assistant_branch(self, *_args, **_kwargs):
            raise AssertionError("private BACKCHANNEL must not modify canonical KV")

        def drain_queued_audio(self, *_args, **_kwargs):
            raise AssertionError("private BACKCHANNEL must not drain canonical audio")

        def session_state(self, *_args, **_kwargs):
            raise AssertionError("private BACKCHANNEL must not inspect canonical KV")

    session = DuplexSession("session")
    token = session.begin_response("backchannel")
    session.active_response = True
    handle = SimpleNamespace(cache_position=117)
    branch = {
        "ephemeral_private": True,
        "ephemeral_decode_handle": handle,
        "cache_position": 115,
        "finalized": False,
        "session_generation_id": token.session_generation_id,
        "spoken_prefix_lock": threading.Lock(),
    }
    worker = Worker()
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        session.session_id,
        branch,
        history_committed=False,
        close_ids=(),
    ))
    session.end_response(token)
    assert worker.released == [handle]
    assert branch["private_slot_released"] is True
    assert branch["finalized"] is True
    assert session.active_response is False

    next_token = session.begin_response("next-speak")
    session.active_response = True
    assert next_token.response_id == "next-speak"
    assert session.active_response is True


def test_unified_cancel_rolls_back_to_open_user_turn_and_drains_without_close_suffix() -> None:
    worker = _FutureWorker(cursor=70, turn_start=60, capacity=100, queued_tokens=4)
    branch = _unified_branch(cursor=70, turn_start=60)

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=False,
        close_ids=(61, 62, 63),
    ))

    assert worker.append_calls == []
    assert worker.rollback_targets == [60]
    assert worker.drain_open_modes == [True]
    assert worker.cursor == 64
    assert branch["finalized"] is True
    assert branch["history_committed"] is False


def test_unified_rejected_response_without_qwen_ack_keeps_empty_prefix_and_rolls_back() -> None:
    worker = _FutureWorker(cursor=70, turn_start=60, capacity=100, queued_tokens=2)
    branch = _unified_branch(cursor=70, turn_start=60)
    branch.update({
        "session_generation_id": 19,
        "spoken_prefix_lock": threading.Lock(),
        "spoken_prefix_kv_position": None,
        "spoken_prefix_text": "",
    })

    # V2 can reject before the first Qwen token is consumed. The frozen FIFO
    # snapshot is (None, ""): no cursor exists and no assistant prefix is spoken.
    _record_turn_local_vocalized_prefix(branch, 19, None, "")
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=False,
        close_ids=(61, 62),
    ))

    assert branch["spoken_prefix_kv_position"] is None
    assert branch["spoken_prefix_text"] == ""
    assert worker.append_calls == []
    assert worker.rollback_targets == [60]
    assert worker.drain_open_modes == [True]
    assert branch["finalized"] is True
    assert branch["history_committed"] is False


def test_unified_spoken_text_without_ack_cursor_is_rejected_explicitly() -> None:
    branch = {
        "session_generation_id": 19,
        "spoken_prefix_lock": threading.Lock(),
        "spoken_prefix_kv_position": None,
        "spoken_prefix_text": "",
    }

    with pytest.raises(RuntimeError, match="spoken text has no acknowledged KV cursor"):
        _record_turn_local_vocalized_prefix(branch, 19, None, "Aspetta!")


def test_unified_cancel_rebuilds_only_acknowledged_spoken_prefix_after_rewind() -> None:
    worker = _FutureWorker(cursor=75, turn_start=60, capacity=120, queued_tokens=2)
    branch = _unified_branch(cursor=75, turn_start=60)
    branch["transition_token_ids"] = (11, 12)
    branch["spoken_prefix_text"] = "Aspetta!"
    branch["spoken_prefix_token_ids"] = (21, 22, 23)
    branch["spoken_prefix_exclusive_cursor"] = 65

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=False,
        close_ids=(91, 92, 93),
    ))

    assert worker.rollback_targets == [60]
    assert worker.append_calls == [
        ((11, 12), 2),
        ((21, 22), 2),
        ((23, 91), 2),
        ((92, 93), 2),
    ]
    assert worker.drain_open_modes == [False]
    assert branch["spoken_prefix_rebuilt"] is True
    assert branch["history_committed"] is False


def test_client_consumed_pcm_uses_complete_segment_exact_tokens_and_exclusive_cursor() -> None:
    branch = {
        "response_id": "response-ack",
        "session_generation_id": 23,
        "speaker_segment_id": "segment-ack",
        "playback_segments": {
            "segment-ack": {
                "response_id": "response-ack",
                "generation_id": 23,
                "speaker_segment_id": "segment-ack",
                "start_sample": 0,
                "end_sample": 200,
                "sample_rate": 24000,
                "emitted_end_sample": 200,
                "done_emitted": True,
                "token_packets": (
                    {"token_id": 501, "exclusive_kv_cursor": 61, "text_piece": "Cia"},
                    {"token_id": 502, "exclusive_kv_cursor": 62, "text_piece": "o"},
                ),
            },
        },
        "turn_start_cache_position": 60,
        "spoken_prefix_token_ids": (),
        "spoken_prefix_exclusive_cursor": 60,
        "spoken_prefix_text": "",
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
    }

    partial = {
        "type": "client_consumed_pcm",
        "response_id": "response-ack",
        "generation_id": 23,
        "speaker_segment_id": "segment-ack",
        "start_sample": 0,
        "end_sample": 199,
        "sample_rate": 24000,
        "sequence": 1,
        "consumption_semantics": "logical_client_device_buffer",
    }
    assert _record_client_consumed_pcm(branch, partial) is False
    assert branch["spoken_prefix_token_ids"] == ()
    assert branch["spoken_prefix_exclusive_cursor"] == 60

    complete = {
        **partial,
        "end_sample": 200,
    }
    assert _record_client_consumed_pcm(branch, complete) is True
    assert branch["spoken_prefix_token_ids"] == (501, 502)
    assert branch["spoken_prefix_exclusive_cursor"] == 62
    assert branch["spoken_prefix_text"] == "Ciao"


def test_unified_cancel_rebuilds_exact_acknowledged_gemma_ids_not_retokenized_text() -> None:
    worker = _FutureWorker(cursor=75, turn_start=60, capacity=120, queued_tokens=2)
    branch = _unified_branch(cursor=75, turn_start=60)
    branch.update({
        "transition_token_ids": (11, 12),
        "spoken_prefix_token_ids": (501, 502),
        "spoken_prefix_exclusive_cursor": 64,
        # Text is deliberately unrelated: exact source IDs are authoritative.
        "spoken_prefix_text": "retokenization would be wrong",
        "spoken_prefix_rebuilt": False,
        "spoken_prefix_lock": threading.Lock(),
    })

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=False,
        close_ids=(91, 92, 93),
    ))

    assert worker.rollback_targets == [60]
    assert worker.append_calls == [
        ((11, 12), 2),
        ((501, 502), 2),
        ((91, 92), 2),
        ((93,), 1),
    ]
    assert branch["spoken_prefix_exclusive_cursor"] == 64
    assert branch["spoken_prefix_rebuilt"] is True


def test_unified_rollback_metrics_separate_physical_rewind_sampled_eos_and_acked_rebuild():
    worker = _FutureWorker(cursor=178, turn_start=159, capacity=200)
    branch = _unified_branch(cursor=178, turn_start=159)
    branch.update({
        "response_id": "response-rollback-metrics",
        "session_generation_id": 12,
        "speaker_segment_id": "segment-rollback-metrics",
        "transition_token_ids": (100, 101, 102, 103, 104, 105),
        "generated_token_count": 13,
        "sampled_generated_token_count": 14,
        "terminal_reason": "eos",
        "spoken_prefix_base_cursor": 165,
        "spoken_prefix_token_ids": (),
        "spoken_prefix_exclusive_cursor": 165,
        "spoken_prefix_text": "",
        "spoken_prefix_text_parts": (),
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
        "playback_segments": {
            "segment-rollback-metrics": {
                "response_id": "response-rollback-metrics",
                "generation_id": 12,
                "speaker_segment_id": "segment-rollback-metrics",
                "start_sample": 0,
                "end_sample": 200,
                "emitted_end_sample": 200,
                "sample_rate": 24000,
                "done_emitted": True,
                "token_packets": (
                    {"token_id": 501, "exclusive_kv_cursor": 166, "text_piece": "Cia"},
                    {"token_id": 502, "exclusive_kv_cursor": 167, "text_piece": "o"},
                    {"token_id": 503, "exclusive_kv_cursor": 168, "text_piece": "!"},
                ),
            },
        },
    })
    ack = {
        "type": "client_consumed_pcm",
        "response_id": "response-rollback-metrics",
        "generation_id": 12,
        "speaker_segment_id": "segment-rollback-metrics",
        "start_sample": 0,
        "end_sample": 200,
        "sample_rate": 24000,
        "sequence": 1,
        "consumption_semantics": "logical_client_device_buffer",
    }
    assert _record_client_consumed_pcm(branch, ack) is True
    assert branch["spoken_prefix_token_ids"] == (501, 502, 503)
    assert branch["spoken_prefix_exclusive_cursor"] == 168

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=False,
        close_ids=(301, 302),
    ))

    metrics = branch["rollback_metrics"]
    assert worker.rollback_targets == [159]
    assert metrics["physical_rewind_tokens"] == 19
    assert metrics["committed_generated_tokens"] == 13
    assert metrics["sampled_generated_tokens"] == 14
    assert metrics["unappended_terminal_tokens"] == 1
    assert metrics["transition_tokens_replayed"] == 6
    assert metrics["acknowledged_source_tokens_replayed"] == 3
    assert metrics["client_consumed_pcm_ack_sequence"] == 1
    assert metrics["acknowledged_source_exclusive_cursor"] == 168
    assert metrics["close_tokens_replayed"] == 2
    assert metrics["cache_position_before_rewind"] == 178
    assert metrics["cache_position_after_rebuild"] == 170
    assert metrics["net_cache_cursor_tokens_removed_after_rebuild"] == 8
    assert worker.cursor == 170


def test_client_consumed_pcm_rejects_stale_identity_and_replayed_sequence() -> None:
    branch = {
        "response_id": "response-current",
        "session_generation_id": 7,
        "speaker_segment_id": "segment-current",
        "playback_segments": {
            "segment-current": {
                "response_id": "response-current",
                "generation_id": 7,
                "speaker_segment_id": "segment-current",
                "start_sample": 0,
                "end_sample": 64,
                "sample_rate": 24000,
                "emitted_end_sample": 64,
                "done_emitted": True,
                "token_packets": (),
            },
        },
        "spoken_prefix_token_ids": (),
        "spoken_prefix_exclusive_cursor": 0,
        "spoken_prefix_text": "",
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
    }
    ack = {
        "type": "client_consumed_pcm",
        "response_id": "response-current",
        "generation_id": 7,
        "speaker_segment_id": "segment-current",
        "start_sample": 0,
        "end_sample": 64,
        "sample_rate": 24000,
        "sequence": 1,
        "consumption_semantics": "logical_client_device_buffer",
    }
    assert _record_client_consumed_pcm(branch, ack) is True
    assert _record_client_consumed_pcm(branch, ack) is False
    assert _record_client_consumed_pcm(branch, {**ack, "sequence": 2, "generation_id": 6}) is False
    assert _record_client_consumed_pcm(branch, {**ack, "sequence": 2, "response_id": "stale"}) is False


def test_client_consumed_pcm_ack_sequence_is_not_packet_id() -> None:
    branch = {
        "response_id": "response-counters",
        "session_generation_id": 1,
        "speaker_segment_id": "segment-counters",
        "playback_segments": {
            "segment-counters": {
                "response_id": "response-counters",
                "generation_id": 1,
                "speaker_segment_id": "segment-counters",
                "start_sample": 0,
                "end_sample": 16,
                "sample_rate": 24000,
                "emitted_end_sample": 16,
                "emitted_packet_id": 0,
                "done_emitted": True,
                "token_packets": (),
            },
        },
        "spoken_prefix_token_ids": (),
        "spoken_prefix_exclusive_cursor": 0,
        "spoken_prefix_text": "",
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
    }
    ack = {
        "type": "client_consumed_pcm",
        "response_id": "response-counters",
        "generation_id": 1,
        "speaker_segment_id": "segment-counters",
        "start_sample": 0,
        "end_sample": 16,
        "sample_rate": 24000,
        "sequence": 1,
        "consumption_semantics": "logical_client_device_buffer",
    }
    assert _record_client_consumed_pcm(branch, ack) is True


def test_segment_done_marker_uses_exact_emitted_range_and_client_schema() -> None:
    branch = {
        "response_id": "response-marker",
        "session_generation_id": 9,
        "playback_segments": {},
    }
    packet = SimpleNamespace(
        response_id="response-marker",
        generation_id=9,
        speaker_segment_id="segment-marker",
        start_sample=0,
        end_sample=16,
        packet_id=0,
        pcm=np.full(16, 0.25, dtype=np.float32),
    )
    _record_emitted_pcm_packet(branch, packet)
    marker = _complete_client_pcm_segment(
        branch,
        speaker_segment_id="segment-marker",
        token_packets=({
            "token_id": 70,
            "exclusive_kv_cursor": 81,
            "text_piece": "Ciao",
        },),
    )

    assert marker == {
        "type": "response.audio.segment_done",
        "response_id": "response-marker",
        "generation_id": 9,
        "speaker_segment_id": "segment-marker",
        "start_sample": 0,
        "end_sample": 16,
        "sample_rate": 24000,
    }
    segment = branch["playback_segments"]["segment-marker"]
    assert segment["done_emitted"] is True
    assert segment["token_packets"] == ({
        "token_id": 70,
        "exclusive_kv_cursor": 81,
        "text_piece": "Ciao",
    },)


def test_emitted_segment_requires_contiguous_packets_and_marker_once() -> None:
    branch = {
        "response_id": "response-marker",
        "session_generation_id": 9,
        "playback_segments": {},
    }
    first = SimpleNamespace(
        generation_id=9, speaker_segment_id="segment-marker",
        start_sample=0, end_sample=16, packet_id=0,
        pcm=np.full(16, 0.25, dtype=np.float32),
    )
    gap = SimpleNamespace(
        generation_id=9, speaker_segment_id="segment-marker",
        start_sample=17, end_sample=24, packet_id=1,
        pcm=np.full(7, 0.25, dtype=np.float32),
    )
    _record_emitted_pcm_packet(branch, first)
    with pytest.raises(RuntimeError, match="non-contiguous"):
        _record_emitted_pcm_packet(branch, gap)
    marker = _complete_client_pcm_segment(branch, speaker_segment_id="segment-marker")
    assert marker["end_sample"] == 16
    assert _complete_client_pcm_segment(branch, speaker_segment_id="segment-marker") is None


def test_closed_tts_segment_requires_actual_codec_eos_not_text_drain() -> None:
    _require_closed_tts_codec_eos({"eos_detected": True, "terminated_by": "eos"})
    with pytest.raises(RuntimeError, match="did not reach codec EOS"):
        _require_closed_tts_codec_eos({
            "eos_detected": False,
            "terminated_by": "budget",
            "text_eos_consumed": True,
            "fifo_drained": True,
        })
    with pytest.raises(RuntimeError, match="did not reach codec EOS"):
        _require_closed_tts_codec_eos({"text_eos_consumed": True, "fifo_drained": True})


def test_unified_finalizer_retries_one_pending_q0_watermark_without_rolling_back() -> None:
    worker = _FutureWorker(
        cursor=70,
        turn_start=60,
        capacity=100,
        queued_tokens=2,
        reserve_watermark_failures=1,
    )
    branch = _unified_branch(cursor=70, turn_start=60)

    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=True,
        close_ids=(71, 72),
    ))

    assert worker.rollback_targets == []
    assert worker.append_calls == [((71, 72), 2)]
    assert worker.drain_open_modes == [False]
    assert worker.cursor == 74
    assert branch["history_committed"] is True


def test_shared_transition_lock_wait_is_per_row_and_uses_injected_clock() -> None:
    class Lock:
        acquired = False

        def acquire(self):
            self.acquired = True

    stamps = iter((1_000_000, 6_250_000))
    lock = Lock()
    tick = DuplexTick(
        frame_id=4,
        capture_start_ns=1,
        capture_end_ns=2,
        enqueue_ns=3,
    )
    tick.listener_batch_id = "batch-4"

    waited_ms = _acquire_shared_transition_lock_with_timing(
        lock, tick, now_ns=lambda: next(stamps),
    )

    assert lock.acquired is True
    assert waited_ms == 5.25
    assert tick.microbatch_unified_transition_lock_wait_ms == 5.25
    event = scheduler_tick_trace_metadata(tick)
    assert event["scheduler_unified_turn_transition_lock_wait_ms"] == 5.25
    assert event["listener_batch_unified_turn_transition_lock_wait_ms"] == 5.25


def test_runtime_metrics_filter_preserves_qwen_clocks_and_finalizer_denominators() -> None:
    fields = _filter_unified_native_event_metrics({
        "first_qwen_id_published_at_ns": 100,
        "qwen_dynamic_prefill_model_entry_at_ns": 145,
        "qwen_first_id_to_prefill_entry_ms": 0.000045,
        "qwen_prefill_clock": "server monotonic_ns",
        "physical_rewind_tokens": 19,
        "sampled_generated_tokens": 14,
        "net_cache_cursor_tokens_removed_after_rebuild": 8,
        "acknowledged_source_token_ids_replayed": [411, 412, 413],
        "queued_audio_drain_tokens": 62,
        "queued_audio_source_frame_ids": list(range(42, 73)),
        "ignored_object": object(),
    })

    assert fields == {
        "first_qwen_id_published_at_ns": 100,
        "qwen_dynamic_prefill_model_entry_at_ns": 145,
        "qwen_first_id_to_prefill_entry_ms": 0.000045,
        "qwen_prefill_clock": "server monotonic_ns",
        "physical_rewind_tokens": 19,
        "sampled_generated_tokens": 14,
        "net_cache_cursor_tokens_removed_after_rebuild": 8,
        "acknowledged_source_token_ids_replayed": [411, 412, 413],
        "queued_audio_drain_tokens": 62,
        "queued_audio_source_frame_ids": list(range(42, 73)),
    }


def test_terminal_metrics_projection_omits_full_response_trace() -> None:
    projected = _compact_unified_native_terminal_metrics({
        "generation_trace": {"thinker_token_ids": list(range(100_000))},
        "scheduler": {"queued_ticks": list(range(100_000))},
        "qwen_prefill": {
            "qwen_dynamic_prefill_model_entry_at_ns": 145,
            "qwen_prefill_clock": "server monotonic_ns",
        },
        "unified_native_rollback": {"physical_rewind_tokens": 19, "raw_trace": "x" * 10_000},
        "unified_native_finalizer": {"status": "completed", "queued_audio_drain_tokens": 62},
        "thinker": {"thinker_token_ids": list(range(100_000))},
    })

    assert projected == {
        "qwen_prefill": {
            "qwen_dynamic_prefill_model_entry_at_ns": 145,
            "qwen_prefill_clock": "server monotonic_ns",
        },
        "unified_native_rollback": {"physical_rewind_tokens": 19},
        "unified_native_finalizer": {
            "queued_audio_drain_tokens": 62,
            "status": "completed",
        },
    }


def test_unified_native_finalizer_result_is_a_separate_metrics_event() -> None:
    event = _unified_native_finalizer_event(
        "response-yield",
        status="completed",
        rollback_metrics={"physical_rewind_tokens": 19},
        drain_metrics={
            "queued_audio_drain_ms": 324.5,
            "queued_audio_drain_frames": 31,
            "queued_audio_drain_tokens": 62,
            "queued_audio_retained_context_source": "assistant_phase_q2_queue",
            "queued_audio_source_frame_ids": list(range(42, 73)),
        },
    )

    assert event["type"] == "response.unified_native_finalized"
    assert event["response_id"] == "response-yield"
    assert event["status"] == "completed"
    assert event["runtime_metrics"]["unified_native_rollback"]["physical_rewind_tokens"] == 19
    assert event["runtime_metrics"]["unified_native_finalizer"]["queued_audio_drain_tokens"] == 62
    assert event["runtime_metrics"]["unified_native_finalizer"]["queued_audio_source_frame_ids"] == list(range(42, 73))


def test_unified_native_finalizer_send_keeps_metrics_after_response_cancelled() -> None:
    sent: list[dict] = []
    event_metrics = {
        "unified_native_rollback": _filter_unified_native_event_metrics({
            "physical_rewind_tokens": 19,
            "sampled_generated_tokens": 14,
            "net_cache_cursor_tokens_removed_after_rebuild": 8,
        }),
        "unified_native_finalizer": _filter_unified_native_event_metrics({
            "queued_audio_drain_ms": 320.0,
            "queued_audio_drain_frames": 31,
            "queued_audio_drain_tokens": 62,
            "queued_audio_retained_context_source": "assistant_phase_q2_queue",
            "queued_audio_source_frame_ids": list(range(42, 73)),
            "status": "completed",
        }),
        "qwen_prefill": _filter_unified_native_event_metrics({
            "first_qwen_id_published_at_ns": 100,
            "qwen_dynamic_prefill_model_entry_at_ns": 145,
            "qwen_prefill_clock": "server monotonic_ns",
        }),
    }

    async def send(payload):
        sent.append(payload)
        return True

    accepted = asyncio.run(_send_unified_native_finalizer_event(
        send,
        "response-yield",
        status="completed",
        runtime_metrics=event_metrics,
    ))

    assert accepted is True
    assert sent == [{
        "type": "response.unified_native_finalized",
        "response_id": "response-yield",
        "status": "completed",
        "runtime_metrics": event_metrics,
    }]
    assert sent[0]["runtime_metrics"]["qwen_prefill"]["qwen_dynamic_prefill_model_entry_at_ns"] == 145
    assert sent[0]["runtime_metrics"]["unified_native_rollback"]["net_cache_cursor_tokens_removed_after_rebuild"] == 8


def test_unified_native_finalizer_prunes_oversized_metrics_in_one_event_pass(monkeypatch) -> None:
    sent: list[dict] = []
    event_metrics = {
        "qwen_prefill": {"qwen_dynamic_prefill_model_entry_at_ns": 145},
        "unified_native_rollback": {"physical_rewind_tokens": 19},
        "unified_native_finalizer": {"status": "completed", "queued_audio_drain_tokens": 62},
        **{
            f"phase-{index:03d}": {"physical_rewind_tokens": 10**250}
            for index in range(100)
        },
    }
    original_dumps = json.dumps
    event_serializations = 0

    def count_event_serializations(value, *args, **kwargs):
        nonlocal event_serializations
        if isinstance(value, dict) and value.get("type") == "response.unified_native_finalized":
            event_serializations += 1
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(server_metrics_module.json, "dumps", count_event_serializations)

    async def send(payload):
        sent.append(payload)
        return True

    assert asyncio.run(_send_unified_native_finalizer_event(
        send,
        "response-yield",
        status="completed",
        runtime_metrics=event_metrics,
    )) is True

    event = sent[0]
    compact_json = original_dumps(event, separators=(",", ":")).encode("utf-8")
    assert event_serializations == 1
    assert len(compact_json) <= 2_048
    assert event["runtime_metrics"]["unified_native_finalizer"]["status"] == "completed"
    assert event["runtime_metrics"]["qwen_prefill"]["qwen_dynamic_prefill_model_entry_at_ns"] == 145
    assert event["runtime_metrics"]["unified_native_rollback"]["physical_rewind_tokens"] == 19


def test_unified_finalizer_rolls_back_untracked_cache_advance_before_close() -> None:
    worker = _FutureWorker(cursor=71, turn_start=60, capacity=100, queued_tokens=2)
    branch = _unified_branch(cursor=70, turn_start=60)

    with pytest.raises(RuntimeError, match="assistant branch was rolled back"):
        asyncio.run(_finalize_unified_native_cache_branch(
            worker,
            "session",
            branch,
            history_committed=True,
            close_ids=(81, 82),
        ))

    assert worker.append_calls == []
    assert worker.rollback_targets == [60]
    assert worker.drain_open_modes == [True]
    assert worker.cursor == 62
