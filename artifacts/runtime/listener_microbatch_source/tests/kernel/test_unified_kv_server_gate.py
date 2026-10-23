from __future__ import annotations

import asyncio
import numpy as np
import pytest
import torch
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

from duplex_kernel.server import (
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
)


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

        def update_instructions(self, value):
            value = str(value)
            with self._generation_lock:
                if value == self.instructions:
                    return False
                self.instructions = value
                self.semantic_context_revision += 1
                return True

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

    def make_import(_backend, token_ids):
        importer = SimpleNamespace(token_ids=tuple(token_ids))
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
    assert len(call[1]) == binding["batch_cache_position"] == 227
    assert binding["semantic_prefix_importer"] is imports[-1]

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


def test_backchannel_generation_budget_caps_real_tokenizer_ids_at_two() -> None:
    assert _controller_guided_say_generation_options(
        action="BACKCHANNEL",
        controller_guided_say=False,
        direct_audio_candidate=False,
        response_protocol=None,
    )["max_tokens"] == 2
    assert _controller_guided_say_generation_options(
        action="BACKCHANNEL",
        controller_guided_say=True,
        direct_audio_candidate=True,
        response_protocol="heard_action_say_v2",
    )["max_tokens"] == 2


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
        lambda _backend, token_ids: SimpleNamespace(token_ids=tuple(token_ids)),
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


def test_unified_frame_logical_q_skips_only_idle_prespeech_and_marks_assistant_queue() -> None:
    quiet = np.zeros(1280, dtype=np.float32)
    assert _unified_listener_frame_contract(
        quiet,
        assistant_audio_queue_open=False,
        user_input_pending=False,
    ) == (0, "prespeech_silence", False)

    voiced = np.full(1280, 0.02, dtype=np.float32)
    assert _unified_listener_frame_contract(
        voiced,
        assistant_audio_queue_open=True,
        user_input_pending=False,
    ) == (0, "assistant_turn_open", True)


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
    assert assistant_frame == (0, "assistant_turn_open", True)

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

    # A shared response owns canonical KV as soon as its queue is open, including prefill.
    assert _shared_listener_frame_queue_contract(
        binding,
        voiced,
        response_id="shared",
        user_input_pending=True,
    ) == (0, "assistant_turn_open", True)
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

    class Worker:
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

    class Worker:
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
    class Worker:
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
        return self._done(SimpleNamespace(cache_position=self.cursor))


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


def test_private_backchannel_budget_without_eos_is_discarded_and_slot_released() -> None:
    from duplex_kernel.session import DuplexSession

    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=True, controller_guided=True,
        terminal_reason="budget_exhausted", cancelled=False
    ) == "backchannel_budget_without_eos"
    assert _backchannel_budget_without_eos(
        "SPEAK", private=True, controller_guided=True,
        terminal_reason="budget_exhausted", cancelled=False
    ) is None
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL", private=True, controller_guided=True,
        terminal_reason="cancelled", cancelled=True
    ) is None
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
    assert not _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason="budget_exhausted",
        publish=lambda *row: published.append(row),
    )
    assert published == []
    assert _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason="eos",
        publish=lambda *row: published.append(row),
    )
    assert published == [buffered[0]]

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
    )
    gap = SimpleNamespace(
        generation_id=9, speaker_segment_id="segment-marker",
        start_sample=17, end_sample=24, packet_id=1,
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
