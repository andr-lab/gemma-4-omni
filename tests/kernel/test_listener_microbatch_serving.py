from __future__ import annotations

import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.listener_microbatch_serving import (
    GEMM_GRAPHS_ENV,
    SEMANTIC_SHADOW_ENV,
    STABLE_TURN_POLICY_Q2_CONTEXT_ENV,
    _build_native_control6_batch_policy,
    _semantic_prefix_request_matches_applied_row,
    _strict_env_flag,
    commit_initial_semantic_prefix_state,
    initial_listener_session_state,
    make_listener_microbatch_worker,
    make_unified_prefix_import,
    prepare_initial_semantic_prefix,
    prepare_initial_semantic_prefix_async,
)
from duplex_kernel.gemma_native_control6_batch_policy import PolicyPreHeadContext
from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.microbatch_contract import ListenerFrame, RuntimeSignature
from duplex_kernel.listener_microbatch_worker import PrefixCacheImport, PreparedInitialPrefix
from duplex_kernel.native_control6_runtime import NativeControl6SemanticPrefixRequest
from duplex_kernel.turn_policy import PolicyInput
from training.duplex_qlora.native_control6 import ACTIONS, HIDDEN_WIDTH, NativeControl6LoRA


@pytest.mark.parametrize(
    ("request_generation", "request_session", "request_epoch", "request_revision", "expected"),
    [
        (7, "s", "e", 4, True),
        (6, "s", "e", 4, False),
        (7, "other", "e", 4, False),
        (7, "s", "old-epoch", 4, False),
        (7, "s", "e", 3, False),
        (None, "s", "e", 4, False),
    ],
)
def test_applied_prefix_request_reuse_requires_full_identity(
    request_generation, request_session, request_epoch, request_revision, expected
) -> None:
    request = SimpleNamespace(
        session_id=request_session,
        state_epoch_id=request_epoch,
        semantic_context_revision=request_revision,
    )
    row = SimpleNamespace(
        session_id="s",
        state_epoch_id="e",
        semantic_context_revision=4,
        generation_id=7,
    )
    assert _semantic_prefix_request_matches_applied_row(
        request, request_generation, row
    ) is expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        (GEMM_GRAPHS_ENV, "0", False),
        (GEMM_GRAPHS_ENV, "1", True),
        (SEMANTIC_SHADOW_ENV, "0", False),
        (SEMANTIC_SHADOW_ENV, "1", True),
        (STABLE_TURN_POLICY_Q2_CONTEXT_ENV, "0", False),
        (STABLE_TURN_POLICY_Q2_CONTEXT_ENV, "1", True),
    ],
)
def test_microbatch_serving_switches_accept_only_explicit_zero_or_one(
    monkeypatch, name: str, value: str, expected: bool
) -> None:
    monkeypatch.setenv(name, value)
    assert _strict_env_flag(name) is expected


@pytest.mark.parametrize("value", ["true", "yes", "", " 1", "1 "])
def test_microbatch_serving_switches_reject_ambiguous_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv(GEMM_GRAPHS_ENV, value)
    with pytest.raises(ValueError, match="exactly '0' or '1'"):
        _strict_env_flag(GEMM_GRAPHS_ENV)


def test_microbatch_serving_switches_default_off(monkeypatch) -> None:
    monkeypatch.delenv(GEMM_GRAPHS_ENV, raising=False)
    monkeypatch.delenv(SEMANTIC_SHADOW_ENV, raising=False)
    monkeypatch.delenv(STABLE_TURN_POLICY_Q2_CONTEXT_ENV, raising=False)
    assert _strict_env_flag(GEMM_GRAPHS_ENV) is False
    assert _strict_env_flag(SEMANTIC_SHADOW_ENV) is False
    assert _strict_env_flag(STABLE_TURN_POLICY_Q2_CONTEXT_ENV) is False


def test_native_control6_serving_factory_builds_mixed_policy_and_reads_frame_context() -> None:
    adapter = NativeControl6LoRA(torch.zeros((len(ACTIONS), HIDDEN_WIDTH)))
    session_policy = GemmaNativeControl6Policy(adapter, checkpoint_path="cpu-test.pt")
    batch_policy, policy_inputs = _build_native_control6_batch_policy(session_policy)
    assert batch_policy is not None and batch_policy.adapter is adapter
    assert callable(policy_inputs)

    pcm = np.zeros(1280, dtype=np.float32)
    snapshot = PolicyInput(
        session_id="s",
        frame_id=5,
        user_pcm=pcm,
        speaker_active=False,
        generation_id=22,
        legal_actions=("LISTEN", "SPEAK"),
        native_control_phase_t=(0.0, 0.0, 0.0, 0.0),
        native_listener_generation=7,
    )
    context = PolicyPreHeadContext(
        policy=session_policy,
        policy_input=snapshot,
        tick_id=("s", 22, 5),
        captured_at_ns=1,
        listener_generation_id=7,
    )
    frame = ListenerFrame(
        session_id="s",
        generation_id=7,
        state_epoch_id="epoch-s",
        frame_id=5,
        ready_at_ns=100,
        pcm=pcm,
        policy_context=context,
    )
    row = SimpleNamespace(
        session_id="s",
        session_generation=7,
        state_epoch_id="epoch-s",
        frame_id=5,
    )
    listener_state = SimpleNamespace(
        session_id="s", generation_id=7, state_epoch_id="epoch-s"
    )
    batch = SimpleNamespace(frames=(frame,), manifest=SimpleNamespace(rows=(row,)))

    assert policy_inputs(batch, (listener_state,)) == (context,)
    with pytest.raises(RuntimeError, match="listener generation differs"):
        altered_row = SimpleNamespace(
            session_id="s",
            session_generation=8,
            state_epoch_id="epoch-s",
            frame_id=5,
        )
        policy_inputs(
            SimpleNamespace(frames=(frame,), manifest=SimpleNamespace(rows=(altered_row,))),
            (listener_state,),
        )


def _signature() -> RuntimeSignature:
    return RuntimeSignature(
        model_id="test-gemma",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256="1" * 64,
        native_library_sha256="0" * 64,
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
    )


def test_initial_worker_state_maps_empty_listener_epoch_without_aliasing_mutable_histories() -> None:
    listener_state = GemmaListenerState()
    listener_state.user_pcm_tail = np.zeros(160, dtype=np.float32)
    session = SimpleNamespace(session_id="s", listener_state=listener_state)

    batch_state = initial_listener_session_state(
        session,
        _signature(),
        native_context_id=31,
        generation_id=7,
    )

    assert batch_state.session_id == "s"
    assert batch_state.generation_id == 7
    assert batch_state.state_epoch_id == listener_state.state_epoch_id
    assert batch_state.next_frame_id == 0
    assert batch_state.state_version == 0
    assert batch_state.sequence_position == 0
    assert batch_state.cache_position == 34
    assert batch_state.native_context_id == 31
    assert batch_state.user_pcm_tail is not listener_state.user_pcm_tail
    batch_state.user_mel_history.append(np.zeros((8, 128), dtype=np.float32))
    assert not listener_state.user_mel_history


def test_initial_worker_state_rejects_replay_without_a_verified_prefix_builder() -> None:
    listener_state = GemmaListenerState(next_frame_id=1, listener_input_frames=1)
    session = SimpleNamespace(session_id="s", listener_state=listener_state)

    with pytest.raises(RuntimeError, match="empty listener epoch"):
        initial_listener_session_state(
            session,
            _signature(),
            native_context_id=32,
        )


def test_unified_prefix_import_builds_one_dynamic_text_only_cache_for_frame_zero() -> None:
    calls: list[tuple[tuple[int, ...], tuple, tuple]] = []

    class Cache:
        def get_seq_length(self, _layer: int = 0) -> int:
            return 4

    def build(prefix_ids, *, replay_q2, replay_frame_ids):
        calls.append((tuple(prefix_ids), tuple(replay_q2), tuple(replay_frame_ids)))
        return {
            "past_key_values": Cache(),
            "prefix_attention": torch.ones((1, 4), dtype=torch.int64),
            "prefix_len": 4,
            "cache_position": 4,
            "replay_frame_ids": (),
            "prefix_build_ms": 1.25,
            "replay_ms": 0.0,
        }

    backend = SimpleNamespace(
        bos_token_id=2,
        boa_token_id=256000,
        audio_token_id=258881,
        eoa_token_id=258883,
        _lock=object(),
        build_semantic_prefix_cache=build,
    )
    tokens = (2, 101, 102, 256000)
    importer = make_unified_prefix_import(backend, tokens)
    state = SimpleNamespace(
        session_id="s",
        generation_id=7,
        state_epoch_id="epoch-s",
        next_frame_id=0,
        sequence_position=0,
    )
    row = SimpleNamespace(
        session_id="s",
        session_generation=7,
        state_epoch_id="epoch-s",
        frame_id=0,
        sequence_position=0,
        cache_position=len(tokens),
        valid_history_tokens=len(tokens),
    )

    first = importer(state, row)
    second = importer(state, row)

    assert getattr(importer, "requires_shared_lm_lock") is True
    assert first is second
    assert first.prefix_len == first.cache_position == len(tokens)
    assert first.past_key_values.get_seq_length(0) == len(tokens)
    assert calls == [(tokens, (), ())]


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        ((2, 101, 258881, 256000), "audio placeholders"),
        ((2, 101, 258883, 256000), "audio placeholders"),
        ((2, 256000, 256000), "exactly one final BOA"),
        ((99, 101, 256000), "BOS token"),
    ],
)
def test_unified_prefix_import_rejects_non_chat_or_audio_prefixes(tokens, message) -> None:
    backend = SimpleNamespace(
        bos_token_id=2,
        boa_token_id=256000,
        audio_token_id=258881,
        eoa_token_id=258883,
        _lock=object(),
        build_semantic_prefix_cache=lambda *_args, **_kwargs: pytest.fail(
            "invalid prefix must fail before model execution"
        ),
    )
    with pytest.raises(ValueError, match=message):
        make_unified_prefix_import(backend, tokens)


def test_initial_semantic_prefix_preparation_does_not_block_the_server_event_loop(monkeypatch) -> None:
    import duplex_kernel.listener_microbatch_serving as serving

    started = threading.Event()
    release = threading.Event()

    def blocked_prepare(*_args, **_kwargs):
        started.set()
        assert release.wait(3)
        return "prepared"

    monkeypatch.setattr(serving, "prepare_initial_semantic_prefix", blocked_prepare)

    async def exercise() -> str:
        preparation = asyncio.create_task(prepare_initial_semantic_prefix_async())
        assert await asyncio.to_thread(started.wait, 2)
        # This coroutine can run while the prefill is held on its background
        # executor thread, so the WebSocket loop remains able to serve peers.
        await asyncio.sleep(0)
        release.set()
        return await preparation

    assert asyncio.run(exercise()) == "prepared"


def test_initial_prefix_stream_finishes_before_shared_model_lock_is_released(monkeypatch) -> None:
    import duplex_kernel.listener_microbatch_serving as serving

    order: list[str] = []

    class ModelLock:
        held = False

        def acquire_fair(self, priority: str) -> bool:
            assert priority == "listener"
            assert not self.held
            self.held = True
            order.append("acquire")
            return True

        def release(self) -> None:
            assert self.held
            order.append("release")
            self.held = False

    model_lock = ModelLock()

    class FakeStream:
        def __init__(self, **_kwargs) -> None:
            pass

        def synchronize(self) -> None:
            assert model_lock.held
            order.append("stream_complete")

    class FakeEvent:
        def __init__(self, **_kwargs) -> None:
            pass

        def record(self, stream) -> None:
            assert isinstance(stream, FakeStream)
            order.append("event_record")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(serving.torch.cuda, "stream", lambda _stream: nullcontext())

    request = NativeControl6SemanticPrefixRequest(
        session_id="s",
        session_generation_id=3,
        state_epoch_id="epoch",
        listener_state_version=0,
        current_frame_id=0,
        semantic_context_revision=5,
        prefix_token_ids=tuple(range(141)),
        prefix_len=141,
        replay_q2=(),
        replay_frame_ids=(),
        fallback_reason=None,
    )

    def build_cache(prefix_ids, *, replay_q2, replay_frame_ids):
        assert model_lock.held
        assert tuple(prefix_ids) == request.prefix_token_ids
        assert not replay_q2 and not replay_frame_ids
        order.append("build")
        return {
            "prefix_len": 141,
            "cache_position": 141,
            "replay_frame_ids": (),
            "past_key_values": object(),
            "prefix_attention": torch.ones((1, 141), dtype=torch.long),
            "prefix_build_ms": 10.0,
            "replay_ms": 0.0,
        }

    backend = SimpleNamespace(device="cuda:0", build_semantic_prefix_cache=build_cache)
    prepared = prepare_initial_semantic_prefix(
        backend,
        model_lock,
        request,
        listener_generation_id=1,
        expected_cache_position=141,
    )

    assert prepared.ready_event is not None
    assert order == ["acquire", "build", "event_record", "stream_complete", "release"]
    assert not model_lock.held


def _initial_prefix_snapshot(session_id: str, state: GemmaListenerState, revision: int = 5):
    request = NativeControl6SemanticPrefixRequest(
        session_id=session_id,
        session_generation_id=3,
        state_epoch_id=state.state_epoch_id,
        listener_state_version=0,
        current_frame_id=0,
        semantic_context_revision=revision,
        prefix_token_ids=tuple(range(141)),
        prefix_len=141,
        replay_q2=(),
        replay_frame_ids=(),
        fallback_reason=None,
    )
    prefix = PrefixCacheImport(
        past_key_values=object(),
        prefix_attention=torch.ones((1, 141), dtype=torch.long),
        prefix_len=141,
        cache_position=141,
        semantic_context_revision=revision,
    )
    prepared = PreparedInitialPrefix(
        prefix=prefix,
        ready_event=object(),
        session_id=session_id,
        listener_generation_id=1,
        state_epoch_id=state.state_epoch_id,
        semantic_context_revision=revision,
        expected_cache_position=141,
    )
    return request, prepared


def test_initial_prefix_commit_applies_only_current_revision_and_pristine_cursor() -> None:
    state = GemmaListenerState()
    session = SimpleNamespace(
        session_id="s",
        semantic_context_revision=5,
        _generation_lock=threading.RLock(),
        listener_state=state,
    )
    request, prepared = _initial_prefix_snapshot("s", state)

    assert commit_initial_semantic_prefix_state(session, request, prepared)
    assert state.semantic_context_revision == 5
    assert state.gemma_lm_prefix_len == 141
    assert torch.equal(state.gemma_lm_prefix_attention, prepared.prefix.prefix_attention)
    assert state.next_frame_id == state.listener_state_version == state.listener_input_frames == 0

    stale_state = GemmaListenerState()
    stale_session = SimpleNamespace(
        session_id="stale",
        semantic_context_revision=6,
        _generation_lock=threading.RLock(),
        listener_state=stale_state,
    )
    stale_request, stale_prepared = _initial_prefix_snapshot("stale", stale_state)
    stale_state.gemma_lm_prefix_len = 34
    assert not commit_initial_semantic_prefix_state(stale_session, stale_request, stale_prepared)
    assert stale_state.semantic_context_revision is None
    assert stale_state.gemma_lm_prefix_len == 34
    assert stale_state.gemma_lm_prefix_attention is None

    stale_cursor_state = GemmaListenerState(
        next_frame_id=1,
        listener_state_version=1,
        listener_input_frames=1,
    )
    stale_cursor_session = SimpleNamespace(
        session_id="stale-cursor",
        semantic_context_revision=5,
        _generation_lock=threading.RLock(),
        listener_state=stale_cursor_state,
    )
    stale_cursor_request, stale_cursor_prepared = _initial_prefix_snapshot(
        "stale-cursor", stale_cursor_state
    )
    assert not commit_initial_semantic_prefix_state(
        stale_cursor_session, stale_cursor_request, stale_cursor_prepared
    )
    assert stale_cursor_state.semantic_context_revision is None
    assert stale_cursor_state.gemma_lm_prefix_len == 34
    assert stale_cursor_state.gemma_lm_prefix_attention is None


def test_initial_prefix_commit_uses_generation_then_listener_state_lock() -> None:
    state = GemmaListenerState()
    order: list[str] = []

    class ObservedLock:
        def __init__(self, name: str) -> None:
            self.name = name
            self.lock = threading.RLock()

        def __enter__(self):
            if self.name == "state":
                assert generation_lock.lock._is_owned()
            self.lock.acquire()
            order.append(self.name + ":acquire")
            return self

        def __exit__(self, *_args):
            order.append(self.name + ":release")
            self.lock.release()

    generation_lock = ObservedLock("generation")
    state._lock = ObservedLock("state")
    session = SimpleNamespace(
        session_id="s",
        semantic_context_revision=5,
        _generation_lock=generation_lock,
        listener_state=state,
    )
    request, prepared = _initial_prefix_snapshot("s", state)

    assert commit_initial_semantic_prefix_state(session, request, prepared)
    assert order == [
        "generation:acquire", "state:acquire", "state:release", "generation:release"
    ]


def _row_output(state: GemmaListenerState, *, frame_id: int = 0, version: int = 1):
    health = SimpleNamespace(status="healthy", checked=True)
    return SimpleNamespace(
        generation_id=7,
        frame_id=frame_id,
        state_epoch_id=state.state_epoch_id,
        listener_state_version=version,
        sequence_position=1,
        cache_position=36,
        valid_history_tokens=36,
        logical_q_len=2,
        ephemeral_reason=None,
        queue_audio_for_next_turn=False,
        kv_rollback_tokens=0,
        gemma_hidden_t=torch.ones((1, 1536), dtype=torch.bfloat16),
        gemma_audio_queries_t=torch.ones((2, 1536), dtype=torch.bfloat16),
        audio_tower_hidden=torch.ones((2, 1536), dtype=torch.bfloat16),
        state_snapshot=SimpleNamespace(
            user_mel_history=(np.ones((8, 128), dtype=np.float32),),
            assistant_mel_history=(np.zeros((8, 128), dtype=np.float32),),
            user_pcm_tail=np.ones(160, dtype=np.float32),
            assistant_pcm_tail=np.zeros(160, dtype=np.float32),
        ),
        native_state=SimpleNamespace(current_position=36, prefix_len=34),
        native_output_health=health,
        semantic_context_revision=0,
        semantic_replay_fallback_reason=None,
        semantic_prefix_len=44,
        listener_model_lock_wait_ms=1.25,
        listener_model_lock_hold_ms=4.5,
    )


def test_batch_row_commit_matches_listener_input_cursors_and_current_output_contract() -> None:
    listener_state = GemmaListenerState()
    frame_done = threading.Event()
    tick_started = threading.Event()
    backend = SimpleNamespace(
        hidden_size=1536,
        _frame_done_event=frame_done,
        _tick_started_event=tick_started,
        _frame_counter=0,
    )
    listener = GemmaListener(backend=backend)
    user_pcm = np.full(1280, 0.25, dtype=np.float32)
    row_output = _row_output(listener_state)
    timing: dict[str, int] = {}

    output = listener.commit_microbatch_output(
        user_pcm,
        None,
        listener_state,
        row_output,
        source_available_at_ns=123_456,
        timing_sink=timing,
    )

    assert output.frame_id == 0
    assert output.source_frame_id == 0
    assert output.source_available_at_ns == 123_456
    assert output.native_output_health is row_output.native_output_health
    assert output.listener_generation_id == row_output.generation_id
    assert output.listener_state_epoch_id == listener_state.state_epoch_id
    assert output.listener_sequence_position == row_output.sequence_position
    assert output.listener_cache_position == row_output.cache_position
    assert output.listener_state_version == 1
    assert output.fresh and not output.silence_skipped
    assert listener_state.next_frame_id == 1
    assert listener_state.listener_input_frames == 1
    assert listener_state.total_samples == 1280
    assert listener_state.last_hidden_source_frame_id == 0
    assert listener_state.gemma_lm_audio_frames == 1
    assert listener_state.semantic_context_revision == 0
    assert listener_state.gemma_lm_prefix_len == 44
    assert len(listener_state.user_pcm_history) == len(listener_state.assistant_pcm_history) == 1
    np.testing.assert_array_equal(listener_state.user_pcm_history[0], user_pcm)
    np.testing.assert_array_equal(listener_state.assistant_pcm_history[0], np.zeros(1280, dtype=np.float32))
    assert len(listener_state.user_mel_history) == len(listener_state.assistant_mel_history) == 1
    assert listener_state.last_lock_wait_ms == 1.25
    assert listener_state.last_lock_hold_ms == 4.5
    assert tick_started.is_set() and frame_done.is_set()
    assert backend._frame_counter == 1
    assert timing["state_lock_acquired_ns"] <= timing["state_lock_released_ns"]
    assert timing["state_lock_wait_ns"] >= 0
    assert timing["state_lock_hold_ns"] >= 0


def test_batch_row_commit_rejects_stale_identity_before_mutating_listener_history() -> None:
    listener_state = GemmaListenerState()
    listener = GemmaListener(backend=SimpleNamespace(hidden_size=1536))

    with pytest.raises(RuntimeError, match="frame does not match"):
        listener.commit_microbatch_output(
            np.zeros(1280, dtype=np.float32),
            None,
            listener_state,
            _row_output(listener_state, frame_id=1),
            source_available_at_ns=123,
        )

    assert listener_state.next_frame_id == 0
    assert listener_state.listener_state_version == 0
    assert not listener_state.user_pcm_history
