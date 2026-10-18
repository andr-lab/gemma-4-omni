from __future__ import annotations

import pytest
import torch
import threading
from types import SimpleNamespace

from duplex_kernel.native_control6_runtime import (
    _latest_audio_soft_q2,
    _prior_audio_replay_batch,
    build_native_control6_semantic_prefix_request,
    prepare_native_control6_semantic_prefix,
)
from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.listener import GemmaListenerState


def test_semantic_replay_keeps_only_current_q2_from_full_window():
    soft = torch.arange(256 * 1536, dtype=torch.float32).reshape(256, 1536)
    current = _latest_audio_soft_q2(soft)
    assert current.shape == (2, 1536)
    assert torch.equal(current, soft[-2:])


def test_semantic_replay_accepts_single_tick_and_rejects_partial_q2():
    single = torch.zeros((2, 1536), dtype=torch.float32)
    assert torch.equal(_latest_audio_soft_q2(single), single)
    with pytest.raises(RuntimeError, match="allineato a tick Q2"):
        _latest_audio_soft_q2(torch.zeros((3, 1536), dtype=torch.float32))


def test_semantic_prefix_replay_ends_at_previous_frame_before_current_query():
    observed = [
        (4, torch.full((2, 1536), 4.0)),
        (5, torch.full((2, 1536), 5.0)),
    ]
    batch, frame_ids, reason = _prior_audio_replay_batch(
        observed, current_frame_id=6
    )
    assert reason is None
    assert frame_ids == (4, 5)
    assert batch.shape == (1, 4, 1536)
    assert torch.equal(batch[0, :2], observed[0][1])
    assert torch.equal(batch[0, 2:], observed[1][1])


def test_semantic_prefix_replay_never_uses_a_future_or_missing_frame():
    future = [(6, torch.ones((2, 1536)))]
    batch, frame_ids, reason = _prior_audio_replay_batch(
        future, current_frame_id=6
    )
    assert batch is None and frame_ids == ()
    assert reason == "no_contiguous_observed_history_before_current_frame"


class _PrefixTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [999] if text == "<|audio|>" else [1]

    def decode(self, token_ids, **_kwargs):
        return "x" * len(token_ids)

    def apply_chat_template(self, _messages, *, tokenize, add_generation_prompt):
        assert tokenize is True and add_generation_prompt is False
        return [17, 999, 18]


def _semantic_request_fixture(*, current_frame_id=2, pending_speak=None, history_frame_ids=None):
    policy = GemmaNativeControl6Policy.__new__(GemmaNativeControl6Policy)
    state = GemmaListenerState(
        next_frame_id=current_frame_id,
        listener_input_frames=current_frame_id,
        listener_state_version=current_frame_id,
    )
    frame_ids = (
        tuple(range(current_frame_id))
        if history_frame_ids is None
        else tuple(int(frame_id) for frame_id in history_frame_ids)
    )
    prior = tuple(
        torch.full((2, 1536), float(frame_id), dtype=torch.bfloat16)
        for frame_id in frame_ids
    )
    state.semantic_audio_soft_history.extend(
        (frame_id, q2) for frame_id, q2 in zip(frame_ids, prior, strict=True)
    )
    backend = SimpleNamespace(
        boa_token_id=777,
        hidden_size=1536,
        build_semantic_prefix_cache=lambda *_args, **_kwargs: None,
    )
    session = SimpleNamespace(
        session_id="semantic-session",
        turn_policy=policy,
        listener_state=state,
        listener=SimpleNamespace(backend=backend),
        session_generation_id=19,
        semantic_context_revision=4,
        active_response=False,
        policy_response_pending=False,
        instructions="Use Italian",
        history=({"role": "user", "content": "ciao"},),
        _generation_lock=threading.RLock(),
    )
    engine = SimpleNamespace(
        native_control6_semantic_prefix_enabled=True,
        tokenizer=_PrefixTokenizer(),
        _thinker_wrapper=SimpleNamespace(processor=SimpleNamespace(audio_token="<|audio|>")),
    )
    request = build_native_control6_semantic_prefix_request(
        engine,
        session,
        current_frame_id=current_frame_id,
        pending_speak=pending_speak,
    )
    return request, prior, state


def test_semantic_prefix_request_freezes_actual_boa_and_only_prior_contiguous_q2():
    request, prior, state = _semantic_request_fixture(current_frame_id=2)

    assert request is not None
    assert request.session_id == "semantic-session"
    assert request.session_generation_id == 19
    assert request.current_frame_id == 2
    assert request.listener_state_version == 2
    assert request.semantic_context_revision == 4
    assert request.prefix_token_ids == (17, 777)
    assert request.prefix_len == 2
    assert request.replay_frame_ids == (0, 1)
    assert len(request.replay_q2) == len(prior)
    assert all(torch.equal(left, right) for left, right in zip(request.replay_q2, prior, strict=True))
    assert request.cache_position == 6
    assert state.next_frame_id == 2
    assert state.listener_state_version == 2


def test_semantic_prefix_request_defers_while_speak_pending():
    blocked, _prior, _state = _semantic_request_fixture(
        current_frame_id=1,
        pending_speak={"source": "pending"},
    )
    assert blocked is None

    request, _prior, _state = _semantic_request_fixture(current_frame_id=3)
    assert request is not None
    assert request.replay_frame_ids == (0, 1, 2)
    assert request.fallback_reason is None


def test_semantic_prefix_request_does_not_replay_when_t_minus_one_is_missing():
    request, _prior, _state = _semantic_request_fixture(
        current_frame_id=3,
        history_frame_ids=(0, 1),
    )
    assert request is not None
    assert request.replay_q2 == ()
    assert request.replay_frame_ids == ()
    assert request.cache_position == request.prefix_len
    assert request.fallback_reason == "no_contiguous_observed_history_before_current_frame"


def test_semantic_prefix_prepare_forwards_frozen_actual_length_and_prior_replay():
    request, _prior, state = _semantic_request_fixture(current_frame_id=2)
    captured = {}

    def prime(_state, prefix_token_ids, **kwargs):
        captured["prefix_token_ids"] = tuple(prefix_token_ids)
        captured.update(kwargs)

    policy = GemmaNativeControl6Policy.__new__(GemmaNativeControl6Policy)
    backend = SimpleNamespace(
        boa_token_id=777,
        hidden_size=1536,
        build_semantic_prefix_cache=lambda *_args, **_kwargs: None,
        prime_semantic_prefix=prime,
    )
    session = SimpleNamespace(
        session_id="semantic-session",
        turn_policy=policy,
        listener_state=state,
        listener=SimpleNamespace(backend=backend),
        session_generation_id=19,
        semantic_context_revision=4,
        active_response=False,
        policy_response_pending=False,
        instructions="Use Italian",
        history=({"role": "user", "content": "ciao"},),
        _generation_lock=threading.RLock(),
    )
    engine = SimpleNamespace(
        native_control6_semantic_prefix_enabled=True,
        tokenizer=_PrefixTokenizer(),
        _thinker_wrapper=SimpleNamespace(processor=SimpleNamespace(audio_token="<|audio|>")),
    )

    assert prepare_native_control6_semantic_prefix(
        engine,
        session,
        current_frame_id=2,
        pending_speak=None,
    )
    assert captured["prefix_token_ids"] == request.prefix_token_ids
    assert captured["prefix_len"] == request.prefix_len == 2
    assert captured["replay_frame_ids"] == request.replay_frame_ids == (0, 1)
    assert tuple(captured["replay_audio_soft"].shape) == (1, 4, 1536)
    assert state.semantic_context_revision == request.semantic_context_revision == 4
