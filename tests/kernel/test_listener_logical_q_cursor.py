from __future__ import annotations

import time

import numpy as np
import pytest

from duplex_kernel.listener_microbatch import ListenerMicrobatchCoordinator
from duplex_kernel.microbatch_contract import (
    CANONICAL_PREFIX_TOKENS,
    ListenerFrame,
    ListenerSessionState,
    NativeBatchCapability,
    RowStateUpdate,
    RuntimeSignature,
)


def _setup(cache_position: int = CANONICAL_PREFIX_TOKENS):
    signature = RuntimeSignature(
        model_id="gemma-test",
        model_config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        topology_sha256="c" * 64,
        native_library_sha256="d" * 64,
        native_abi_version=3,
        batch_api_version=3,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
    )
    capability = NativeBatchCapability(
        api_version=3,
        native_abi_version=3,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
        supports_arbitrary_batch_size=True,
        stream_priority=-1,
        stream_identity="listener-test-stream",
    )
    coordinator = ListenerMicrobatchCoordinator(
        capability=capability, runtime_signature=signature, coalesce_window_ns=0
    )
    coordinator.register_session(ListenerSessionState(
        session_id="s",
        generation_id=1,
        state_epoch_id="epoch-s",
        runtime_signature=signature,
        cache_position=cache_position,
        valid_history_tokens=cache_position,
        native_context_id=9,
    ))
    return coordinator


def _frame(
    logical_q_len: int,
    *,
    ephemeral_reason: str | None = None,
    queue_audio_for_next_turn: bool = False,
) -> ListenerFrame:
    return ListenerFrame(
        session_id="s",
        generation_id=1,
        state_epoch_id="epoch-s",
        frame_id=0,
        ready_at_ns=time.monotonic_ns(),
        pcm=np.zeros(1280, dtype=np.float32),
        logical_q_len=logical_q_len,
        ephemeral_reason=ephemeral_reason,
        queue_audio_for_next_turn=queue_audio_for_next_turn,
    )


@pytest.mark.parametrize("logical_q_len", [0, 1, 2])
def test_logical_query_count_controls_cache_commit_but_not_input_sequence(logical_q_len: int) -> None:
    coordinator = _setup()
    coordinator.enqueue(_frame(
        logical_q_len,
        ephemeral_reason="prespeech_silence" if logical_q_len == 0 else None,
    ))
    batch = coordinator.take_ready(time.monotonic_ns())
    assert batch is not None
    row = batch.manifest.rows[0]
    assert row.logical_q_len == logical_q_len

    coordinator.complete(batch, (RowStateUpdate(
        session_id="s",
        frame_id=0,
        cache_position=row.cache_position + logical_q_len,
        valid_history_tokens=row.valid_history_tokens + logical_q_len,
    ),))

    state = coordinator.session_state("s")
    assert state.next_frame_id == 1
    assert state.state_version == 1
    assert state.sequence_position == 1
    assert state.cache_position == CANONICAL_PREFIX_TOKENS + logical_q_len
    assert state.valid_history_tokens == CANONICAL_PREFIX_TOKENS + logical_q_len


def test_physical_q2_capacity_is_still_required_for_logical_q1() -> None:
    coordinator = _setup(cache_position=16_383)
    frame = _frame(1)
    coordinator.enqueue(frame)
    with pytest.raises(RuntimeError, match="physical Q2 capacity exhausted before dispatch"):
        coordinator.take_ready(time.monotonic_ns())


@pytest.mark.parametrize("reason", ["prespeech_silence", "assistant_turn_open"])
def test_ephemeral_reasons_are_explicit_and_assistant_audio_is_marked_for_next_turn(reason: str) -> None:
    queue_audio = reason == "assistant_turn_open"
    frame = _frame(0, ephemeral_reason=reason, queue_audio_for_next_turn=queue_audio)
    coordinator = _setup()
    coordinator.enqueue(frame)
    batch = coordinator.take_ready(time.monotonic_ns())
    assert batch is not None
    row = batch.manifest.rows[0]
    assert row.logical_q_len == 0
    assert row.ephemeral_reason == reason
    assert row.queue_audio_for_next_turn is queue_audio


def test_zero_logical_q_requires_explicit_ephemeral_reason() -> None:
    with pytest.raises(ValueError, match="requires an explicit ephemeral_reason"):
        _frame(0)


def test_alternating_audio_text_and_ephemeral_audio_keeps_cursors_aligned() -> None:
    coordinator = _setup()
    first = _frame(2)
    coordinator.enqueue(first)
    audio_batch = coordinator.take_ready(time.monotonic_ns())
    assert audio_batch is not None
    audio_row = audio_batch.manifest.rows[0]
    coordinator.complete(audio_batch, (RowStateUpdate(
        session_id="s", frame_id=0, cache_position=audio_row.cache_position + 2,
        valid_history_tokens=audio_row.valid_history_tokens + 2,
    ),))

    text_after_audio = coordinator.advance_cache_position(
        "s", token_count=1, expected_generation_id=1, expected_cache_position=36,
    )
    assert (text_after_audio.next_frame_id, text_after_audio.state_version,
            text_after_audio.sequence_position, text_after_audio.cache_position) == (1, 1, 1, 37)

    silent_tick = ListenerFrame(
        session_id="s", generation_id=1, state_epoch_id="epoch-s", frame_id=1,
        ready_at_ns=time.monotonic_ns(), pcm=np.zeros(1280, dtype=np.float32),
        logical_q_len=0, ephemeral_reason="assistant_turn_open",
        queue_audio_for_next_turn=True,
    )
    coordinator.enqueue(silent_tick)
    ephemeral_batch = coordinator.take_ready(time.monotonic_ns())
    assert ephemeral_batch is not None
    ephemeral_row = ephemeral_batch.manifest.rows[0]
    coordinator.complete(ephemeral_batch, (RowStateUpdate(
        session_id="s", frame_id=1,
        cache_position=ephemeral_row.cache_position,
        valid_history_tokens=ephemeral_row.valid_history_tokens,
    ),))

    text_after_ephemeral = coordinator.advance_cache_position(
        "s", token_count=1, expected_generation_id=1, expected_cache_position=37,
    )
    assert (text_after_ephemeral.next_frame_id, text_after_ephemeral.state_version,
            text_after_ephemeral.sequence_position, text_after_ephemeral.cache_position) == (2, 2, 2, 38)
