from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pytest
import torch

from duplex_kernel.server_unified_templates import (
    _listener_frame_is_voiced,
    _unified_listener_frame_contract,
)
from test_listener_microbatch_worker import (
    _FakeCudaEvent,
    _mark_queued_soft_cuda_for_cpu_fixture,
    _prime_native_prefix,
    _register,
    _rig,
    _state,
)


@pytest.fixture(autouse=True)
def _cpu_fake_cuda_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)


@pytest.fixture
def rig_factory():
    rigs = []

    def create(**kwargs):
        rig = _rig(**kwargs)
        rigs.append(rig)
        return rig

    yield create
    for rig in rigs:
        try:
            rig.worker.stop(drain=False).result(timeout=3)
        except BaseException:
            pass


def _assistant_q0(
    state,
    frame_id: int,
    *,
    retained: bool = False,
    queue_epoch: int | None = 5,
    reason: str | None = None,
    onset_frame_id: int | None = None,
    pcm: np.ndarray | None = None,
):
    from duplex_kernel.microbatch_contract import ListenerFrame

    return ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=frame_id,
        ready_at_ns=time.monotonic_ns(),
        pcm=(
            np.full(1280, 0.25, dtype=np.float32)
            if pcm is None else pcm
        ),
        logical_q_len=0,
        ephemeral_reason="assistant_turn_open",
        queue_audio_for_next_turn=retained,
        queue_epoch=queue_epoch,
        queue_audio_reason=reason,
        queue_audio_onset_frame_id=onset_frame_id,
    )


def _open_queue(rig, state, *, response_id: str = "onset-preroll") -> None:
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
        response_id=response_id,
    ).result(timeout=3)


def _submit(rig, frame):
    return rig.worker.submit_tick((frame,)).result(timeout=4)[frame.session_id]


def _retained_spans(rig, session_id: str):
    return [
        span
        for span in rig.worker._assistant_audio_queue_frame_spans[session_id]
        if span.retained
    ]


def test_weak_voice_and_fricative_frames_have_no_static_rms_cutoff() -> None:
    weak_fricative = np.zeros(1280, dtype=np.float32)
    weak_fricative[100:116] = 1e-4
    assert np.sqrt(np.mean(np.square(weak_fricative))) < 0.005
    assert _listener_frame_is_voiced(weak_fricative)
    assert _unified_listener_frame_contract(
        weak_fricative,
        assistant_audio_queue_open=True,
        user_input_pending=False,
        voice_active=True,
    ) == (2, None, False)
    assert _unified_listener_frame_contract(
        weak_fricative,
        assistant_audio_queue_open=True,
        user_input_pending=False,
        voice_active=False,
    ) == (2, None, False)


def test_onset_atomically_transfers_three_q2_preroll_frames_without_duplicates(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-three-frames")
    _open_queue(rig, state)

    for frame_id in range(3):
        _submit(rig, _assistant_q0(state, frame_id, queue_epoch=12))
    ring_before_onset = tuple(rig.worker._pre_roll_ring[state.session_id])
    assert [item.source_frame_id for item in ring_before_onset] == [0, 1, 2]
    assert not rig.worker._queued_audio_soft[state.session_id]

    _submit(rig, _assistant_q0(
        state,
        3,
        retained=True,
        queue_epoch=12,
        reason="utterance_body",
        onset_frame_id=3,
    ))

    queued = tuple(rig.worker._queued_audio_soft[state.session_id])
    assert [item.source_frame_id for item in queued] == [0, 1, 2, 3]
    assert len({item.source_frame_id for item in queued}) == len(queued)
    assert len({item.audio_soft_tokens.data_ptr() for item in queued}) == len(queued)
    assert all(
        torch.equal(queued[index].audio_soft_tokens, ring_before_onset[index].audio_soft_tokens)
        for index in range(3)
    )
    assert not rig.worker._pre_roll_ring[state.session_id]
    spans = _retained_spans(rig, state.session_id)
    assert [
        (span.first_frame_id, span.last_frame_id, span.reason, span.onset_frame_id, span.queue_epoch)
        for span in spans
    ] == [
        (0, 2, "pre_roll", 3, 12),
        (3, 3, "utterance_body", 3, 12),
    ]


def test_continuous_noise_ring_is_never_replayed_without_vad_onset(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-noise-no-onset")
    _open_queue(rig, state)
    noise = np.full(1280, 0.001, dtype=np.float32)

    for frame_id in range(12):
        _submit(rig, _assistant_q0(
            state,
            frame_id,
            queue_epoch=3,
            pcm=noise,
        ))

    assert len(rig.worker._pre_roll_ring[state.session_id]) == 3
    assert not rig.worker._queued_audio_soft[state.session_id]
    assert all(not span.retained for span in rig.worker._assistant_audio_queue_frame_spans[state.session_id])
    assert rig.native is not None
    forwards_before_drain = len(rig.native.steps)
    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    ).result(timeout=4)
    assert len(rig.native.steps) == forwards_before_drain
    assert drained.queued_audio_drain_metrics.frame_count == 0
    assert drained.queued_audio_drain_metrics.token_count == 0
    assert drained.queued_audio_drain_metrics.source_frame_ids == ()
    assert drained.queued_audio_drain_metrics.elapsed_ms < 5.0
    assert not rig.worker._pre_roll_ring[state.session_id]


def test_two_onsets_keep_pause_frames_under_separate_onset_owners(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-two-onsets")
    _open_queue(rig, state)

    for frame_id in range(3):
        _submit(rig, _assistant_q0(state, frame_id, queue_epoch=7))
    for frame_id, reason in ((3, "utterance_body"), (4, "utterance_body"), (5, "endpoint_tail")):
        _submit(rig, _assistant_q0(
            state,
            frame_id,
            retained=True,
            queue_epoch=7,
            reason=reason,
            onset_frame_id=3,
        ))
    for frame_id in range(6, 9):
        _submit(rig, _assistant_q0(state, frame_id, queue_epoch=7))
    _submit(rig, _assistant_q0(
        state,
        9,
        retained=True,
        queue_epoch=7,
        reason="utterance_body",
        onset_frame_id=9,
    ))

    queued_ids = [item.source_frame_id for item in rig.worker._queued_audio_soft[state.session_id]]
    assert queued_ids == list(range(10))
    assert len(set(queued_ids)) == 10
    span_rows = [
        (span.first_frame_id, span.last_frame_id, span.reason, span.onset_frame_id)
        for span in _retained_spans(rig, state.session_id)
    ]
    assert span_rows == [
        (0, 2, "pre_roll", 3),
        (3, 4, "utterance_body", 3),
        (5, 5, "endpoint_tail", 3),
        (6, 8, "pre_roll", 9),
        (9, 9, "utterance_body", 9),
    ]


def test_pre_roll_from_prior_generation_or_queue_epoch_is_rejected(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-isolation")
    _open_queue(rig, state)
    for frame_id in range(3):
        _submit(rig, _assistant_q0(state, frame_id, queue_epoch=12))

    ring = rig.worker._pre_roll_ring[state.session_id]
    ring[0] = replace(
        ring[0],
        session_generation=state.generation_id - 1,
        source_queue_epoch=11,
    )
    _submit(rig, _assistant_q0(
        state,
        3,
        retained=True,
        queue_epoch=12,
        reason="utterance_body",
        onset_frame_id=3,
    ))

    queued_ids = [item.source_frame_id for item in rig.worker._queued_audio_soft[state.session_id]]
    assert queued_ids == [1, 2, 3]
    assert 0 not in queued_ids
    assert not rig.worker._pre_roll_ring[state.session_id]


def test_onset_at_queue_close_preserves_original_capture_ids_in_drain(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-close-boundary")
    _open_queue(rig, state)
    for frame_id in range(3):
        _submit(rig, _assistant_q0(state, frame_id, queue_epoch=8))
    rig.worker.mark_assistant_audio_queue_closing(
        state.session_id,
        queue_close_epoch=8,
    ).result(timeout=3)
    from duplex_kernel.microbatch_contract import ListenerFrame

    # This captured Q2 frame arrives after queue-close publication. The worker
    # recategorizes it as assistant-phase Q0 while keeping its capture ID.
    _submit(rig, ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=3,
        ready_at_ns=time.monotonic_ns(),
        pcm=np.full(1280, 0.25, dtype=np.float32),
        logical_q_len=2,
        queue_epoch=8,
    ))
    assert [item.source_frame_id for item in rig.worker._queued_audio_soft[state.session_id]] == [0, 1, 2, 3]
    _mark_queued_soft_cuda_for_cpu_fixture(rig, state.session_id)

    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    ).result(timeout=6)
    assert drained.queued_audio_drain_metrics.source_frame_ids == (0, 1, 2, 3)
    assert drained.queued_audio_drain_metrics.frame_count == 4


def test_reset_and_response_rollback_clear_speculative_ring(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-lifecycle")
    _open_queue(rig, state)
    for frame_id in range(2):
        _submit(rig, _assistant_q0(state, frame_id, queue_epoch=2))
    assert len(rig.worker._pre_roll_ring[state.session_id]) == 2

    rig.worker.rollback_assistant_branch(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        target_cache_position=34,
    ).result(timeout=4)
    assert not rig.worker._pre_roll_ring[state.session_id]
    assert not rig.worker._queued_audio_soft[state.session_id]

    new_state = _state(rig.coordinator.runtime_signature, state.session_id, generation=8)
    rig.worker.reset_session(
        state.session_id,
        new_state,
        prefix_import=rig.prefix_import,
    ).result(timeout=4)
    assert not rig.worker._pre_roll_ring[state.session_id]
    assert rig.worker._assistant_audio_active_onset[state.session_id] is None


def test_listener_error_discards_ring_and_uncommitted_retention(rig_factory) -> None:
    rig = rig_factory()
    state = _state(rig.coordinator.runtime_signature, "preroll-error-cleanup")
    _open_queue(rig, state)
    _submit(rig, _assistant_q0(state, 0, queue_epoch=4))
    assert len(rig.worker._pre_roll_ring[state.session_id]) == 1
    assert rig.native is not None
    rig.native.fail_step = RuntimeError("fixture native step failure")

    with pytest.raises(RuntimeError, match="fixture native step failure"):
        _submit(rig, _assistant_q0(state, 1, queue_epoch=4))

    assert not rig.worker._pre_roll_ring[state.session_id]
    assert not rig.worker._queued_audio_soft[state.session_id]
    assert not rig.worker._policy_q2_soft_pending[state.session_id]
    assert state.session_id not in rig.worker._assistant_audio_queue_open
