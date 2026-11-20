from __future__ import annotations

from dataclasses import replace
from threading import RLock

import numpy as np
import pytest
import torch

from duplex_kernel.turn_policy import (
    ControlAction,
    ControlState,
    PolicyDecision,
    PolicyInput,
    SpeakerGate,
)
from test_listener_microbatch_worker import (
    _assistant_q0,
    _frame,
    _prime_native_prefix,
    _register,
    _rig,
    _state,
)


class _FakeCudaEvent:
    def __init__(self, **_kwargs) -> None:
        pass

    def record(self, _stream) -> None:
        pass


@pytest.fixture(autouse=True)
def _fake_cuda_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)


class _YieldBatchPolicy:
    policy_id = "test_yield_batch_policy"

    def step_native_output_views(self, row_sources, policy_inputs):
        decisions = []
        for row_source, policy_input in zip(row_sources, policy_inputs, strict=True):
            row = row_source.decision_row
            decisions.append(PolicyDecision(
                action=ControlAction.YIELD,
                frame_id=row.frame_id,
                generation_id=policy_input.generation_id,
                listener_frame_id=row.frame_id,
                session=row.session_id,
                source="test_yield_batch_policy",
            ))
        return tuple(decisions)


def _enable_voiced_speaker_policy(rig, generation_id: int) -> None:
    assert rig.worker._runtime is not None
    rig.worker._runtime.policy = _YieldBatchPolicy()
    rig.worker._runtime.policy_inputs = lambda batch, states: tuple(
        PolicyInput(
            session_id=row.session_id,
            frame_id=row.frame_id,
            user_pcm=frame.pcm,
            speaker_active=True,
            generation_id=generation_id,
            legal_actions=("CONTINUE", "YIELD"),
        )
        for row, frame in zip(batch.manifest.rows, batch.frames, strict=True)
    )


def test_queue_open_forks_before_text_and_accepts_previous_frame_watermark() -> None:
    rig = _rig(stable_turn_policy_q2_context=True)
    state = _state(rig.coordinator.runtime_signature, "speaking-context-open")
    _register(rig, state)
    _prime_native_prefix(rig, state)

    # Give queue-open a nonzero frame watermark, then model a previous row whose
    # frame ID is one behind the snapshot while its state/cursors are current.
    rig.worker.submit_tick((_frame(state, frame_id=0),)).result(timeout=4)
    before_open = rig.worker.session_state(state.session_id).result(timeout=3)
    previous_row = rig.worker._last_native_rows[state.session_id]
    assert previous_row.frame_id == before_open.next_frame_id
    rig.worker._last_native_rows[state.session_id] = replace(
        previous_row, frame_id=before_open.next_frame_id - 1,
    )

    capacity_calls_before_open = len(rig.native.capacity_requests)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=before_open.cache_position,
        reserve_tail_tokens=0,
        response_id="speaking-context-open-response",
    ).result(timeout=3)

    cache = rig.worker._q2_decision_caches[state.session_id]
    assert len(rig.native.slot_fork_calls) == 1
    assert len(rig.native.capacity_requests) == capacity_calls_before_open + 1
    assert cache.base_cache_position == before_open.cache_position
    assert cache.last_applied_frame_id == before_open.next_frame_id - 1
    assert cache.q2_applied_count == 0
    assert cache.fork_row.cache_position == before_open.cache_position
    assert rig.native.slot_fork_calls[0][0].cache_position == before_open.cache_position

    # Queue-open captured the private slot before the Thinker advances canonical
    # text. The fork stays anchored at its original cursor.
    rig.worker.append_text_tokens(
        state.session_id,
        (901, 902),
        expected_generation_id=state.generation_id,
        expected_cache_position=before_open.cache_position,
        append_scope_id="speaking-context-open-text",
    ).result(timeout=4)
    after_text = rig.worker.session_state(state.session_id).result(timeout=3)
    assert after_text.cache_position == before_open.cache_position + 2
    assert rig.worker._q2_decision_caches[state.session_id].fork_row.cache_position == (
        before_open.cache_position
    )
    assert len(rig.native.slot_fork_calls) == 1

    rig.worker.stop(drain=False).result(timeout=3)


def test_metadata_rebind_preserves_private_kv_cursor_across_text_and_listener_ticks() -> None:
    rig = _rig(stable_turn_policy_q2_context=True)
    state = _state(rig.coordinator.runtime_signature, "speaking-context-rebind")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
    ).result(timeout=3)

    first = rig.worker.submit_tick((
        _assistant_q0(state, 0, queue_audio_for_next_turn=False),
    )).result(timeout=4)[state.session_id]
    assert first.decision_view is not None
    assert rig.worker._q2_decision_caches[state.session_id].fork_row.cache_position == 36

    rig.worker.append_text_tokens(
        state.session_id,
        (903,),
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        append_scope_id="speaking-context-rebind-text",
    ).result(timeout=4)
    capacity_calls_before_second_tick = len(rig.native.capacity_requests)
    second = rig.worker.submit_tick((
        _assistant_q0(state, 1, queue_audio_for_next_turn=False),
    )).result(timeout=4)[state.session_id]

    assert second.decision_view is not None
    assert len(rig.native.capacity_requests) >= capacity_calls_before_second_tick + 1
    assert len(rig.native.slot_fork_calls) == 1
    assert len(rig.native.metadata_rebind_calls) == 2
    old_fork_row, current_listener_row = rig.native.metadata_rebind_calls[-1]
    assert old_fork_row.cache_position == 36
    assert current_listener_row.cache_position == 35
    assert second.decision_view.fork_row.cache_position == 36
    cache = rig.worker._q2_decision_caches[state.session_id]
    assert cache.base_cache_position == 34
    assert cache.last_applied_frame_id == 1
    assert cache.q2_applied_count == 2
    assert cache.fork_row.cache_position == 38

    rig.worker.stop(drain=False).result(timeout=3)


def test_voiced_yield_uses_q2_fork_when_retention_is_false() -> None:
    rig = _rig(stable_turn_policy_q2_context=True)
    state = _state(rig.coordinator.runtime_signature, "speaking-context-yield")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
    ).result(timeout=3)
    _enable_voiced_speaker_policy(rig, state.generation_id)

    output = rig.worker.submit_tick((
        _assistant_q0(
            state,
            0,
            queue_audio_for_next_turn=False,
            pcm=np.full(1280, 0.25, dtype=np.float32),
        ),
    )).result(timeout=4)[state.session_id]

    assert output.decision_view is not None
    assert output.queue_audio_for_next_turn is False
    assert output.policy_decision is not None
    assert output.policy_decision.action is ControlAction.YIELD
    assert output.decision_view.source_row.frame_id == 0
    assert output.native_output_health.row.session_id == state.session_id
    assert output.native_output_health.row.logical_q_len == 0
    assert output.decision_view.fork_row.session_id.startswith("__q2view__")
    assert not rig.worker._queued_audio_soft[state.session_id]
    assert not rig.worker._policy_q2_soft_pending[state.session_id]
    canonical = rig.worker.session_state(state.session_id).result(timeout=3)
    assert canonical.cache_position == canonical.valid_history_tokens == 34
    cache = rig.worker._q2_decision_caches[state.session_id]
    assert cache.base_cache_position == 34
    assert cache.last_applied_frame_id == 0
    assert cache.q2_applied_count == 1
    assert cache.fork_row.cache_position == 36

    rig.worker.stop(drain=False).result(timeout=3)


def test_q2_fork_cursor_is_derived_from_compact_watermark() -> None:
    rig = _rig(stable_turn_policy_q2_context=True)
    state = _state(rig.coordinator.runtime_signature, "speaking-context-watermark")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
    ).result(timeout=3)

    for frame_id in range(5):
        output = rig.worker.submit_tick((
            _assistant_q0(state, frame_id, queue_audio_for_next_turn=False),
        )).result(timeout=4)[state.session_id]
        assert output.decision_view is not None
        cache = rig.worker._q2_decision_caches[state.session_id]
        assert cache.q2_applied_count == frame_id + 1
        assert cache.last_applied_frame_id == frame_id
        assert cache.fork_row.cache_position == cache.base_cache_position + 2 * cache.q2_applied_count
        assert rig.worker._q2_cache_expected_cursor(cache) == (
            cache.base_cache_position + 2 * cache.q2_applied_count
        )

    cache = rig.worker._q2_decision_caches[state.session_id]
    assert not hasattr(cache, "frame_keys")
    assert cache.q2_applied_count == 5
    assert cache.last_applied_frame_id == 4
    assert cache.fork_row.cache_position == 44
    assert len(rig.native.slot_fork_calls) == 1

    rig.worker.stop(drain=False).result(timeout=3)


def test_lost_stable_fork_fails_closed_and_cleanup_releases_queue_state() -> None:
    rig = _rig(stable_turn_policy_q2_context=True)
    state = _state(rig.coordinator.runtime_signature, "speaking-context-lost")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
    ).result(timeout=3)

    first = rig.worker.submit_tick((
        _assistant_q0(state, 0, queue_audio_for_next_turn=False),
    )).result(timeout=4)[state.session_id]
    assert first.decision_view is not None
    prior_cache = rig.worker._q2_decision_caches[state.session_id]
    fork_session_id = prior_cache.fork_row.session_id
    prior_q2_step_count = sum(
        bool(session_ids) and all(sid.startswith("__q2view__") for sid in session_ids)
        for session_ids, _thread_id in rig.native.steps
    )
    rig.native._metadata_rebind_api_available = False

    failed_closed = rig.worker.submit_tick((
        _assistant_q0(state, 1, queue_audio_for_next_turn=False),
    )).result(timeout=4)[state.session_id]

    assert failed_closed.decision_view is None
    assert failed_closed.decision_view_unavailable_reason
    assert failed_closed.policy_decision is not None
    assert failed_closed.policy_decision.action is ControlAction.HARD_STOP
    assert failed_closed.policy_decision.reason == "stable_q2_context_unavailable_fail_closed"
    assert failed_closed.policy_decision.source == "control6_context_fail_closed"
    assert state.session_id not in rig.worker._q2_decision_caches
    assert [release[0] for release in rig.native.releases].count(fork_session_id) == 1
    assert prior_cache.last_applied_frame_id == 0
    assert prior_cache.q2_applied_count == 1
    assert sum(
        bool(session_ids) and all(sid.startswith("__q2view__") for sid in session_ids)
        for session_ids, _thread_id in rig.native.steps
    ) == prior_q2_step_count

    class _SpeakingSession:
        def __init__(self) -> None:
            self._generation_lock = RLock()
            self.session_generation_id = state.generation_id
            self.speaker_segment_id = "speaking-context-lost-segment"
            self.control_state = ControlState.SPEAKING
            self.stop_requested = False
            self.stop_reason = None

        def invalidate_generation(self, _reason: str) -> None:
            self.session_generation_id += 1

        def request_control_segment_close(self) -> None:
            pass

    gate = SpeakerGate().apply(failed_closed.policy_decision, _SpeakingSession())
    assert gate.accepted is True
    assert gate.action is ControlAction.HARD_STOP
    assert gate.stop_synthesis is True
    assert gate.flush_audio_buffer is True
    assert gate.next_state is ControlState.STOPPING

    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        user_turn_already_open=False,
    ).result(timeout=4)
    assert drained.queued_audio_drain_metrics.frame_count == 0
    assert drained.cache_position == 34
    assert state.session_id not in rig.worker._assistant_audio_queue_open
    assert not rig.worker._policy_q2_soft_pending[state.session_id]
    assert state.session_id not in rig.worker._q2_decision_caches

    rig.worker.stop(drain=True).result(timeout=3)
