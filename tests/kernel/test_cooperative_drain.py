from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import replace
import math
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import duplex_kernel.listener_microbatch_worker as worker_module
from duplex_kernel.microbatch_contract import ListenerFrame
from duplex_kernel.server_metrics import (
    _filter_unified_native_event_metrics,
    _queued_audio_drain_metric_fields,
)
from duplex_kernel.server_unified_cache import _finalize_unified_native_cache_branch
from duplex_kernel.server import _begin_private_backchannel_decode
from test_listener_microbatch_worker import (
    _CudaProvenanceTensor,
    _FakeCudaEvent,
    _assistant_q0,
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
def _rig_factory():
    rigs = []

    def create(**kwargs):
        rig = _rig(**kwargs)
        rigs.append(rig)
        return rig

    yield create
    for rig in rigs:
        if rig.native is not None and rig.native.step_continue is not None:
            rig.native.step_continue.set()
        try:
            rig.worker.stop(drain=False).result(timeout=3)
        except BaseException:
            pass


def _queue_voiced_backlog(rig, state, frame_count: int) -> None:
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
        response_id="response-cooperative-drain",
    ).result(timeout=3)
    for frame_id in range(frame_count):
        rig.worker.submit_tick((_assistant_q0(state, frame_id),)).result(timeout=4)
    _mark_queued_soft_cuda_for_cpu_fixture(rig, state.session_id)


def _post_close_frame(state, frame_id: int, queue_epoch: int) -> ListenerFrame:
    import numpy as np

    return ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=frame_id,
        ready_at_ns=time.monotonic_ns(),
        pcm=np.full(1280, 0.4, dtype=np.float32),
        logical_q_len=2,
        queue_audio_for_next_turn=False,
        queue_epoch=queue_epoch,
    )


def test_backchannel_ephemeral_fork_waits_for_worker_without_transition_lock() -> None:
    transition_lock = threading.Lock()

    def complete_with_lock_free_worker_access(value):
        future: Future = Future()

        def complete():
            if not transition_lock.acquire(blocking=False):
                future.set_exception(AssertionError(
                    "backchannel worker future waited while transition lock was held"
                ))
                return
            transition_lock.release()
            future.set_result(value)

        thread = threading.Thread(target=complete, daemon=True)
        thread.start()
        return future

    class _Worker:
        def session_state(self, session_id):
            assert session_id == "backchannel-lock-release"
            return complete_with_lock_free_worker_access(SimpleNamespace(
                generation_id=4,
                cache_position=34,
            ))

        def begin_ephemeral_decode(
            self,
            session_id,
            *,
            expected_generation_id,
            expected_cache_position,
        ):
            assert session_id == "backchannel-lock-release"
            assert expected_generation_id == 4
            assert expected_cache_position == 34
            return complete_with_lock_free_worker_access(SimpleNamespace(
                parent_generation_id=4,
                source_cache_position=34,
            ))

    async def open_fork():
        return await _begin_private_backchannel_decode(
            _Worker(),
            "backchannel-lock-release",
            can_continue=lambda: True,
            transition_lock=transition_lock,
        )

    handle = asyncio.run(open_fork())
    assert handle.parent_generation_id == 4
    assert handle.source_cache_position == 34
    assert not transition_lock.acquire(blocking=False)
    transition_lock.release()


def test_backlog_replays_one_q2_quantum_and_serves_ready_listener_tick(
    _rig_factory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _rig_factory()
    state = _state(rig.coordinator.runtime_signature, "cooperative-drain-backlog")
    frame_count = 50  # 4 seconds at the 80 ms listener frame cadence.
    _queue_voiced_backlog(rig, state, frame_count)
    rig.worker.mark_assistant_audio_queue_closing(
        state.session_id, queue_close_epoch=7
    ).result(timeout=3)

    assert rig.native is not None
    replay_start = len(rig.native.manifests)
    rig.native.step_entered = threading.Event()
    rig.native.step_continue = threading.Event()
    rig.native.step_block_at_count = len(rig.native.steps) + 1
    original_replay = rig.native.step_queued_audio_async

    def paced_replay(*args, **kwargs):
        time.sleep(0.002)
        return original_replay(*args, **kwargs)

    rig.native.step_queued_audio_async = paced_replay
    original_frame_type = worker_module.QueuedAudioSoftFrame

    def cuda_provenance_frame(*args, **kwargs):
        item = original_frame_type(*args, **kwargs)
        return replace(
            item,
            audio_soft_tokens=_CudaProvenanceTensor(item.audio_soft_tokens),
        )

    monkeypatch.setattr(worker_module, "QueuedAudioSoftFrame", cuda_provenance_frame)
    drain_future = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    )
    assert rig.native.step_entered.wait(3)

    tick_future = rig.worker.submit_tick((
        _post_close_frame(state, frame_count, queue_epoch=7),
    ))
    rig.native.step_continue.set()
    tick_output = tick_future.result(timeout=5)[state.session_id]
    assert tick_output.logical_q_len == 0
    assert tick_output.queue_audio_for_next_turn is True
    assert not drain_future.done(), "the owner must return to listener work between Q2 quanta"

    drained = drain_future.result(timeout=10)
    assert drained.queued_audio_drain_metrics.frame_count == frame_count + 1
    assert drained.queued_audio_drain_metrics.source_frame_ids == tuple(range(frame_count + 1))
    assert drained.queued_audio_drain_metrics.source_queue_epochs[-1] == 7
    assert drained.cache_position == 34 + 2 * (frame_count + 1)

    primary = [
        manifest.rows[0]
        for manifest in rig.native.manifests[replay_start:]
        if manifest.rows[0].session_id == state.session_id
    ]
    interleaved_tick_index = next(
        index for index, row in enumerate(primary)
        if row.frame_id == frame_count and row.logical_q_len == 0
    )
    later_replay_index = next(
        index for index, row in enumerate(primary)
        if row.logical_q_len == 2 and row.frame_id > frame_count
    )
    assert interleaved_tick_index < later_replay_index
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_empty_drain_is_fast_and_issues_no_native_forward(_rig_factory) -> None:
    rig = _rig_factory()
    state = _state(rig.coordinator.runtime_signature, "cooperative-drain-empty")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
    ).result(timeout=3)
    assert rig.native is not None
    forward_count = len(rig.native.steps)

    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    ).result(timeout=3)

    metrics = drained.queued_audio_drain_metrics
    assert metrics.frame_count == 0
    assert metrics.elapsed_ms < 5.0
    assert metrics.drain_replay_forward_ms == 0.0
    assert len(rig.native.steps) == forward_count
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_preexisting_post_close_q0_is_transferred_after_sealed_queue(_rig_factory) -> None:
    rig = _rig_factory()
    state = _state(rig.coordinator.runtime_signature, "cooperative-drain-handoff")
    _queue_voiced_backlog(rig, state, 2)
    rig.worker.mark_assistant_audio_queue_closing(
        state.session_id, queue_close_epoch=7
    ).result(timeout=3)
    post_close = rig.worker.submit_tick((
        _post_close_frame(state, 2, queue_epoch=7),
    )).result(timeout=4)[state.session_id]
    assert post_close.logical_q_len == 0
    _mark_queued_soft_cuda_for_cpu_fixture(rig, state.session_id)

    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    ).result(timeout=5)

    metrics = drained.queued_audio_drain_metrics
    assert metrics.source_frame_ids == (0, 1, 2)
    assert metrics.source_queue_epochs == (None, None, 7)
    assert drained.cache_position == 40
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_canonical_text_append_is_rejected_while_drain_is_active(_rig_factory) -> None:
    rig = _rig_factory()
    state = _state(rig.coordinator.runtime_signature, "cooperative-drain-text")
    _queue_voiced_backlog(rig, state, 3)
    assert rig.native is not None
    rig.native.step_entered = threading.Event()
    rig.native.step_continue = threading.Event()
    rig.native.step_block_at_count = len(rig.native.steps) + 1
    drain_future = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    )
    assert rig.native.step_entered.wait(3)

    append_future = rig.worker.append_text_tokens(
        state.session_id,
        (901, 902),
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    )
    rig.native.step_continue.set()
    with pytest.raises(RuntimeError, match="drain is active"):
        append_future.result(timeout=5)
    drain_future.result(timeout=5)
    assert rig.worker.session_state(state.session_id).result(timeout=3).cache_position == 40
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_reset_waits_until_active_drain_cleanup_is_published(_rig_factory) -> None:
    rig = _rig_factory()
    state = _state(rig.coordinator.runtime_signature, "cooperative-drain-reset")
    _queue_voiced_backlog(rig, state, 3)
    assert rig.native is not None
    rig.native.step_entered = threading.Event()
    rig.native.step_continue = threading.Event()
    rig.native.step_block_at_count = len(rig.native.steps) + 1
    drain_future = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    )
    assert rig.native.step_entered.wait(3)
    next_state = _state(
        rig.coordinator.runtime_signature,
        state.session_id,
        generation=state.generation_id + 1,
    )

    reset_future = rig.worker.reset_session(
        state.session_id,
        next_state,
        prefix_import=rig.prefix_import,
    )
    original_reset = rig.worker._reset_session

    def ordered_reset(*args, **kwargs):
        assert drain_future.done()
        assert state.session_id not in rig.worker._active_queued_audio_drains
        return original_reset(*args, **kwargs)

    rig.worker._reset_session = ordered_reset
    time.sleep(0.02)
    assert not reset_future.done(), "reset must wait behind owner drain cleanup"
    rig.native.step_continue.set()

    drained = drain_future.result(timeout=5)
    reset_future.result(timeout=5)
    current = rig.worker.session_state(state.session_id).result(timeout=3)
    assert drained.cache_position == 40
    assert current.generation_id == state.generation_id + 1
    assert current.cache_position == 34
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_cancelled_finalizer_waits_for_ordered_drain_cleanup(_rig_factory) -> None:
    async def run_test() -> None:
        rig = _rig_factory()
        state = _state(rig.coordinator.runtime_signature, "cooperative-drain-cancel")
        _queue_voiced_backlog(rig, state, 3)
        assert rig.native is not None
        rig.native.step_entered = threading.Event()
        rig.native.step_continue = threading.Event()
        rig.native.step_block_at_count = len(rig.native.steps) + 1
        owner_futures: list[Future] = []
        submit_drain = rig.worker.drain_queued_audio

        def capture_drain_future(*args, **kwargs):
            future = submit_drain(*args, **kwargs)
            owner_futures.append(future)
            return future

        rig.worker.drain_queued_audio = capture_drain_future
        transition_lock = threading.Lock()
        branch = {
            "listener_generation_id": state.generation_id,
            "turn_start_cache_position": 34,
            "cache_position": 34,
            "queue_open": True,
            "queue_open_epoch": 0,
            "queue_boundary_epoch": 0,
            "queue_epoch_state": {"value": 0},
            "transition_lock": transition_lock,
        }

        task = asyncio.create_task(_finalize_unified_native_cache_branch(
            rig.worker,
            state.session_id,
            branch,
            history_committed=False,
            close_ids=(),
        ))
        assert await asyncio.to_thread(rig.native.step_entered.wait, 3)
        assert transition_lock.acquire(blocking=False), (
            "the transition lock must be released while the owner Future is pending"
        )
        transition_lock.release()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cancellation must wait for the owner to clean up the drain"
        assert owner_futures and not owner_futures[0].cancelled()

        rig.native.step_continue.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        drained = owner_futures[0].result(timeout=3)
        snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
        assert snapshot.cache_position == drained.cache_position == 40
        assert owner_futures[0].done() and not owner_futures[0].cancelled()

    asyncio.run(run_test())


def test_drain_phase_metrics_are_separate_and_emitted(_rig_factory) -> None:
    rig = _rig_factory()
    state = _state(rig.coordinator.runtime_signature, "cooperative-drain-metrics")
    _queue_voiced_backlog(rig, state, 2)
    transition_lock = threading.Lock()
    branch = {
        "listener_generation_id": state.generation_id,
        "turn_start_cache_position": 34,
        "cache_position": 34,
        "queue_open": True,
        "queue_open_epoch": 0,
        "queue_boundary_epoch": 0,
        "queue_epoch_state": {"value": 0},
        "transition_lock": transition_lock,
    }

    drained = asyncio.run(_finalize_unified_native_cache_branch(
        rig.worker,
        state.session_id,
        branch,
        history_committed=False,
        close_ids=(),
    ))
    metrics = drained.queued_audio_drain_metrics
    assert metrics.drain_queue_wait_ms >= 0.0
    assert metrics.drain_transition_lock_wait_ms >= 0.0
    assert metrics.drain_replay_forward_ms > 0.0
    assert metrics.drain_completion_wait_ms >= 0.0
    flattened = _queued_audio_drain_metric_fields(drained)
    filtered = _filter_unified_native_event_metrics(flattened)
    for key in (
        "drain_queue_wait_ms",
        "owner_wait_ms",
        "drain_transition_lock_wait_ms",
        "drain_replay_forward_ms",
        "drain_completion_wait_ms",
    ):
        assert key in flattened and key in filtered
        assert math.isfinite(float(filtered[key]))
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly
