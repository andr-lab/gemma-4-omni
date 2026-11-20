from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_summary_rebuild import memory_owner, revision, sources
from duplex_kernel.listener_microbatch_worker import (
    CanonicalListenerQ2Frame,
    MemoryMaintenanceState as State,
    MemoryPrefillChunk,
    ResponseDecodePurpose,
)
from duplex_kernel.microbatch_contract import BatchRowDescriptor, ListenerFrame
from duplex_kernel.native_listener_microbatch import (
    NativeGemmaListenerMicrobatch,
    _SessionBinding,
)
from test_branch_lifecycle_budgets import append, begin


class Fence:
    def __init__(self, ready: bool = False):
        self.ready = ready
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.ready


def _canonical_q2(state, frame_id: int, value: int) -> CanonicalListenerQ2Frame:
    row = BatchRowDescriptor(
        session_id=state.session_id,
        session_generation=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=frame_id + 1,
        listener_state_version=frame_id + 1,
        sequence_position=frame_id + 1,
        cache_position=34 + 2 * (frame_id + 1),
        valid_history_tokens=34 + 2 * (frame_id + 1),
        native_context_id=state.native_context_id,
        input_frame_sha256=f"{frame_id:064x}",
        reference_pcm_sha256=None,
        frame_ready_at_ns=frame_id,
        reference_frame_id=None,
    )
    frame = ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=frame_id,
        ready_at_ns=frame_id,
        pcm=np.zeros(1280, dtype=np.float32),
    )
    return CanonicalListenerQ2Frame(
        frame=frame,
        row=row,
        audio_soft_tokens=torch.full((2, 1536), value, dtype=torch.bfloat16),
    )


def _rebuild_to_catch_up(worker, *, cutoff: int):
    handle = worker._begin_canonical_rebuild("s", revision(), sources(through=cutoff))
    while worker._maintenance_jobs["s"].pending_memory_tokens:
        worker._append_rebuild_chunk(handle, None)
    return handle


def test_live_q2_catch_up_is_incremental_contiguous_and_duplicate_free(memory_owner):
    worker, state, native = memory_owner
    ring = worker._recent_listener_q2["s"]
    ring.extend(_canonical_q2(state, frame_id, frame_id + 1) for frame_id in range(3))
    state.next_frame_id = 3
    state.state_version = state.sequence_position = 3

    live_inputs = []
    original = native.step_queued_audio_async

    def capture(manifest, soft, **kwargs):
        live_inputs.append(float(soft[0, 0, 0]))
        return original(manifest, soft, **kwargs)

    native.step_queued_audio_async = capture
    handle = _rebuild_to_catch_up(worker, cutoff=0)
    worker._append_rebuild_chunk(handle, MemoryPrefillChunk(
        audio_soft=np.ones((2, 1536), dtype=np.float32), first_frame_id=0, last_frame_id=0,
    ))
    worker._append_rebuild_chunk(handle, None)
    assert worker._maintenance_states["s"] is State.CATCH_UP

    first = worker._append_rebuild_chunk(handle, None)
    assert first["covered_through_frame_id"] == 1
    assert worker._maintenance_states["s"] is State.CATCH_UP
    second = worker._append_rebuild_chunk(handle, None)
    assert second["covered_through_frame_id"] == 2
    worker._append_rebuild_chunk(handle, None)

    assert worker._maintenance_states["s"] is State.READY
    assert live_inputs == [2.0, 3.0]
    assert worker._maintenance_jobs["s"].covered_through_frame_id == 2


def test_native_binding_promotion_swaps_descriptors_without_copying_kv():
    native = NativeGemmaListenerMicrobatch.__new__(NativeGemmaListenerMicrobatch)
    native._closed = False
    native._active_output_generation = None
    native._session_by_id = {}
    native._session_by_context = {}
    native._assert_open = lambda: None

    canonical = BatchRowDescriptor(
        session_id="s", session_generation=7, state_epoch_id="epoch", frame_id=4,
        listener_state_version=4, sequence_position=4, cache_position=42,
        valid_history_tokens=42, native_context_id=101, input_frame_sha256="a" * 64,
        reference_pcm_sha256=None, frame_ready_at_ns=0, reference_frame_id=None,
    )
    work = replace(canonical, session_id="__memory__s__job", native_context_id=202)
    canonical_binding = _SessionBinding(
        slot=0, session_id="s", native_context_id=101, generation=7,
        epoch_text="epoch", epoch_numeric=7, expected_frame_id=4,
        expected_state_version=4, expected_sequence_position=4,
        cache_position=42, valid_history_tokens=42, prefix_len=34,
    )
    work_binding = _SessionBinding(
        slot=1, session_id=work.session_id, native_context_id=202, generation=7,
        epoch_text="job-epoch", epoch_numeric=8, expected_frame_id=0,
        expected_state_version=0, expected_sequence_position=0,
        cache_position=40, valid_history_tokens=40, prefix_len=34,
        ephemeral_fork=True,
    )
    native._session_by_id = {"s": canonical_binding, work.session_id: work_binding}
    native._session_by_context = {101: canonical_binding, 202: work_binding}
    native.session_state = lambda row: SimpleNamespace(
        session_generation=7, prefix_len=34,
    )
    native.rebind_ephemeral_session_metadata = lambda row, current: replace(
        row, state_epoch_id=current.state_epoch_id, frame_id=current.frame_id,
        listener_state_version=current.listener_state_version,
        sequence_position=current.sequence_position,
    )
    native.copy_session_cache_from = lambda *_args, **_kwargs: pytest.fail(
        "same-owner canonical promotion must not copy the KV cache"
    )

    promoted = native.promote_ephemeral_session_binding(work, canonical)

    assert promoted.promoted_row.session_id == "s"
    assert promoted.promoted_row.native_context_id == 202
    assert promoted.retiring_row.session_id == work.session_id
    assert promoted.retiring_row.native_context_id == 101
    assert native._session_by_id["s"] is work_binding
    assert native._session_by_id[work.session_id] is canonical_binding
    assert (work_binding.slot, canonical_binding.slot) == (1, 0)
    assert work_binding.ephemeral_fork is False


def test_retiring_slot_is_reused_only_after_cuda_fence(memory_owner, monkeypatch):
    worker, _, native = memory_owner
    handle = _rebuild_to_catch_up(worker, cutoff=-1)
    event = Fence(False)
    worker._maintenance_jobs["s"].last_ready_event = event
    worker._cancel_maintenance(handle)

    assert worker._retiring_owners
    assert event.queries >= 1
    with pytest.raises(RuntimeError, match="WORK role already occupied"):
        worker._begin_memory_job("s", sources())

    event.ready = True
    worker._reap_retiring_owners()
    assert not worker._retiring_owners
    assert native.released.count(handle.session_id) == 1
    worker._begin_memory_job("s", sources())


def test_cancelled_append_future_and_close_keep_pending_native_lease(memory_owner):
    from duplex_kernel.listener_microbatch_worker import _Command

    worker, _, _ = memory_owner
    handle = _rebuild_to_catch_up(worker, cutoff=-1)
    pending = Fence(False)
    started, continue_native = __import__("threading").Event(), __import__("threading").Event()
    native = worker._runtime.native
    original = native.step_async

    def blocking_step(*args, **kwargs):
        started.set()
        assert continue_native.wait(timeout=2)
        return original(*args, **kwargs)

    native.step_async = blocking_step
    future = worker._submit_command("append_rebuild_chunk", handle, None)
    worker._commands.put(_Command("append_rebuild_chunk", (handle, None), future))
    owner = __import__("threading").Thread(target=worker._drain_commands)
    owner.start()
    assert started.wait(timeout=2)
    assert future.cancel()
    continue_native.set()
    owner.join(timeout=2)
    assert not owner.is_alive()
    assert not worker._maintenance_jobs
    assert not worker._work_owners
    assert worker._retiring_owners

    worker._close_session("s")
    assert worker._retiring_owners
    retired = next(iter(worker._retiring_owners.values()))
    pending.ready = True
    worker._reap_retiring_owners()
    assert retired.row.session_id not in worker._runtime.native._session_by_id


def test_rebuild_failure_does_not_invalidate_active_response(memory_owner, monkeypatch):
    worker, state, native = memory_owner
    active = begin(memory_owner, purpose=ResponseDecodePurpose.ACTIVE)
    handle = _rebuild_to_catch_up(worker, cutoff=-1)

    def fail_rebuild(manifest, *_args, **_kwargs):
        if manifest.rows[0].session_id.startswith("__memory__"):
            raise TimeoutError("rebuild deadline expired")
        raise AssertionError("ACTIVE append must not use the failed rebuild path")

    monkeypatch.setattr(native, "step_async", fail_rebuild)
    with pytest.raises(TimeoutError, match="deadline expired"):
        worker._append_rebuild_chunk(handle, None)

    assert worker._maintenance_states["s"] is State.FAILED
    assert state.native_context_id == 101
    assert append(memory_owner, active, 1).token_count == 1
