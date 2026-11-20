from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_branch_lifecycle_budgets import owner, begin, append
from duplex_kernel.listener_microbatch_worker import (
    KvOwnerRole, KvWorkKind, MemoryMaintenanceHandle, MemoryMaintenanceState as State,
    MemoryPrefillChunk, MemorySourceSnapshot, PrefixCacheImport, ResponseDecodePurpose,
)
from duplex_kernel.session import DuplexSession, MemoryRevision


@pytest.fixture
def memory_owner(owner):
    worker, state, native = owner
    native._session_by_id = {"s": SimpleNamespace(
        session_id="s", native_context_id=101, slot=0, prefix_len=34, ephemeral_fork=False,
    )}
    native._session_by_context = {101: native._session_by_id["s"]}
    native.rows = {101: worker._last_native_rows["s"]}

    def import_session(row, cache, attention, *, prefix_len, producer_stream):
        binding = SimpleNamespace(session_id=row.session_id, native_context_id=row.native_context_id,
                                  slot=len(native.rows), prefix_len=prefix_len, ephemeral_fork=False)
        native._session_by_id[row.session_id] = binding
        native._session_by_context[row.native_context_id] = binding
        native.rows[row.native_context_id] = row

    def session_state(row):
        actual = native.rows[row.native_context_id]
        assert (row.cache_position, row.frame_id, row.listener_state_version) == (
            actual.cache_position, actual.frame_id, actual.listener_state_version,
        )
        binding = native._session_by_id[row.session_id]
        assert binding.native_context_id == row.native_context_id
        return SimpleNamespace(prefix_len=binding.prefix_len)

    original_fork = native.fork_session_slot

    def fork(row, *, fork_session_id, fork_native_context_id):
        forked = original_fork(row, fork_session_id=fork_session_id,
                               fork_native_context_id=fork_native_context_id)
        import_session(forked, None, None, prefix_len=34, producer_stream=None)
        return forked

    native.fork_session_slot = fork
    worker._runtime.memory_token_encoder = lambda text: (1, 2)
    original_step = native.step_async

    def step(manifest, inputs, **kwargs):
        session_state(manifest.rows[0])
        output = original_step(manifest, inputs, **kwargs)
        row = output.rows[0]
        native.rows[row.native_context_id] = replace(
            row, cache_position=row.cache_position + row.logical_q_len,
            valid_history_tokens=row.valid_history_tokens + row.logical_q_len,
        )
        return output

    def release(row):
        if row.native_context_id in native.rows:
            session_state(row)
            native.rows.pop(row.native_context_id)
            native._session_by_id.pop(row.session_id)
            native._session_by_context.pop(row.native_context_id)
        native.released.append(row.session_id)

    def rebind(row, canonical):
        session_state(row)
        session_state(canonical)
        rebound = replace(row, frame_id=canonical.frame_id,
                          listener_state_version=canonical.listener_state_version,
                          sequence_position=canonical.sequence_position,
                          state_epoch_id=canonical.state_epoch_id)
        native.rows[row.native_context_id] = rebound
        return rebound

    native.import_session = import_session
    native.session_state = session_state
    native.step_async = step
    native.step_queued_audio_async = step
    native.release_session_async = release
    native.rebind_ephemeral_session_metadata = rebind
    return owner


def sources(*, through=-1, ranges=(), revisions=()):
    return MemorySourceSnapshot(
        prefix=PrefixCacheImport(object(), torch.ones((1, 34), dtype=torch.int64), 34),
        through_frame_id=through, source_ranges=ranges, source_revisions=revisions,
        eos_token_ids=(0,),
    )


def revision(*, ranges=(), revisions=()):
    return MemoryRevision("m1", None, ranges, revisions, "Una nota semantica.", None)


def finish_summary(worker, source=None):
    handle = worker._begin_memory_job("s", source or sources())
    worker._append_memory_input(handle, MemoryPrefillChunk(token_ids=(1, 2)))
    result = worker._advance_memory_decode(handle)
    assert result["done"]
    return handle


def ready_rebuild(worker, *, retained=None, memory=None):
    handle = worker._begin_canonical_rebuild("s", memory or revision(), retained or sources())
    while worker._maintenance_jobs["s"].pending_memory_tokens:
        worker._append_rebuild_chunk(handle, None)
    worker._append_rebuild_chunk(handle, None)
    worker._append_rebuild_chunk(handle, None)
    assert worker._maintenance_states["s"] is State.READY
    return handle


def test_lifecycle_and_handle_identity(memory_owner, monkeypatch):
    worker, _, _ = memory_owner
    observed = []
    original = worker._set_maintenance_state

    def record(sid, value):
        observed.append(value)
        original(sid, value)

    monkeypatch.setattr(worker, "_set_maintenance_state", record)
    assert worker._maintenance_states["s"] is State.IDLE
    summary = finish_summary(worker)
    assert isinstance(summary, MemoryMaintenanceHandle)
    with pytest.raises(FrozenInstanceError):
        summary.work_kind = KvWorkKind.REBUILD
    rebuild = ready_rebuild(worker)
    assert rebuild.work_kind is KvWorkKind.REBUILD
    assert rebuild != summary
    worker._promote_canonical_rebuild(rebuild)
    assert observed == [State.SUMMARIZING, State.BUILDING, State.CATCH_UP,
                        State.READY, State.SWAP, State.IDLE]
    assert not worker._work_owners


def test_summary_is_private_and_work_is_exclusive(memory_owner):
    worker, state, native = memory_owner
    active = begin(memory_owner)
    before = (state.cache_position, state.state_version, state.next_frame_id)
    summary = worker._begin_memory_job("s", sources())
    worker._append_memory_input(summary, MemoryPrefillChunk(token_ids=(9,)))
    assert (state.cache_position, state.state_version, state.next_frame_id) == before
    assert native.steps == 1
    assert worker._ephemeral_decode_branches[active.handle_id].token_count == 0
    with pytest.raises(RuntimeError, match="WORK role already occupied"):
        worker._begin_memory_job("s", sources())
    with pytest.raises(RuntimeError, match="WORK"):
        begin(memory_owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="busy")
    worker._cancel_maintenance(summary)
    assert worker._maintenance_states["s"] is State.IDLE
    assert not worker._work_owners
    worker._cancel_maintenance(summary)
    assert append(memory_owner, active, 1).token_count == 1


def test_rebuild_keeps_active_snapshot_and_retires_old_canonical(memory_owner):
    worker, state, native = memory_owner
    active = begin(memory_owner)
    active_row = worker._ephemeral_decode_branches[active.handle_id].row
    rebuild = ready_rebuild(worker)
    old_context = state.native_context_id
    worker._promote_canonical_rebuild(rebuild)
    assert state.native_context_id != old_context
    assert worker._kv_slot_roles["s"] is KvOwnerRole.CANONICAL
    assert any(value.row.native_context_id == old_context
               for value in worker._retiring_owners.values())
    assert KvOwnerRole.RETIRING in worker._kv_slot_roles.values()
    assert worker._ephemeral_decode_branches[active.handle_id].row == active_row
    assert active.source_cache_epoch_id == 0 and state.cache_epoch_id == 1
    assert append(memory_owner, active, 1).token_count == 1
    worker._reap_retiring_owners()
    assert not worker._retiring_owners
    assert "s" not in native.released


def test_failure_releases_work_and_does_not_touch_canonical(memory_owner, monkeypatch):
    worker, state, native = memory_owner
    summary = worker._begin_memory_job("s", sources())
    def fail(*args, **kwargs):
        raise ValueError("broken prefill")
    monkeypatch.setattr(native, "step_async", fail)
    with pytest.raises(ValueError, match="broken prefill"):
        worker._append_memory_input(summary, MemoryPrefillChunk(token_ids=(3,)))
    assert not worker._work_owners
    assert worker._maintenance_states["s"] is State.FAILED
    assert state.cache_position == 34 and state.native_context_id == 101
    worker._reap_retiring_owners()
    worker._begin_memory_job("s", sources())


def test_stale_handle_cannot_cancel_rebuild_or_new_job(memory_owner):
    worker, _, _ = memory_owner
    summary = finish_summary(worker)
    rebuild = ready_rebuild(worker)
    worker._cancel_maintenance(summary)
    assert worker._work_owners["s"].kind is KvWorkKind.REBUILD
    worker._cancel_maintenance(rebuild)
    current = worker._begin_memory_job("s", sources())
    worker._cancel_maintenance(summary)
    assert worker._maintenance_jobs["s"].handle == current


def test_changed_revisions_reject_swap_and_release_work(memory_owner):
    worker, state, _ = memory_owner
    state.next_frame_id = 2
    source = sources(through=1, ranges=((0, 1),), revisions=(2,))
    finish_summary(worker, source)
    rebuild = ready_rebuild(worker, memory=revision(ranges=((0, 1),), revisions=(2,)),
                            retained=sources(through=1))
    with pytest.raises(RuntimeError, match="revision"):
        worker._promote_canonical_rebuild(rebuild, expected_source_revisions=(3,))
    assert worker._maintenance_states["s"] is State.FAILED
    assert not worker._work_owners
    assert state.native_context_id == 101


def test_ready_cache_must_catch_up_new_live_frames(memory_owner):
    worker, state, native = memory_owner
    rebuild = ready_rebuild(worker)
    state.next_frame_id = state.sequence_position = state.state_version = 1
    row = replace(worker._last_native_rows["s"], frame_id=1,
                  sequence_position=1, listener_state_version=1)
    worker._last_native_rows["s"] = row
    native.rows[101] = row
    with pytest.raises(RuntimeError, match="catch-up"):
        worker._promote_canonical_rebuild(rebuild)
    assert worker._maintenance_states["s"] is State.CATCH_UP
    chunk = MemoryPrefillChunk(audio_soft=np.ones((2, 1536), dtype=np.float32),
                               first_frame_id=0, last_frame_id=0)
    worker._append_rebuild_chunk(rebuild, chunk)
    worker._append_rebuild_chunk(rebuild, None)
    worker._promote_canonical_rebuild(rebuild)
    assert state.next_frame_id == 1


def test_retirement_fence_prevents_fourth_slot(memory_owner, monkeypatch):
    worker, _, _ = memory_owner
    handle = ready_rebuild(worker)
    from duplex_kernel import listener_microbatch_worker as module
    monkeypatch.setattr(module, "_record_shadow_event", lambda stream: SimpleNamespace(query=lambda: False))
    worker._promote_canonical_rebuild(handle)
    with pytest.raises(RuntimeError, match="WORK role already occupied"):
        worker._begin_memory_job("s", sources())


@pytest.mark.parametrize("operation", ["close", "reset"])
def test_session_cleanup_invalidates_maintenance(memory_owner, operation):
    worker, state, _ = memory_owner
    handle = worker._begin_memory_job("s", sources())
    if operation == "close":
        worker._close_session("s")
        assert "s" not in worker._maintenance_states
    else:
        new_state = replace(state, generation_id=8, state_epoch_id="next-epoch", native_context_id=202)
        worker._reset_session("s", new_state, lambda *args: None)
        assert worker._maintenance_states["s"] is State.IDLE
    assert not worker._work_owners and not worker._maintenance_jobs
    worker._cancel_maintenance(handle)


def test_open_turn_archive_replays_immutable_cpu_soft_tokens(memory_owner):
    worker, state, _ = memory_owner
    state.next_frame_id = 4
    session = DuplexSession("s")
    original = np.ones((4, 1536), dtype=np.float32)
    for chunk_id, first in enumerate((0, 2)):
        session.reserve_audio_history_archive(
            pcm=None, soft_tokens_reference=original, encoder_profile="q2",
            first_frame_id=first, last_frame_id=first + 1, base_messages=[],
            state_epoch_id="epoch", turn_id=5, chunk_id=chunk_id, turn_open=True,
        )
    original[:] = 7
    chunks = session.prepare_memory_audio_chunks()
    assert len(chunks) == 4
    assert all(chunk.turn_open for chunk in chunks)
    assert [chunk.first_frame_id for chunk in chunks] == [0, 1, 2, 3]
    assert all(np.all(chunk.audio_soft == 1) for chunk in chunks)
    with pytest.raises(ValueError):
        chunks[0].audio_soft[0, 0] = 0
    rebuild = worker._begin_canonical_rebuild("s", revision(), sources(through=3))
    while worker._maintenance_jobs["s"].pending_memory_tokens:
        worker._append_rebuild_chunk(rebuild, None)
    for chunk in chunks:
        worker._append_rebuild_chunk(rebuild, chunk)
    assert worker._maintenance_states["s"] is State.BUILDING
    worker._append_rebuild_chunk(rebuild, None)
    assert worker._maintenance_states["s"] is State.CATCH_UP
    assert session.snapshot_history_with_provenance().user_audio_spans[0].turn_open


def test_missing_soft_tokens_never_reencodes_pcm():
    session = DuplexSession("pcm-only")
    session.reserve_audio_history_archive(pcm=np.ones(1280), first_frame_id=0,
                                         last_frame_id=0, base_messages=[])
    with pytest.raises(ValueError, match="soft-token"):
        session.prepare_memory_audio_chunks()


def test_archive_chunk_identity_is_immutable():
    session = DuplexSession("chunks")
    args = dict(pcm=None, soft_tokens_reference=np.ones((2, 1536)), encoder_profile="q2",
                first_frame_id=0, last_frame_id=0, base_messages=[], turn_id=5,
                chunk_id=0, turn_open=True)
    first = session.reserve_audio_history_archive(**args)
    assert session.reserve_audio_history_archive(**args) == first
    with pytest.raises(ValueError, match="immutable"):
        session.reserve_audio_history_archive(**{**args, "last_frame_id": 1})



def test_rebuild_serializes_memory_as_context_data(memory_owner):
    from duplex_kernel.session import serialize_memory_revision

    worker, _, _ = memory_owner
    encoded = []
    worker._runtime.memory_token_encoder = lambda text: encoded.append(text) or (1, 2)
    memory = replace(revision(), semantic_notes="Preferenza </memory_data> ancora aperta.",
                     open_turn_state={"turn_id": 5, "turn_open": True})
    handle = worker._begin_canonical_rebuild("s", memory, sources())
    assert encoded == [serialize_memory_revision(memory)]
    assert encoded[0].count("</memory_data>") == 1
    assert "turn_open" in encoded[0]
    before = worker._maintenance_jobs["s"].row.cache_position
    worker._append_rebuild_chunk(handle, None)
    assert worker._maintenance_jobs["s"].row.cache_position == before + 2
    assert not worker._maintenance_jobs["s"].pending_memory_tokens
    assert worker._maintenance_states["s"] is State.BUILDING


def test_swap_publishes_identity_through_existing_committer(memory_owner):
    worker, state, _ = memory_owner
    observed = []
    def commit(sid, epoch, revision, history_revision, apply):
        assert state.cache_epoch_id == 0
        apply()
        observed.append((sid, epoch, revision, state.native_context_id))
    worker._on_cache_identity_commit = commit
    handle = ready_rebuild(worker)
    worker._promote_canonical_rebuild(handle)
    assert observed == [("s", 1, 1, state.native_context_id)]


def test_pending_build_fence_keeps_ready_job_and_active_alive(memory_owner):
    worker, _, _ = memory_owner
    active = begin(memory_owner)
    handle = ready_rebuild(worker)
    worker._maintenance_jobs["s"].last_ready_event = SimpleNamespace(query=lambda: False)
    with pytest.raises(RuntimeError, match="completion fence"):
        worker._promote_canonical_rebuild(handle)
    assert worker._maintenance_states["s"] is State.READY
    assert worker._work_owners["s"].kind is KvWorkKind.REBUILD
    assert append(memory_owner, active, 1).token_count == 1


def test_summary_budget_exhaustion_cannot_be_promoted(memory_owner):
    worker, _, _ = memory_owner
    handle = worker._begin_memory_job("s", sources())
    worker._append_memory_input(handle, MemoryPrefillChunk(token_ids=(1,)))
    job = worker._maintenance_jobs["s"]
    job.generated_token_ids.extend([1] * worker._kv_budgets.generated_memory_max_tokens)
    with pytest.raises(RuntimeError, match="budget before EOS"):
        worker._advance_memory_decode(handle)
    assert worker._maintenance_states["s"] is State.FAILED
    assert not worker._work_owners


def test_summary_rejects_audio_beyond_frozen_cutoff(memory_owner):
    worker, state, _ = memory_owner
    state.next_frame_id = 3
    handle = worker._begin_memory_job("s", sources(through=0))
    with pytest.raises(ValueError, match="source cutoff"):
        worker._append_memory_input(handle, MemoryPrefillChunk(
            audio_soft=np.ones((2, 1536)), first_frame_id=1, last_frame_id=1,
        ))
    assert worker._maintenance_states["s"] is State.FAILED
    assert not worker._work_owners


def test_rebuild_rejects_missing_audio_frames(memory_owner):
    worker, state, _ = memory_owner
    state.next_frame_id = 2
    handle = worker._begin_canonical_rebuild("s", revision(), sources(through=1))
    while worker._maintenance_jobs["s"].pending_memory_tokens:
        worker._append_rebuild_chunk(handle, None)
    with pytest.raises(ValueError, match="consecutive"):
        worker._append_rebuild_chunk(handle, MemoryPrefillChunk(
            audio_soft=np.ones((2, 1536)), first_frame_id=1, last_frame_id=1,
        ))
    assert worker._maintenance_states["s"] is State.FAILED
    assert not worker._work_owners


def test_begin_import_failure_releases_work(memory_owner, monkeypatch):
    worker, state, native = memory_owner
    def fail(*args, **kwargs):
        raise RuntimeError("import failed")
    monkeypatch.setattr(native, "import_session", fail)
    with pytest.raises(RuntimeError, match="import failed"):
        worker._begin_memory_job("s", sources())
    assert not worker._work_owners and not worker._maintenance_jobs
    assert worker._maintenance_states["s"] is State.FAILED
    assert state.native_context_id == 101


def test_archive_cpu_bfloat16_is_replayable_and_immutable():
    session = DuplexSession("bf16")
    soft = torch.ones((2, 1536), dtype=torch.bfloat16)
    session.reserve_audio_history_archive(pcm=None, soft_tokens_reference=soft,
                                         first_frame_id=0, last_frame_id=0,
                                         base_messages=[], turn_open=True)
    soft.zero_()
    chunk, = session.prepare_memory_audio_chunks()
    assert chunk.turn_open and np.all(chunk.audio_soft == 1)
    with pytest.raises(ValueError):
        chunk.audio_soft.setflags(write=True)


def test_maintenance_commands_use_owner_queue(memory_owner, monkeypatch):
    from concurrent.futures import Future
    from duplex_kernel.listener_microbatch_worker import _Command

    worker, _, _ = memory_owner
    operations = []
    def submit(operation, *args):
        operations.append(operation)
        future = Future()
        future.set_result(worker._execute_command(_Command(operation, args)))
        return future
    monkeypatch.setattr(worker, "_submit_command", submit)
    summary = worker.begin_memory_job("s", sources()).result()
    worker.append_memory_input(summary, MemoryPrefillChunk(token_ids=(1,))).result()
    worker.advance_memory_decode(summary).result()
    rebuild = worker.begin_canonical_rebuild("s", revision(), sources()).result()
    while worker._maintenance_states["s"] is not State.READY:
        worker.append_rebuild_chunk(rebuild, None).result()
    worker.promote_canonical_rebuild(rebuild).result()
    worker.cancel_maintenance(rebuild).result()
    assert set(operations) == {
        "begin_memory_job", "append_memory_input", "advance_memory_decode",
        "begin_canonical_rebuild", "append_rebuild_chunk", "promote_canonical_rebuild",
        "cancel_maintenance",
    }



def test_summary_decodes_private_tokens_until_semantic_eos(memory_owner, monkeypatch):
    worker, state, native = memory_owner
    active = begin(memory_owner)
    before = (state.cache_position, state.next_frame_id, state.state_version)
    monkeypatch.setattr(worker._runtime.token_adapter, "logits_from_final_norm_hidden",
                        lambda hidden: torch.tensor([[0., 1.]]) if native.steps == 1
                        else torch.tensor([[1., 0.]]))
    summary = worker._begin_memory_job("s", sources())
    worker._append_memory_input(summary, MemoryPrefillChunk(token_ids=(4,)))
    assert worker._advance_memory_decode(summary) == {"token_ids": (1,), "done": False}
    assert worker._advance_memory_decode(summary) == {"token_ids": (1,), "done": True}
    assert (state.cache_position, state.next_frame_id, state.state_version) == before
    assert worker._ephemeral_decode_branches[active.handle_id].token_count == 0


def test_stale_parent_memory_invalidates_summary(memory_owner):
    worker, state, _ = memory_owner
    finish_summary(worker)
    state.semantic_context_revision = 9
    with pytest.raises(RuntimeError, match="parent memory revision"):
        worker._begin_canonical_rebuild("s", revision(), sources())
    assert worker._maintenance_states["s"] is State.FAILED
    assert not worker._work_owners


def test_invalid_rebuild_memory_encoder_releases_summary_work(memory_owner):
    worker, _, _ = memory_owner
    finish_summary(worker)
    worker._runtime.memory_token_encoder = lambda text: ()
    with pytest.raises(ValueError, match="nonempty token IDs"):
        worker._begin_canonical_rebuild("s", revision(), sources())
    assert worker._maintenance_states["s"] is State.FAILED
    assert not worker._work_owners


def test_prefill_chunk_freezes_queued_input():
    soft = np.ones((2, 1536), dtype=np.float32)
    ids = [1, 2]
    audio_chunk = MemoryPrefillChunk(audio_soft=soft, first_frame_id=0, last_frame_id=0)
    text_chunk = MemoryPrefillChunk(token_ids=ids)
    soft[:] = 7
    ids.append(3)
    assert np.all(audio_chunk.audio_soft == 1)
    assert text_chunk.token_ids == (1, 2)


def test_rebuild_failure_preserves_canonical_and_active(memory_owner, monkeypatch):
    worker, state, native = memory_owner
    active = begin(memory_owner)
    rebuild = worker._begin_canonical_rebuild("s", revision(), sources())
    original = native.step_async
    def fail(manifest, *args, **kwargs):
        if manifest.rows[0].session_id.startswith("__memory__"):
            raise RuntimeError("private rebuild failure")
        return original(manifest, *args, **kwargs)
    monkeypatch.setattr(native, "step_async", fail)
    with pytest.raises(RuntimeError, match="private rebuild failure"):
        worker._append_rebuild_chunk(rebuild, None)
    assert not worker._work_owners
    assert state.native_context_id == 101 and state.cache_position == 34
    assert append(memory_owner, active, 1).token_count == 1
