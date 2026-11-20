from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel import listener_microbatch_worker as worker_module
from duplex_kernel.listener_microbatch_worker import (
    ASSISTANT_RESPONSE_MAX_TOKENS,
    BranchTokenBudget,
    DESIRED_SWAP_THRESHOLD_TOKENS,
    GENERATED_MEMORY_MAX_TOKENS,
    INITIAL_PREFIX_MAX_TOKENS,
    KvCacheBudgets,
    KvOwnerRole,
    KvWorkKind,
    LISTENER_MAX_SEQ_LEN,
    ListenerMicrobatchRuntime,
    ListenerMicrobatchWorker,
    ListenerWatermark,
    MEMORY_PREPARATION_TRIGGER_TOKENS,
    NEW_CACHE_TARGET_TOKENS,
    OPERATIONAL_CACHE_BUDGET_TOKENS,
    RECENT_AUDIO_REFERENCE_FRAMES,
    RecoverableSessionCapacityError,
    ResponseDecodePurpose,
)
from duplex_kernel.microbatch_contract import (
    BatchRowDescriptor, ListenerSessionState, RuntimeSignature, snapshot_listener_state,
)


class NativeSlots:
    def __init__(self):
        self.released = []
        self.steps = 0

    def ensure_capacity(self, count):
        return count

    def fork_session_slot(self, row, *, fork_session_id, fork_native_context_id):
        return replace(row, session_id=fork_session_id, native_context_id=fork_native_context_id)

    def release_session_async(self, row):
        self.released.append(row.session_id)

    def step_async(self, manifest, inputs, **kwargs):
        self.steps += 1
        return SimpleNamespace(
            tensor=torch.zeros((1, inputs.shape[1], 1536)), rows=manifest.rows,
            generation=self.steps, manifest_id=manifest.manifest_id,
        )

    def verify_output_health(self, output):
        return (SimpleNamespace(
            row=output.rows[0], generation=output.generation, checked=True, nonfinite=False,
        ),)


class TokenAdapter:
    def prepare(self, ids, *, logical_q_lens):
        q = max(2, len(ids) + len(ids) % 2)
        return SimpleNamespace(
            logical_q_lens=(logical_q_lens,), inputs_embeds=torch.zeros((1, q, 1536)),
            per_layer_inputs=torch.zeros((1, q, 35, 256)),
        )

    def logits_from_final_norm_hidden(self, hidden):
        return hidden.mean(-1)


@pytest.fixture
def owner(monkeypatch):
    signature = RuntimeSignature(
        model_id="fixture", model_config_sha256="a" * 64, checkpoint_sha256="b" * 64,
        topology_sha256="c" * 64, native_library_sha256="d" * 64,
        native_abi_version=3, batch_api_version=2, attention_backend_id=3,
        rmsnorm_backend_id=2, max_seq_len=16384, stream_q_len=2,
    )
    state = ListenerSessionState(
        session_id="s", generation_id=7, state_epoch_id="session-epoch",
        runtime_signature=signature, cache_position=34, valid_history_tokens=34,
        native_context_id=101, applied_history_revision=0,
    )
    native = NativeSlots()
    runtime = ListenerMicrobatchRuntime(
        listener_stream=SimpleNamespace(device=torch.device("cpu"), wait_event=lambda event: None),
        frontend=SimpleNamespace(close_session=lambda sid: None),
        native=native, policy=None, policy_inputs=None, token_adapter=TokenAdapter(),
    )
    worker = ListenerMicrobatchWorker(
        None, runtime_factory=lambda: runtime, on_result=lambda *args: None,
        on_error=lambda *args: None,
    )
    worker._runtime = runtime

    def rebase(sid, cursor):
        state.cache_position = state.valid_history_tokens = cursor

    def close(sid):
        state.closed = True
        return SimpleNamespace(state=state)

    worker._coordinator = SimpleNamespace(
        capability=SimpleNamespace(max_seq_len=16384, stream_identity="cpu:fixture", stream_priority=-1), runtime_signature=signature,
        session_state=lambda sid: snapshot_listener_state(state),
        register_session=lambda value: None, rebase_cache_position=rebase, close_session=close,
    )
    worker._register_session(state, lambda *args: None)
    worker._imported_sessions.add("s")
    worker._last_native_rows["s"] = BatchRowDescriptor(
        session_id="s", session_generation=7, state_epoch_id="session-epoch", frame_id=0,
        listener_state_version=0, sequence_position=0, cache_position=34,
        valid_history_tokens=34, native_context_id=101, input_frame_sha256="a" * 64,
        reference_pcm_sha256=None, frame_ready_at_ns=0, reference_frame_id=None,
    )
    monkeypatch.setattr(worker_module, "_record_shadow_event", lambda stream: None)
    return worker, state, native


def begin(owner, *, purpose=ResponseDecodePurpose.ACTIVE, revision="r0", max_tokens=8,
          response_id="response", response_generation_id=1, **kwargs):
    worker, state, _ = owner
    source = snapshot_listener_state(state)
    watermark = ListenerWatermark(
        session_id="s", generation_id=source.generation_id, state_epoch_id=source.state_epoch_id,
        frame_id=source.next_frame_id, state_version=source.state_version,
        sequence_position=source.sequence_position, cache_position=source.cache_position,
        cache_epoch_id=source.cache_epoch_id, applied_history_revision=source.applied_history_revision,
    )
    key = ("s", response_id, response_generation_id, purpose.value)
    worker._latest_response_revisions[key] = revision
    return worker._begin_response_decode(
        "s", expected_generation_id=7, response_id=response_id,
        response_generation_id=response_generation_id, revision_id=revision, purpose=purpose,
        source_watermark=watermark, max_tokens=max_tokens, prompt_token_budget=6,
        revision_key=key, **kwargs,
    )


def append(owner, handle, count, *, generated=True):
    worker, _, _ = owner
    row = worker._ephemeral_decode_branches[handle.handle_id].row
    return worker._append_response_decode_ids(
        handle, tuple(range(100, 100 + count)), expected_cache_position=row.cache_position,
        generated=generated,
    )


def test_service_defaults_preserve_native_abi():
    assert (
        OPERATIONAL_CACHE_BUDGET_TOKENS, MEMORY_PREPARATION_TRIGGER_TOKENS,
        DESIRED_SWAP_THRESHOLD_TOKENS, NEW_CACHE_TARGET_TOKENS, INITIAL_PREFIX_MAX_TOKENS,
        GENERATED_MEMORY_MAX_TOKENS, RECENT_AUDIO_REFERENCE_FRAMES,
        ASSISTANT_RESPONSE_MAX_TOKENS, LISTENER_MAX_SEQ_LEN,
    ) == (4096, 2048, 3072, 2048, 512, 256, 200, 768, 16384)
    assert KvCacheBudgets().operational_cache_budget_tokens == 4096
    with pytest.raises(ValueError):
        KvCacheBudgets(operational_cache_budget_tokens=16385)


def test_roles_and_work_exclusion(owner):
    worker, _, _ = owner
    active = begin(owner)
    candidate = begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="replan")
    assert worker._kv_slot_roles["s"] is KvOwnerRole.CANONICAL
    assert active.kv_owner_role is KvOwnerRole.ACTIVE
    assert candidate.kv_owner_role is KvOwnerRole.WORK
    assert active.session_id == "s"
    assert active.response_generation_id == 1
    assert active.kv_owner_epoch != candidate.kv_owner_epoch
    assert active.memory_revision_id is None
    assert active.source_history_revision == 0
    for kind in (KvWorkKind.SUMMARY, KvWorkKind.REBUILD):
        with pytest.raises(RuntimeError, match="WORK"):
            worker._reserve_work_owner("s", kind, BranchTokenBudget(34, 0, 0, 0))
    worker._end_response_decode(candidate)
    summary = worker._reserve_work_owner(
        "s", KvWorkKind.SUMMARY, BranchTokenBudget(34, 0, 0, 256),
    )
    assert summary.kv_owner_role is KvOwnerRole.WORK
    with pytest.raises(RuntimeError, match="WORK"):
        begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="replan2")
    worker._release_work_owner(summary)
    rebuild = worker._reserve_work_owner("s", KvWorkKind.REBUILD, BranchTokenBudget(34, 0, 0, 0))
    assert rebuild.kind is KvWorkKind.REBUILD


def test_canonical_swap_preserves_active_and_work_append(owner):
    worker, state, native = owner
    active = begin(owner)
    candidate = begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="replan")
    worker._refresh_session_prefix("s", 0, 0, 24, lambda *args: None)
    state.cache_epoch_id += 1
    state.applied_history_revision = 9
    state.semantic_context_revision = 2
    assert native.released == ["s"]
    assert append(owner, active, 1).token_count == 1
    assert append(owner, candidate, 1).token_count == 1
    assert active.source_cache_epoch_id == candidate.source_cache_epoch_id == 0
    promoted = worker._promote_response_decode(
        candidate, expected_active_handle=active, expected_active_revision_id="r0",
    )
    assert promoted.kv_owner_role is KvOwnerRole.ACTIVE
    assert promoted.kv_owner_epoch == candidate.kv_owner_epoch
    worker._end_response_decode(candidate)
    assert append(owner, promoted, 1).token_count == 2


def test_close_invalidates_all_roles_and_response_budget(owner):
    worker, state, native = owner
    active = begin(owner)
    candidate = begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="replan")
    worker._close_session("s")
    assert state.closed
    assert set(native.released) == {"s", active.fork_session_id, candidate.fork_session_id}
    assert not worker._kv_slot_roles
    assert not worker._work_owners
    assert not worker._response_token_budgets
    with pytest.raises(RuntimeError):
        worker._append_private_ids(active, (1,), expected_cache_position=34, is_transition=False)


def test_close_releases_summary_reservation(owner):
    worker, _, _ = owner
    worker._reserve_work_owner("s", KvWorkKind.SUMMARY, BranchTokenBudget(34, 0, 0, 256))
    worker._close_session("s")
    assert not worker._work_owners
    assert not worker._kv_slot_roles


def test_open_checks_complete_cache_budget_before_fork(owner):
    worker, state, native = owner
    state.cache_position = state.valid_history_tokens = 3322
    worker._last_native_rows["s"] = replace(
        worker._last_native_rows["s"], cache_position=3322, valid_history_tokens=3322,
    )
    active = begin(owner, max_tokens=768)
    assert active.max_tokens == 768
    worker._end_response_decode(active)
    state.cache_position = state.valid_history_tokens = 3323
    worker._last_native_rows["s"] = replace(
        worker._last_native_rows["s"], cache_position=3323, valid_history_tokens=3323,
    )
    with pytest.raises(RecoverableSessionCapacityError):
        begin(owner, revision="r1", max_tokens=768)
    assert not worker._ephemeral_decode_branches
    assert native.steps == 0


def test_transition_and_frozen_prefix_share_admission_budget(owner):
    worker, state, _ = owner
    state.cache_position = state.valid_history_tokens = 3300
    worker._last_native_rows["s"] = replace(
        worker._last_native_rows["s"], cache_position=3300, valid_history_tokens=3300,
    )
    with pytest.raises(RecoverableSessionCapacityError):
        begin(owner, max_tokens=768, transition_tokens=6, frozen_assistant_prefix_tokens=23)
    assert not worker._ephemeral_decode_branches


def test_work_budget_limits_summary_and_rebuild(owner):
    worker, _, _ = owner
    with pytest.raises(ValueError, match="memory"):
        worker._reserve_work_owner("s", KvWorkKind.SUMMARY, BranchTokenBudget(34, 0, 0, 257))
    with pytest.raises(ValueError, match="target"):
        worker._reserve_work_owner("s", KvWorkKind.REBUILD, BranchTokenBudget(2049, 0, 0, 0))
    assert not worker._work_owners


def test_replan_never_renews_original_response_budget(owner):
    worker, _, native = owner
    active = begin(owner, max_tokens=8)
    append(owner, active, 5)
    candidate = begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="replan", max_tokens=8)
    assert candidate.max_tokens == 3
    append(owner, candidate, 2, generated=False)
    append(owner, candidate, 2)
    worker._end_response_decode(candidate)
    retry = begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="retry", max_tokens=8)
    assert retry.max_tokens == 1
    append(owner, retry, 1)
    steps = native.steps
    with pytest.raises(RuntimeError, match="response.*budget"):
        append(owner, active, 1)
    assert native.steps == steps
    worker._end_response_decode(retry)
    with pytest.raises(RuntimeError, match="response.*budget"):
        begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="exhausted")


def test_service_budget_is_configurable_and_enforced_on_append(owner):
    worker, _, native = owner
    worker._runtime.kv_budgets = KvCacheBudgets(assistant_response_max_tokens=3)
    active = begin(owner, max_tokens=8)
    assert active.max_tokens == 3
    append(owner, active, 3)
    with pytest.raises(RuntimeError, match="response.*budget"):
        append(owner, active, 1)
    assert native.steps == 1


def test_stale_work_release_cannot_free_new_owner(owner):
    worker, _, _ = owner
    first = worker._reserve_work_owner("s", KvWorkKind.SUMMARY, BranchTokenBudget(34, 0, 0, 256))
    worker._release_work_owner(first)
    second = worker._reserve_work_owner("s", KvWorkKind.REBUILD, BranchTokenBudget(34, 0, 0, 0))
    worker._release_work_owner(first)
    assert worker._work_owners["s"] == second


def test_response_transition_api_cannot_bypass_prompt_reservation(owner):
    worker, _, native = owner
    handle = begin(owner)
    worker._append_private_transition_tokens(handle, (1, 2, 3, 4), expected_cache_position=34)
    with pytest.raises(RecoverableSessionCapacityError):
        append(owner, handle, 3, generated=False)
    assert native.steps == 1


def test_canonical_replacement_rejects_oversize_before_retirement(owner):
    worker, _, native = owner
    active = begin(owner)
    with pytest.raises(ValueError, match="target"):
        worker._refresh_session_prefix("s", 0, 0, 2049, lambda *args: None)
    assert native.released == []
    assert append(owner, active, 1).token_count == 1


def test_custom_service_thresholds_configure_runtime_defaults():
    budgets = KvCacheBudgets(
        operational_cache_budget_tokens=8192, memory_preparation_trigger_tokens=4096,
        desired_swap_threshold_tokens=6144, new_cache_target_tokens=3072,
    )
    runtime = ListenerMicrobatchRuntime(
        listener_stream=None, frontend=None, native=None, policy=None, policy_inputs=None,
        kv_budgets=budgets,
    )
    assert runtime.compaction_trigger_tokens == 4096
    assert runtime.compaction_target_tokens == 3072


def test_semantic_staging_and_replan_share_work_reservation(owner):
    worker, _, _ = owner
    pending = set()

    def stage(request, **kwargs):
        pending.add(request.session_id)
        return True

    worker._shadow_worker = SimpleNamespace(
        stage=stage, owns_session_work=lambda sid: sid in pending,
    )
    request = SimpleNamespace(session_id="s", cache_position=100, prefix_len=34)
    assert worker.stage_semantic_prefix(request, listener_generation_id=7, sequence_position=0)
    with pytest.raises(RuntimeError, match="WORK"):
        begin(owner, purpose=ResponseDecodePurpose.CANDIDATE, revision="busy")
    worker._refresh_session_prefix("s", 0, 0, 24, lambda *args: None)
    assert worker._work_owners["s"].kind is KvWorkKind.REBUILD
    pending.clear()
    worker._reap_shadow_work_owner("s")
    assert not worker._work_owners


def test_closed_generation_rejects_private_append_and_releases_its_slot(owner):
    worker, state, native = owner
    active = begin(owner)
    state.generation_id += 1
    with pytest.raises(RuntimeError, match="generation"):
        append(owner, active, 1)
    assert active.fork_session_id in native.released
    assert active.handle_id not in worker._ephemeral_decode_branches
