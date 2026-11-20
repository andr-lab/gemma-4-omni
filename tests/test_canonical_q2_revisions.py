from __future__ import annotations

import ast
import inspect
import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duplex_kernel import server as server_module
from duplex_kernel.gemma_native_control6_batch_policy import (
    DecisionRowSource, GemmaNativeControl6BatchPolicy, PolicyPreHeadContext,
)
from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.listener_microbatch_worker import (
    KvOwnerRole, KvWorkKind, ResponseDecodePurpose, _Command,
)
from duplex_kernel.microbatch_contract import ListenerBatchManifest, ListenerFrame, snapshot_listener_state
from duplex_kernel.native_listener_microbatch import NativeBatchOutput
from duplex_kernel.server_audio_snapshot_policy import _native_control6_decision_has_valid_input
from duplex_kernel.server_realtime_scheduler import _evaluate_listener_silence_skip
from duplex_kernel.server_response_control import (
    _native_control6_policy_hidden_for_tick, _request_policy_response_revision,
)
from duplex_kernel.server_unified_templates import _unified_listener_frame_contract
from duplex_kernel.turn_policy import ControlAction, ControlState, PolicyDecision, PolicyInput
from training.duplex_qlora.native_control6 import ACTIONS, HIDDEN_WIDTH, NativeControl6LoRA
from test_branch_lifecycle_budgets import owner, begin, append


class RecordingHead(NativeControl6LoRA):
    def __init__(self):
        super().__init__(torch.zeros((len(ACTIONS), HIDDEN_WIDTH)))
        self.seen = []

    def forward(self, hidden, *, phase_t=None, **kwargs):
        self.seen.append(hidden.detach().clone())
        logits = torch.full((hidden.shape[0], len(ACTIONS)), -20.0)
        logits[:, ACTIONS.index("YIELD")] = 20.0
        return logits


def policy_state(**overrides):
    values = dict(
        session_id="s", generation_id=7, native_listener_generation=7, frame_id=0,
        speaker_active=True, user_pcm=np.zeros(1280, dtype=np.float32),
        native_control_hidden=torch.ones((1, HIDDEN_WIDTH)),
        native_control_hidden_frame_id=0, native_control_hidden_at_ns=time.monotonic_ns(),
        legal_actions=("CONTINUE", "YIELD"), native_control_phase_t=(0.0, 0.0, 0.0, 0.0),
    )
    values.update(overrides)
    return PolicyInput(**values)


@pytest.mark.parametrize("queue_open", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("voiced", [False, True])
def test_every_ordinary_frame_retains_canonical_q2(queue_open, pending, voiced):
    pcm = np.full(1280, 0.1 if voiced else 0.0, dtype=np.float32)
    assert _unified_listener_frame_contract(
        pcm, assistant_audio_queue_open=queue_open, user_input_pending=pending,
        voice_active=voiced,
    ) == (2, None, False)


def test_silence_never_skips_the_realtime_listener():
    result = _evaluate_listener_silence_skip(
        SimpleNamespace(), SimpleNamespace(pcm=np.zeros(1280), frame_id=0), None,
        SimpleNamespace(voice_run_active=False), False, True,
        shared_listener_microbatch_enabled=False, listener_generation=7, pending_plan=False,
    )
    assert not result.allow_silence_skip
    assert not result.transition_lock_acquired


def test_zero_pcm_preserves_genuine_hidden_and_scores_the_head():
    adapter = RecordingHead()
    policy = GemmaNativeControl6Policy(adapter, checkpoint_path="fixture")
    state = policy_state(native_control_no_input_tick=True)
    hidden, no_input = _native_control6_policy_hidden_for_tick(
        user_pcm=state.user_pcm, speaker_active=True, listener_hidden=state.native_control_hidden,
    )
    assert hidden is state.native_control_hidden
    assert not no_input
    decision = policy.step(state)
    assert decision.action is ControlAction.YIELD
    assert len(adapter.seen) == 1
    assert torch.equal(adapter.seen[0], hidden)
    assert decision.listener_frame_id == state.frame_id


def test_zero_pcm_without_hidden_cannot_invent_continue():
    policy = GemmaNativeControl6Policy(RecordingHead(), checkpoint_path="fixture")
    with pytest.raises(RuntimeError, match="fresh.*hidden"):
        policy.step(policy_state(native_control_hidden=None, native_control_no_input_tick=True))
    assert not _native_control6_decision_has_valid_input(
        fresh_hidden=False, no_input_tick=True, speaker_active=True,
        action="CONTINUE", legal_actions=("CONTINUE",),
    )


def test_precomputed_logits_are_required_even_for_a_legacy_zero_pcm_flag():
    policy = GemmaNativeControl6Policy(RecordingHead(), checkpoint_path="fixture")
    state = policy_state(native_control_no_input_tick=True)
    with pytest.raises(ValueError, match="logits"):
        policy.step_with_logits(state, None, tick_id=("s", 7, 0))
    logits = torch.full((len(ACTIONS),), -20.0)
    logits[ACTIONS.index("YIELD")] = 20.0
    decision = policy.step_with_logits(state, logits, tick_id=("s", 7, 0))
    assert decision.action is ControlAction.YIELD


def test_mixed_batch_scores_zero_pcm_rows_with_current_native_hidden(owner):
    worker, state, _ = owner
    row = worker._last_native_rows["s"]
    manifest = ListenerBatchManifest(
        manifest_id="current", runtime_signature=state.runtime_signature,
        stream_identity="cpu:fixture", stream_priority=-1, dispatch_at_ns=1,
        max_coalesce_wait_ns=0, dispatch_reason="coalesce_deadline", rows=(row,),
    )
    native_hidden = torch.ones((1, 2, HIDDEN_WIDTH), dtype=torch.bfloat16)
    output = NativeBatchOutput(
        manifest_id="current", generation=1, rows=(row,), tensor=native_hidden,
        completion_event=None,
    )
    adapter = RecordingHead()
    policy = GemmaNativeControl6Policy(adapter, checkpoint_path="fixture")
    batch_policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="fixture")
    context = PolicyPreHeadContext(
        policy=policy, policy_input=policy_state(
            native_control_hidden=None, native_control_hidden_frame_id=None,
            native_control_hidden_at_ns=None, native_control_no_input_tick=True,
        ), tick_id=("s", 7, 0), captured_at_ns=time.monotonic_ns(), listener_generation_id=7,
    )
    decisions = batch_policy.step_native_output_views(
        (DecisionRowSource(native_output=output, manifest=manifest, row_index=0, decision_row=row),),
        (context,),
    )
    assert decisions[0].action is ControlAction.YIELD
    assert batch_policy.forward_call_count == 1
    assert torch.equal(adapter.seen[0], native_hidden[:, 0, :])
    assert decisions[0].diagnostics["native_output_provenance"]["used_for_policy"] is True


def test_worker_does_not_downgrade_q2_when_a_legacy_queue_is_open(owner):
    worker, state, _ = owner
    enqueued = []
    worker._coordinator.ready_depth = 1
    worker._coordinator.enqueue = enqueued.append
    worker._assistant_audio_queue_open.add("s")
    for frame_id in range(4):
        frame = ListenerFrame(
            session_id="s", generation_id=7, state_epoch_id=state.state_epoch_id,
            frame_id=frame_id, ready_at_ns=frame_id * 80_000_000,
            pcm=np.full(1280, 0.1 if frame_id % 2 else 0.0, dtype=np.float32),
        )
        worker._execute_command(_Command("submit_tick", ((frame,),), Future()))
        assert enqueued[-1].logical_q_len == 2
        assert enqueued[-1].ephemeral_reason is None
        assert not enqueued[-1].queue_audio_for_next_turn
        worker._tick_future = None
        worker._outstanding_frames["s"] = 0
        worker._outstanding_logical_tokens["s"] = 0
        state.next_frame_id += 1
        state.state_version += 1
        state.sequence_position += 1
        state.cache_position += 2
        state.valid_history_tokens += 2
    assert [frame.frame_id for frame in enqueued] == [0, 1, 2, 3]
    assert [frame.ready_at_ns for frame in enqueued] == [0, 80_000_000, 160_000_000, 240_000_000]


def revision_session():
    requests = []
    session = SimpleNamespace(
        session_id="s", response_id="response", session_generation_id=7,
        active_response=True, control_state=ControlState.SPEAKING,
        _generation_lock=threading.RLock(),
    )
    branch = dict(
        response_id="response", session_generation_id=7, revisioned_live_say=True,
        spoken_prefix_lock=threading.RLock(),
        revision_coordinator=SimpleNamespace(request_revision=lambda frame: requests.append(frame) or True),
    )
    return session, branch, requests


@pytest.mark.parametrize("action", list(ControlAction))
def test_only_an_explicit_replan_requests_a_revision(action):
    session, branch, requests = revision_session()
    decision = PolicyDecision(action=action, frame_id=12, generation_id=7, session="s")
    accepted = _request_policy_response_revision(session, decision, branch)
    assert accepted is (action is ControlAction.REPLAN)
    assert requests == ([12] if action is ControlAction.REPLAN else [])
    assert session.session_generation_id == 7
    assert session.active_response
    assert session.control_state is ControlState.SPEAKING


@pytest.mark.parametrize("change", ["generation", "session", "response", "cancelled", "coordinator"])
def test_replan_cannot_revise_a_stale_or_closed_response(change):
    session, branch, requests = revision_session()
    decision = PolicyDecision(action=ControlAction.REPLAN, frame_id=12, generation_id=7, session="s")
    if change == "generation":
        decision = replace(decision, generation_id=6)
    elif change == "session":
        decision = replace(decision, session="other")
    elif change == "response":
        branch["response_id"] = "old"
    elif change == "cancelled":
        branch["playback_cancelled"] = True
    else:
        branch.pop("revision_coordinator")
    assert not _request_policy_response_revision(session, decision, branch)
    assert requests == []


def test_vad_voiced_branch_has_no_revision_side_effect():
    tree = ast.parse(inspect.getsource(server_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "input_frame_voiced" in ast.unparse(node.test):
            calls = [call.func.attr for call in ast.walk(node)
                     if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)]
            assert "request_revision" not in calls
            assert "begin_response_decode" not in calls


@pytest.mark.parametrize("action, expected, role", [
    (ControlAction.SPEAK, ResponseDecodePurpose.ACTIVE, KvOwnerRole.ACTIVE),
    (ControlAction.REPLAN, ResponseDecodePurpose.CANDIDATE, KvOwnerRole.WORK),
])
def test_control6_response_action_opens_only_its_private_role(owner, action, expected, role):
    worker, state, _ = owner
    before = snapshot_listener_state(state)
    purpose = ResponseDecodePurpose.for_action(action)
    assert purpose is expected
    handle = begin(owner, purpose=purpose)
    assert handle.kv_owner_role is role
    assert handle.purpose == expected.value
    if action is ControlAction.REPLAN:
        assert worker._work_owners["s"].kind is KvWorkKind.REPLAN
    worker._append_private_transition_tokens(handle, (201, 202), expected_cache_position=34)
    append(owner, handle, 1)
    assert snapshot_listener_state(state) == before
    assert worker._last_native_rows["s"].cache_position == 34


def test_backchannel_has_its_own_bounded_branch_and_leaves_canonical_untouched(owner):
    worker, state, _ = owner
    before = snapshot_listener_state(state)
    with pytest.raises(ValueError):
        ResponseDecodePurpose.for_action(ControlAction.BACKCHANNEL)
    handle = worker._begin_ephemeral_decode("s", expected_generation_id=7, expected_cache_position=34)
    assert handle.purpose == "backchannel"
    assert handle.max_tokens == 2
    worker._append_private_transition_tokens(handle, (201, 202), expected_cache_position=34)
    worker._append_private_ids(handle, (10, 11), expected_cache_position=36, is_transition=False)
    with pytest.raises(RuntimeError):
        worker._append_private_ids(handle, (12,), expected_cache_position=38, is_transition=False)
    assert snapshot_listener_state(state) == before
    assert not worker._work_owners
