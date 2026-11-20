from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.server_diagnostics import (
    _capture_native_control6_policy_error,
    _native_control6_event_annotations,
    _native_control6_policy_evaluation_state,
    _native_control6_policy_error_diagnostics,
    _record_native_control6_policy_invalidation,
    _validate_native_control6_worker_decision_provenance,
)
from duplex_kernel.turn_policy import ControlAction, PolicyInput
from training.duplex_qlora.native_control6 import ACTIONS, HIDDEN_WIDTH, NativeControl6LoRA


def test_first_provenance_error_invalidates_decision_and_stays_primary() -> None:
    decision = SimpleNamespace(action=ControlAction.CONTINUE)
    decision.session = "canonical-session"
    decision.generation_id = 4
    decision.frame_id = 91
    decision.listener_frame_id = 91
    provenance = {
        "canonical_session_id": "canonical-session",
        "listener_generation_id": 8,
        "turn_generation_id": 4,
        "state_epoch_id": "epoch-a",
        "frame_id": 91,
        "listener_state_version": 32,
    }

    with pytest.raises(RuntimeError, match="provenance differs") as raised:
        _validate_native_control6_worker_decision_provenance(
            decision,
            provenance,
            binding_available=True,
            canonical_session_id="canonical-session",
            expected_listener_generation_id=7,
            expected_turn_generation_id=4,
            expected_state_epoch_id="epoch-a",
            expected_frame_id=91,
            expected_listener_state_version=32,
        )
    original = raised.value

    decision, native_policy_error = _capture_native_control6_policy_error(
        decision, original
    )

    assert decision is None
    assert native_policy_error is original
    assert "provenance differs" in str(native_policy_error)
    assert "lacks a fresh listener hidden" not in str(native_policy_error)


def test_policy_evaluation_state_reports_scored_exact_zero_and_unavailable() -> None:
    adapter = NativeControl6LoRA(torch.zeros((len(ACTIONS), HIDDEN_WIDTH)))
    forward_calls: list[torch.Tensor] = []
    adapter.register_forward_hook(lambda _module, _inputs, output: forward_calls.append(output))
    policy = GemmaNativeControl6Policy(adapter, checkpoint_path="cpu-localization-test.pt")
    scored = policy.step(PolicyInput(
        session_id="policy-state-scored",
        frame_id=12,
        user_pcm=None,
        speaker_active=False,
        generation_id=4,
        native_listener_generation=4,
        native_control_hidden=torch.zeros((1, HIDDEN_WIDTH)),
        native_control_hidden_frame_id=12,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert len(forward_calls) == 1
    skipped = SimpleNamespace(
        action=ControlAction.CONTINUE,
        diagnostics={
            "policy_head_skipped": True,
            "decision_view_unavailable_reason": "stable Q2 context unavailable",
        },
    )

    assert _native_control6_policy_evaluation_state(
        decision=scored, head_scored=len(forward_calls) == 1
    ) == "scored"
    scored.diagnostics.update({
        "policy_evaluation_state": "scored",
        "native_control6_head_scored": True,
    })
    decision_event_fields = _native_control6_event_annotations(
        scored, head_scored=True, accepted=True
    )
    assert decision_event_fields["policy_evaluation_state"] == "scored"
    assert decision_event_fields["native_control6_head_scored"] is True

    zero_forward_calls: list[torch.Tensor] = []
    idle_adapter = NativeControl6LoRA(torch.zeros((len(ACTIONS), HIDDEN_WIDTH)))
    idle_adapter.register_forward_hook(
        lambda _module, _inputs, output: zero_forward_calls.append(output)
    )
    idle_policy = GemmaNativeControl6Policy(
        idle_adapter, checkpoint_path="cpu-localization-zero-test.pt"
    )
    exact_zero_continue = idle_policy.step(PolicyInput(
        session_id="policy-state-zero",
        frame_id=13,
        user_pcm=None,
        speaker_active=True,
        generation_id=4,
        native_listener_generation=4,
        legal_actions=("CONTINUE", "YIELD"),
        native_control_no_input_tick=True,
    ))
    assert exact_zero_continue.action is ControlAction.CONTINUE
    assert not zero_forward_calls
    assert _native_control6_policy_evaluation_state(
        decision=exact_zero_continue,
        head_scored=False,
        no_input_tick=True,
    ) == "exact_zero_continue"
    assert _native_control6_policy_evaluation_state(
        decision=skipped, head_scored=False
    ) == "unavailable"
    assert _native_control6_policy_evaluation_state(
        decision=scored,
        head_scored=True,
        error=RuntimeError("forward failed after dispatch"),
    ) == "unavailable"
    assert _native_control6_policy_evaluation_state(
        decision=skipped,
        head_scored=False,
        decision_view_unavailable_reason="stable Q2 fork was invalidated",
    ) == "unavailable"


def test_rejection_diagnostics_keep_q2_reason_identities_and_full_cause() -> None:
    original = RuntimeError("provenance mismatch: " + ("x" * 300))
    identity = {
        "session_id": "canonical-session",
        "listener_generation_id": {"expected": 7, "observed": 8},
        "turn_generation_id": {"expected": 12, "observed": 12},
        "state_epoch_id": {"expected": "epoch-a", "observed": "epoch-b"},
        "frame_id": {"expected": 91, "observed": 90},
        "listener_state_version": {"expected": 33, "observed": 32},
        "cache_position": {"expected": 204, "observed": 206},
    }

    diagnostics = _native_control6_policy_error_diagnostics(
        original,
        identity=identity,
        decision_view_unavailable_reason="stable Q2 fork could not be rebound",
        invalidation_repeat_count=2,
    )

    assert diagnostics["policy_evaluation_state"] == "unavailable"
    assert diagnostics["native_control6_head_scored"] is False
    assert diagnostics["reason"] == str(original)[:240]
    assert diagnostics["error_detail"] == str(original)
    assert diagnostics["original_cause"] == repr(original)
    assert diagnostics["decision_view_unavailable_reason"] == (
        "stable Q2 fork could not be rebound"
    )
    assert diagnostics["invalidation_repeat_count"] == 2
    for name, values in identity.items():
        assert diagnostics[name] == values


def test_invalidation_warning_is_emitted_once_per_session_epoch(caplog) -> None:
    session_id = "policy-error-localization-rate-limit"
    caplog.set_level(logging.DEBUG, logger="duplex_kernel.server")

    assert _record_native_control6_policy_invalidation(
        session_id=session_id, state_epoch_id="epoch-1", frame_id=10, reason="cause-a"
    ) == 0
    assert _record_native_control6_policy_invalidation(
        session_id=session_id, state_epoch_id="epoch-1", frame_id=11, reason="cause-b"
    ) == 1
    assert _record_native_control6_policy_invalidation(
        session_id=session_id, state_epoch_id="epoch-1", frame_id=12, reason="cause-c"
    ) == 2
    assert _record_native_control6_policy_invalidation(
        session_id=session_id, state_epoch_id="epoch-2", frame_id=13, reason="cause-d"
    ) == 0

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    debug = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(warnings) == 2
    assert len(debug) == 2
    assert "epoch-1" in warnings[0].getMessage()
    assert "epoch-2" in warnings[1].getMessage()
    assert "repeat=1" in debug[0].getMessage()
    assert "repeat=2" in debug[1].getMessage()
