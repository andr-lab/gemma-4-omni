from __future__ import annotations

import math
import inspect
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

import duplex_kernel.server as server_module
from duplex_kernel.gemma_native_control6_policy import (
    GemmaNativeControl6Policy,
    is_controller6_policy,
    legal_control_actions as native_control6_legal_actions,
    native_control6_effective_members,
)
from duplex_kernel.gemma_native_control6_batch_policy import NativeControl6RawScoreSnapshot
from duplex_kernel.duplex_control6 import DuplexControl6Policy
from duplex_kernel.duplex_control6 import legal_control_actions
from duplex_kernel.server import _native_control6_event_annotations
from duplex_kernel.turn_policy import ControlAction, ControlState, NoOpPolicy, PolicyInput, SpeakerGate
from training.duplex_qlora.native_control6 import ACTIONS, HIDDEN_WIDTH, NativeControl6LoRA
from client.duplex_stream_e2e import _project_server_event_for_log


class FixedSixLogitAdapter(NativeControl6LoRA):
    def __init__(self, logits: list[float]) -> None:
        super().__init__(torch.zeros((len(ACTIONS), HIDDEN_WIDTH)))
        self.fixed_logits = torch.tensor(logits, dtype=torch.float32)

    def forward(
        self, normalized_hidden: torch.Tensor, *, phase_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert normalized_hidden.shape == (1, HIDDEN_WIDTH)
        if phase_t is not None:
            assert phase_t.shape == (1, 4)
        return self.fixed_logits.to(normalized_hidden.device).unsqueeze(0)


def _state(**overrides: object) -> PolicyInput:
    values: dict[str, object] = {
        "session_id": "s1",
        "frame_id": 12,
        "user_pcm": None,
        "speaker_active": False,
        "generation_id": 4,
        "native_listener_generation": 4,
        "native_control_hidden": torch.zeros((1, HIDDEN_WIDTH)),
        "native_control_hidden_frame_id": 12,
        "native_control_hidden_at_ns": time.monotonic_ns(),
        "legal_actions": ("LISTEN", "SPEAK"),
    }
    values.update(overrides)
    return PolicyInput(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("speaker_active", "user_activity_detected"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_native_control6_effective_member_mapping_is_total_and_conserves_mass(
    speaker_active: bool,
    user_activity_detected: bool,
) -> None:
    members = native_control6_effective_members(
        speaker_active=speaker_active,
        user_activity_detected=user_activity_detected,
    )
    bucket_order = ("HOLD", "SPEAK", "YIELD", "BACKCHANNEL")
    flattened = [action for bucket in bucket_order for action in members[bucket]]
    assert len(flattened) == len(set(flattened))

    raw_probabilities = np.asarray([0.03, 0.07, 0.11, 0.17, 0.23, 0.39])
    physical = {
        bucket: sum(raw_probabilities[ACTIONS.index(action)] for action in members[bucket])
        for bucket in bucket_order
    }
    mapped_actions = set(flattened)
    excluded_actions = set(ACTIONS) - mapped_actions
    expected = (
        {
            "HOLD": ("CONTINUE",),
            "SPEAK": (),
            "YIELD": ("LISTEN", "YIELD", "REPLAN"),
            "BACKCHANNEL": (),
        }
        if speaker_active and user_activity_detected
        else {
            "HOLD": ("LISTEN", "CONTINUE"),
            "SPEAK": (),
            "YIELD": ("YIELD", "REPLAN"),
            "BACKCHANNEL": (),
        }
        if speaker_active
        else {
            "HOLD": ("LISTEN", "CONTINUE"),
            "SPEAK": ("SPEAK", "REPLAN"),
            "YIELD": (),
            "BACKCHANNEL": ("BACKCHANNEL",),
        }
    )
    assert members == expected
    assert mapped_actions | excluded_actions == set(ACTIONS)
    # Raw mass for physically unavailable rows is deliberately dropped before
    # the separate legal-action normalization; no row is silently reassigned.
    mapped_mass = sum(physical.values())
    expected_mapped_mass = sum(raw_probabilities[ACTIONS.index(action)] for action in mapped_actions)
    assert mapped_mass == pytest.approx(float(expected_mapped_mass))


def test_native_policy_does_not_turn_illegal_raw_listen_into_vocal_speak() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([100.0, 2.0, 1.0, 0.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt",
    )

    result = policy.step(_state())

    assert result.action is ControlAction.LISTEN
    assert result.listener_frame_id == 12
    assert result.source.startswith("gemma_lm_head_selected_rows+")
    assert result.diagnostics["base_action"] == "LISTEN"
    assert result.diagnostics["selected_action"] == "LISTEN"
    assert result.diagnostics["policy_fallback_reason"] is None


def test_control6_speak_waits_for_endpoint_without_masking_live_response_actions() -> None:
    while_user_speaks = legal_control_actions(
        active_response=False,
        response_pending=False,
        user_turn_pending=True,
        peer_turn_active=True,
        backchannel_available=True,
        renderer_plan_pending=False,
    )
    assert "SPEAK" not in while_user_speaks
    assert "BACKCHANNEL" in while_user_speaks

    during_assistant_response = legal_control_actions(
        active_response=True,
        response_pending=False,
        user_turn_pending=True,
        peer_turn_active=True,
        backchannel_available=False,
        renderer_plan_pending=False,
    )
    assert "SPEAK" not in during_assistant_response
    assert {"CONTINUE", "YIELD", "REPLAN"}.issubset(during_assistant_response)

    after_endpoint = legal_control_actions(
        active_response=False,
        response_pending=False,
        user_turn_pending=True,
        peer_turn_active=False,
        backchannel_available=False,
        renderer_plan_pending=False,
    )
    assert "SPEAK" in after_endpoint

    with_no_user_turn = legal_control_actions(
        active_response=False,
        response_pending=False,
        user_turn_pending=False,
        peer_turn_active=False,
        backchannel_available=False,
        renderer_plan_pending=False,
    )
    assert "SPEAK" not in with_no_user_turn


def test_native_policy_reports_raw_candidate_and_legal_selection_without_changing_action() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, 7.0, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )

    result = policy.step(_state(
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))

    # REPLAN remains the six-way raw argmax, but its probability is collapsed
    # into the physical YIELD candidate before selection.
    assert result.action is ControlAction.CONTINUE
    assert result.diagnostics["base_action"] == "YIELD"
    assert result.diagnostics["selected_action"] == "YIELD"
    assert result.diagnostics["native_control6_raw_row_index"] == 3
    assert result.diagnostics["native_control6_raw_token_id"] == 256004
    assert result.diagnostics["native_control6_raw_token"] == "<unused92>"
    assert result.diagnostics["native_control6_selected_row_index"] == 3
    assert result.diagnostics["native_control6_selected_token_id"] == 256004
    assert result.diagnostics["native_control6_selected_token"] == "<unused92>"
    assert result.diagnostics["policy_fallback_reason"] == (
        "two_distinct_consecutive_fresh_ticks_required"
    )


def test_native_policy_accepts_high_confidence_combined_yield_on_first_tick() -> None:
    assert "peer_turn_active" in PolicyInput.__dataclass_fields__, (
        "PolicyInput must expose peer_turn_active for native Control6 activity"
    )
    assert "high_confidence_single_tick_threshold" in inspect.signature(
        GemmaNativeControl6Policy
    ).parameters, "native Control6 must expose its single-tick confidence threshold"
    # REPLAN and YIELD are separate trained rows but one physical interruption
    # action while the assistant is speaking. Their combined probability is
    # well above .85 even though neither row alone is.
    logits = [-10.0] * len(ACTIONS)
    logits[ACTIONS.index("YIELD")] = 2.0
    logits[ACTIONS.index("REPLAN")] = 2.0
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt"
    )
    result = policy.step(_state(
        frame_id=12,
        speaker_active=True,
        peer_turn_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))

    assert result.action is ControlAction.YIELD
    assert result.diagnostics["native_control6_selected_effective_probability"] >= 0.85
    assert result.diagnostics["native_control6_hysteresis_count"] == 1


def test_native_policy_high_confidence_speak_is_fast_but_lower_confidence_keeps_hysteresis() -> None:
    assert "peer_turn_active" in PolicyInput.__dataclass_fields__, (
        "PolicyInput must expose peer_turn_active for native Control6 activity"
    )
    assert "high_confidence_single_tick_threshold" in inspect.signature(
        GemmaNativeControl6Policy
    ).parameters, "native Control6 must expose its single-tick confidence threshold"
    high_logits = [-10.0] * len(ACTIONS)
    high_logits[ACTIONS.index("SPEAK")] = 5.0
    high_policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(high_logits), checkpoint_path="cpu-test.pt"
    )
    high_result = high_policy.step(_state(
        speaker_active=False,
        peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert high_result.action is ControlAction.SPEAK
    assert high_result.diagnostics["native_control6_hysteresis_count"] == 1

    # SPEAK+REPLAN effective probability is below .85 after legal-mass
    # normalization; the configured default still requires adjacent ticks.
    lower_logits = [0.0] * len(ACTIONS)
    lower_logits[ACTIONS.index("SPEAK")] = 1.6
    lower_logits[ACTIONS.index("REPLAN")] = 1.6
    lower_policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(lower_logits), checkpoint_path="cpu-test.pt"
    )
    first = lower_policy.step(_state(
        frame_id=20,
        native_control_hidden_frame_id=20,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    second = lower_policy.step(_state(
        frame_id=21,
        native_control_hidden_frame_id=21,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert first.action is ControlAction.LISTEN
    assert first.diagnostics["native_control6_selected_effective_probability"] < 0.85
    assert second.action is ControlAction.SPEAK


def test_policy_input_peer_activity_defaults_false_and_rejects_non_bool() -> None:
    assert "peer_turn_active" in PolicyInput.__dataclass_fields__, (
        "PolicyInput must expose peer_turn_active for native Control6 activity"
    )
    state = _state()
    assert state.peer_turn_active is False
    with pytest.raises((TypeError, ValueError), match="peer_turn_active|bool"):
        _state(peer_turn_active=1)


def test_idle_continue_row_contributes_to_physical_listen_hold_not_backchannel() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([-2.0, -3.0, 10.0, -4.0, 5.0, -5.0]),
        checkpoint_path="cpu-test.pt",
    )

    result = policy.step(_state(legal_actions=("LISTEN", "BACKCHANNEL", "SPEAK")))

    assert result.diagnostics["base_action"] == "CONTINUE"
    assert result.action is ControlAction.LISTEN
    assert result.diagnostics["selected_action"] == "LISTEN"
    assert result.diagnostics["policy_fallback_reason"] is None


@pytest.mark.parametrize(
    ("logits", "legal", "speaker_active", "expected_raw", "expected", "expected_candidate", "expected_fallback"),
    [
        ([-1.0, -2.0, -3.0, -4.0, 10.0, -5.0], ("LISTEN", "BACKCHANNEL", "SPEAK"), False, "BACKCHANNEL", ControlAction.BACKCHANNEL, "BACKCHANNEL", None),
        ([-1.0, 10.0, -3.0, -4.0, -5.0, -6.0], ("LISTEN", "SPEAK"), False, "SPEAK", ControlAction.LISTEN, "SPEAK", "two_distinct_consecutive_fresh_ticks_required"),
        ([0.0, -1.0, 2.0, 10.0, -2.0, -3.0], ("CONTINUE", "YIELD"), True, "YIELD", ControlAction.CONTINUE, "YIELD", "two_distinct_consecutive_fresh_ticks_required"),
    ],
)
def test_native_policy_keeps_legal_raw_actions_and_uses_active_continue_fallback(
    logits: list[float], legal: tuple[str, ...], speaker_active: bool,
    expected_raw: str, expected: ControlAction, expected_candidate: str,
    expected_fallback: str | None,
) -> None:
    result = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    ).step(_state(legal_actions=legal, speaker_active=speaker_active))

    assert result.action is expected
    assert result.diagnostics["base_action"] == expected_raw
    assert result.diagnostics["selected_action"] == expected_candidate
    assert result.diagnostics.get("policy_fallback_reason") == expected_fallback


@pytest.mark.parametrize("fused_readback", [False, True])
def test_masked_probability_mass_underflow_returns_finite_legal_hold(
    fused_readback: bool,
) -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([-1.0e30, 1.0e30, -1.0e30, -1.0e30, -1.0e30, 1.0e30]),
        checkpoint_path="cpu-test.pt", fused_readback=fused_readback,
    )

    result = policy.step(_state(legal_actions=("LISTEN",)))

    assert result.action is ControlAction.LISTEN
    assert math.isfinite(result.confidence)
    assert result.confidence == 0.0
    assert result.diagnostics["native_control6_legal_effective_mass"] == 0.0
    assert result.diagnostics["native_control6_zero_legal_mass_safe_hold"] is True
    assert result.diagnostics["policy_fallback_reason"] == "zero_legal_physical_mass_safe_hold"


def test_native_policy_consumes_near_tie_raw_snapshot_but_keeps_live_legal_mask() -> None:
    class CountingAdapter(FixedSixLogitAdapter):
        def __init__(self) -> None:
            super().__init__([50.0, 40.0, 3.0, 3.0000002, -1.0, -2.0])
            self.forward_calls = 0

        def forward(
            self, normalized_hidden: torch.Tensor, *, phase_t: torch.Tensor | None = None
        ) -> torch.Tensor:
            self.forward_calls += 1
            return super().forward(normalized_hidden, phase_t=phase_t)

    adapter = CountingAdapter()
    policy = GemmaNativeControl6Policy(
        adapter, checkpoint_path="cpu-test.pt",
    )
    snapshot = NativeControl6RawScoreSnapshot(
        manifest_id="manifest-17",
        output_generation=17,
        session_id="s1",
        session_generation=4,
        state_epoch_id="epoch-s1",
        listener_state_version=8,
        frame_id=12,
        sequence_position=3,
        cache_position=46,
        query_index=0,
        checkpoint_path="cpu-test.pt",
        raw_logits=(0.0, 0.0, 3.0, 3.0000002, -1.0, -2.0),
    )

    result = policy.step(_state(
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
        native_control6_raw_scores=snapshot,
    ))

    assert adapter.forward_calls == 0
    assert result.action is ControlAction.CONTINUE
    assert result.diagnostics["native_control6_raw_row_index"] == 3
    assert result.diagnostics["native_control6_selected_row_index"] == 2
    assert result.diagnostics["native_control6_effective_candidate"] == "CONTINUE"
    assert result.diagnostics["native_control6_batched_raw_scores"] is True
    assert result.diagnostics["native_control6_batch_manifest_id"] == "manifest-17"


def test_phase_conditioned_policy_scores_after_phase_and_keeps_raw_snapshot_diagnostic_only() -> None:
    class CountingAdapter(FixedSixLogitAdapter):
        def __init__(self) -> None:
            super().__init__([0.0, 10.0, 2.0, 1.0, -1.0, -2.0])
            self.forward_calls = 0

        def forward(
            self, normalized_hidden: torch.Tensor, *, phase_t: torch.Tensor | None = None
        ) -> torch.Tensor:
            self.forward_calls += 1
            assert phase_t is not None
            return super().forward(normalized_hidden, phase_t=phase_t)

    adapter = CountingAdapter()
    policy = GemmaNativeControl6Policy(
        adapter, checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )
    snapshot = NativeControl6RawScoreSnapshot(
        manifest_id="manifest-17",
        output_generation=17,
        session_id="s1",
        session_generation=4,
        state_epoch_id="epoch-s1",
        listener_state_version=8,
        frame_id=12,
        sequence_position=3,
        cache_position=46,
        query_index=0,
        checkpoint_path="cpu-test.pt",
        raw_logits=(100.0, -100.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = policy.step(_state(
        legal_actions=("LISTEN", "SPEAK"),
        native_control_phase_t=(0.1, 1.0, 0.2, -0.3),
        native_control6_raw_scores=snapshot,
    ))

    assert result.action is ControlAction.LISTEN
    assert result.diagnostics["selected_action"] == "SPEAK"
    assert adapter.forward_calls == 1
    assert result.diagnostics["native_control6_batched_raw_scores"] is False
    assert result.diagnostics["native_control6_batched_raw_scores_diagnostic_only"] is True
    assert result.diagnostics["native_control6_batch_manifest_id"] == "manifest-17"


def test_raw_snapshot_uses_stable_listener_generation_not_turn_generation() -> None:
    snapshot = NativeControl6RawScoreSnapshot(
        manifest_id="manifest-17",
        output_generation=17,
        session_id="s1",
        session_generation=4,
        state_epoch_id="epoch-s1",
        listener_state_version=8,
        frame_id=12,
        sequence_position=3,
        cache_position=46,
        query_index=0,
        checkpoint_path="cpu-test.pt",
        raw_logits=(0.0, 0.0, 3.0, 3.0000002, -1.0, -2.0),
    )
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)), checkpoint_path="cpu-test.pt"
    )

    state = _state(
        generation_id=19,
        native_listener_generation=4,
        native_control6_raw_scores=snapshot,
    )
    assert policy.step(state).action is ControlAction.LISTEN

    # A listener reset/rebind invalidates the old row even though the turn
    # generation and selected frame still match.
    with pytest.raises(RuntimeError, match="provenance"):
        policy.step(replace(state, native_listener_generation=5))


@pytest.mark.parametrize(
    ("snapshot_overrides", "message"),
    [
        ({"session_id": "other"}, "provenance"),
        ({"session_generation": 5}, "provenance"),
        ({"frame_id": 13}, "provenance"),
        ({"checkpoint_path": "other.pt"}, "provenance"),
    ],
)
def test_native_policy_rejects_raw_score_snapshot_from_another_row_or_checkpoint(
    snapshot_overrides: dict[str, object], message: str,
) -> None:
    snapshot = NativeControl6RawScoreSnapshot(
        manifest_id="manifest-17",
        output_generation=17,
        session_id="s1",
        session_generation=4,
        state_epoch_id="epoch-s1",
        listener_state_version=8,
        frame_id=12,
        sequence_position=3,
        cache_position=46,
        query_index=0,
        checkpoint_path="cpu-test.pt",
        raw_logits=(0.0, 0.0, 3.0, 3.0000002, -1.0, -2.0),
    )
    snapshot = replace(snapshot, **snapshot_overrides)
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)), checkpoint_path="cpu-test.pt"
    )

    with pytest.raises(RuntimeError, match=message):
        policy.step(_state(native_control6_raw_scores=snapshot))


@pytest.mark.parametrize("logits", [
    [0.0, 1.0, 7.0, 12.0, -1.0, -2.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.4028234663852886e38, -3.4028234663852886e38, 1.0, 2.0, -4.0, 0.0],
])
@pytest.mark.parametrize(
    ("speaker_active", "legal"),
    [
        (False, ("LISTEN", "BACKCHANNEL", "SPEAK")),
        (True, ("CONTINUE", "YIELD")),
    ],
)
def test_fused_control6_readback_matches_ordinary_for_physical_legal_sets(
    logits: list[float], speaker_active: bool, legal: tuple[str, ...],
) -> None:
    state = _state(speaker_active=speaker_active, legal_actions=legal)
    baseline = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt"
    ).step(state)
    fused = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt",
        fused_readback=True,
    ).step(state)

    assert fused.action is baseline.action
    assert fused.confidence == baseline.confidence
    for key in (
        "base_action", "native_control6_raw_row_index",
        "native_control6_selected_row_index", "native_action_probability",
        "native_control6_effective_candidate", "native_control6_legal_effective_mass",
    ):
        assert fused.diagnostics[key] == baseline.diagnostics[key]


def test_fused_control6_uses_one_packed_readback_and_no_scalar_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_calls = 0
    item_calls = 0
    bool_calls = 0
    original_cpu = torch.Tensor.cpu
    original_item = torch.Tensor.item
    original_bool = torch.Tensor.__bool__
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, 7.0, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt", fused_readback=True,
    )

    def counted_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    def counted_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        nonlocal item_calls
        item_calls += 1
        return original_item(self, *args, **kwargs)

    def counted_bool(self: torch.Tensor) -> bool:
        nonlocal bool_calls
        bool_calls += 1
        return original_bool(self)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)

    policy.step(_state(speaker_active=True, legal_actions=("CONTINUE", "YIELD")))

    assert cpu_calls == 1
    assert item_calls == 0
    assert bool_calls == 0


def test_default_control6_readback_path_returns_required_physical_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu_calls = 0
    item_calls = 0
    bool_calls = 0
    original_cpu = torch.Tensor.cpu
    original_item = torch.Tensor.item
    original_bool = torch.Tensor.__bool__
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, 7.0, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )

    def counted_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    def counted_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        nonlocal item_calls
        item_calls += 1
        return original_item(self, *args, **kwargs)

    def counted_bool(self: torch.Tensor) -> bool:
        nonlocal bool_calls
        bool_calls += 1
        return original_bool(self)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)

    policy.step(_state(speaker_active=True, legal_actions=("CONTINUE", "YIELD")))

    assert cpu_calls == 0
    assert item_calls == 5
    assert bool_calls == 2
    assert cpu_calls + item_calls + bool_calls == 7


@pytest.mark.parametrize("bad_logit", [float("nan"), float("inf"), float("-inf")])
def test_fused_control6_readback_fails_closed_on_nonfinite_logits(bad_logit: float) -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, bad_logit, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt", fused_readback=True,
    )

    with pytest.raises(RuntimeError, match="invalid six-way logits"):
        policy.step(_state(speaker_active=True, legal_actions=("CONTINUE", "YIELD")))


def test_fused_control6_readback_rejects_malformed_packed_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "stack", lambda _tensors: torch.tensor([1.0, 2.0, 3.0]))
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, 7.0, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt", fused_readback=True,
    )

    with pytest.raises(RuntimeError, match="invalid fused physical-action readback"):
        policy.step(_state(speaker_active=True, legal_actions=("CONTINUE", "YIELD")))


def test_actual_native_fastpath_selection_reaches_decision_event_and_client_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "_NATIVE_CONTROL6_TRACE_ENABLED", False)
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, 7.0, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )
    decision = policy.step(_state(
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))
    gate_event = SpeakerGate().apply(decision, type("Session", (), {
        "control_state": ControlState.SPEAKING,
        "session_generation_id": 4,
    })())

    # Exercise the real six-row policy result, then the server's event annotation
    # and the pair client's evidence-field projection. No tensor readback is
    # introduced by either projection layer.
    event = {
        "type": "duplex_control6.decision",
        "native_control6_head_scored": True,
        "action": decision.action.value,
        "accepted": gate_event.accepted,
        **_native_control6_event_annotations(
            decision, head_scored=True, accepted=gate_event.accepted
        ),
    }
    projected = _project_server_event_for_log(event)

    assert decision.diagnostics["base_action"] == "YIELD"
    assert decision.action is ControlAction.CONTINUE
    assert gate_event.accepted is True
    assert projected["native_control6_raw_row_index"] == 3
    assert projected["native_control6_raw_token_id"] == 256004
    assert projected["native_control6_raw_token"] == "<unused92>"
    assert projected["native_control6_selected_row_index"] == 3
    assert projected["native_control6_selected_token_id"] == 256004
    assert projected["native_control6_selected_token"] == "<unused92>"
    assert projected["gate_applied_action"] == "CONTINUE"
    assert projected["gate_application_scope"] == "control_plane_acceptance_not_audible_output"
    assert projected["native_control6_token_selection_scope"] == (
        "six_projected_rows_collapsed_to_physical_actions_not_full_vocabulary_sample"
    )
    assert projected["policy_fallback_reason"] == (
        "two_distinct_consecutive_fresh_ticks_required"
    )
    assert "native_control6_input_rows" not in projected


def test_native_yield_candidate_can_select_accepted_continue_without_changing_speaking_state() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0, 1.0, 7.0, 12.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )
    session = type("Session", (), {})()
    session.control_state = ControlState.SPEAKING
    session.session_generation_id = 4

    result = policy.step(_state(
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))
    gate_event = SpeakerGate().apply(result, session)

    assert result.diagnostics["base_action"] == "YIELD"
    assert result.action is ControlAction.CONTINUE
    assert gate_event.accepted is True
    assert gate_event.action is ControlAction.CONTINUE
    assert session.control_state is ControlState.SPEAKING


def test_native_policy_opt_in_score_trace_exposes_raw_six_and_collapsed_physical_rows() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        checkpoint_path="cpu-test.pt",
        trace_scores=True,
    )

    result = policy.step(_state(legal_actions=("LISTEN", "SPEAK")))

    assert len(result.diagnostics["native_action_base_logits"]) == len(ACTIONS)
    assert len(result.diagnostics["native_action_logits"]) == len(ACTIONS)
    probabilities = result.diagnostics["native_action_probabilities"]
    assert len(probabilities) == len(ACTIONS)
    assert probabilities[3] == 0.0
    assert probabilities[4] == 0.0
    assert sum(probabilities) == pytest.approx(1.0)
    assert sum(result.diagnostics["native_action_raw_probabilities"]) == pytest.approx(1.0)
    effective = result.diagnostics["native_action_effective_probabilities"]
    assert effective["HOLD"] == pytest.approx(
        result.diagnostics["native_action_raw_probabilities"][0]
        + result.diagnostics["native_action_raw_probabilities"][2]
    )
    assert effective["SPEAK"] == pytest.approx(
        result.diagnostics["native_action_raw_probabilities"][1]
        + result.diagnostics["native_action_raw_probabilities"][5]
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"native_control_hidden": None}, "requires fresh"),
        ({"native_control_hidden_frame_id": 11}, "refuses stale"),
        ({"native_control_hidden_at_ns": 0}, "availability watermark"),
        ({"native_control_hidden": torch.zeros((2, HIDDEN_WIDTH))}, "exactly one"),
        ({"legal_actions": ()}, "non-empty legal-action"),
        ({"legal_actions": ("HARD_STOP",)}, "unknown legal actions"),
    ],
)
@pytest.mark.parametrize("fused_readback", [False, True])
def test_native_policy_fails_closed_on_stale_or_invalid_input(
    overrides: dict[str, object], message: str, fused_readback: bool
) -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)),
        checkpoint_path="cpu-test.pt",
        fused_readback=fused_readback,
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        policy.step(_state(**overrides))

def test_native_no_input_does_not_hide_pending_speak_but_keeps_idle_safety() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)),
        checkpoint_path="cpu-test.pt",
    )

    with pytest.raises(RuntimeError, match="fresh listener hidden for a pending user turn"):
        policy.step(_state(
            native_control_hidden=None,
            native_control_hidden_frame_id=None,
            native_control_hidden_at_ns=None,
            native_control_no_input_tick=True,
            legal_actions=("LISTEN", "SPEAK"),
        ))

    idle = policy.step(_state(
        native_control_hidden=None,
        native_control_hidden_frame_id=None,
        native_control_hidden_at_ns=None,
        native_control_no_input_tick=True,
        legal_actions=("LISTEN",),
    ))
    assert idle.action is ControlAction.LISTEN
    assert idle.listener_frame_id == idle.frame_id
    assert not idle.diagnostics.get("native_control6_head_scored", False)
    assert not any("_token" in key or "_row_index" in key for key in idle.diagnostics)

    active = policy.step(_state(
        native_control_hidden=None,
        native_control_hidden_frame_id=None,
        native_control_hidden_at_ns=None,
        native_control_no_input_tick=True,
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))
    assert active.action is ControlAction.CONTINUE
    assert active.listener_frame_id == active.frame_id
    assert not active.diagnostics.get("native_control6_head_scored", False)


def test_fresh_zero_pcm_tick_still_uses_native_logits() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([1.0, 100.0, 2.0, 0.0, -1.0, -2.0]),
        checkpoint_path="cpu-test.pt",
    )
    result = policy.step(_state(
        user_pcm=torch.zeros(320),
        native_control_no_input_tick=False,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert result.action is ControlAction.SPEAK
    assert result.diagnostics["native_control6_effective_candidate"] == "SPEAK"


def test_both_control6_policy_classes_share_controller_lifecycle_capability() -> None:
    native = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)),
        checkpoint_path="cpu-test.pt",
    )
    legacy = object.__new__(DuplexControl6Policy)
    assert is_controller6_policy(native)
    assert is_controller6_policy(legacy)
    assert not is_controller6_policy(NoOpPolicy())


def test_physical_collapse_sums_yield_and_replan_before_argmax() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([-8.0, -8.0, 1.0, 0.75, -8.0, 0.75]),
        checkpoint_path="cpu-test.pt",
        trace_scores=True,
        minimum_transition_confidence=0.0,
    )
    first = policy.step(_state(
        frame_id=12,
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))
    second = policy.step(_state(
        frame_id=13,
        native_control_hidden_frame_id=13,
        native_control_hidden_at_ns=time.monotonic_ns(),
        speaker_active=True,
        legal_actions=("CONTINUE", "YIELD"),
    ))

    # Each learned row loses to CONTINUE, but their physical YIELD mass wins
    # after the speaking-state REPLAN -> YIELD collapse.
    assert first.action is ControlAction.CONTINUE
    assert first.diagnostics["native_control6_effective_candidate"] == "YIELD"
    assert second.action is ControlAction.YIELD
    assert second.diagnostics["base_action"] == "CONTINUE"
    assert second.diagnostics["native_action_raw_probabilities"][3] < second.diagnostics[
        "native_action_raw_probabilities"][2]
    assert second.diagnostics["native_action_raw_probabilities"][5] < second.diagnostics[
        "native_action_raw_probabilities"][2]
    effective = second.diagnostics["native_action_effective_probabilities"]
    assert effective["YIELD"] == pytest.approx(
        second.diagnostics["native_action_raw_probabilities"][3]
        + second.diagnostics["native_action_raw_probabilities"][5]
    )
    assert effective["YIELD"] > effective["HOLD"]
    assert second.diagnostics["native_control6_legal_effective_mass"] == pytest.approx(
        effective["HOLD"] + effective["YIELD"]
    )


def test_replan_collapses_to_speak_while_idle_and_speak_is_legal_during_peer_turn() -> None:
    legal = native_control6_legal_actions(
        speaker_active=False,
        user_turn_pending=False,
        peer_turn_active=True,
    )
    assert legal == ("LISTEN", "BACKCHANNEL", "SPEAK")

    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([-8.0, 0.0, -8.0, -8.0, -8.0, 3.0]),
        checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )
    first = policy.step(_state(
        frame_id=12,
        speaker_active=False,
        legal_actions=legal,
    ))
    second = policy.step(_state(
        frame_id=13,
        native_control_hidden_frame_id=13,
        native_control_hidden_at_ns=time.monotonic_ns(),
        speaker_active=False,
        legal_actions=legal,
    ))

    assert first.action is ControlAction.LISTEN
    assert first.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert second.action is ControlAction.SPEAK
    assert second.diagnostics["base_action"] == "REPLAN"
    assert second.diagnostics["selected_action"] == "SPEAK"
    session = type("Session", (), {
        "control_state": ControlState.LISTENING,
        "session_generation_id": 4,
    })()
    gate_event = SpeakerGate().apply(second, session)
    assert gate_event.accepted is True
    assert gate_event.action is ControlAction.SPEAK
    assert session.control_state is ControlState.SPEAKING


@pytest.mark.parametrize(
    ("speaker_active", "legal", "expected_candidates"),
    [
        (
            False,
            ("LISTEN", "BACKCHANNEL", "SPEAK"),
            ("LISTEN", "SPEAK", "LISTEN", "LISTEN", "BACKCHANNEL", "SPEAK"),
        ),
        (
            True,
            ("CONTINUE", "YIELD"),
            ("CONTINUE", "CONTINUE", "CONTINUE", "YIELD", "CONTINUE", "YIELD"),
        ),
    ],
)
def test_each_of_six_learned_rows_maps_to_only_a_physical_action(
    speaker_active: bool,
    legal: tuple[str, ...],
    expected_candidates: tuple[str, ...],
) -> None:
    for row_index, expected in enumerate(expected_candidates):
        logits = [-1000.0] * len(ACTIONS)
        logits[row_index] = 0.0
        decision = GemmaNativeControl6Policy(
            FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt",
        ).step(_state(
            speaker_active=speaker_active,
            legal_actions=legal,
        ))
        assert decision.diagnostics["native_control6_effective_candidate"] == expected
        assert decision.action.value != "REPLAN"


def test_hysteresis_requires_distinct_consecutive_frames_and_resets_on_gap() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([-8.0, 4.0, -8.0, -8.0, -8.0, -8.0]),
        checkpoint_path="cpu-test.pt",
        high_confidence_single_tick_threshold=1.0,
    )

    def step(frame_id: int) -> object:
        return policy.step(_state(
            frame_id=frame_id,
            native_control_hidden_frame_id=frame_id,
            native_control_hidden_at_ns=time.monotonic_ns(),
            speaker_active=False,
            legal_actions=("LISTEN", "BACKCHANNEL", "SPEAK"),
        ))

    assert step(12).action is ControlAction.LISTEN
    duplicate = step(12)
    assert duplicate.action is ControlAction.LISTEN
    assert duplicate.diagnostics["native_control6_hysteresis_count"] == 1
    assert step(14).action is ControlAction.LISTEN
    assert step(15).action is ControlAction.SPEAK


def test_hysteresis_resets_on_candidate_state_change_and_missing_hidden() -> None:
    policy = GemmaNativeControl6Policy(
        # After legal physical-group normalization SPEAK/YIELD are both in
        # [.70, .85), so the first tick remains a pending hysteresis hold.
        FixedSixLogitAdapter([-0.3, 1.3, 0.0, 0.0, -0.5, 1.2]),
        checkpoint_path="cpu-test.pt",
        minimum_transition_confidence=0.0,
    )
    idle_legal = ("LISTEN", "BACKCHANNEL", "SPEAK")
    speaking_legal = ("CONTINUE", "YIELD")

    first_speak = policy.step(_state(
        frame_id=12, speaker_active=False, legal_actions=idle_legal,
    ))
    changed_state = policy.step(_state(
        frame_id=13,
        native_control_hidden_frame_id=13,
        native_control_hidden_at_ns=time.monotonic_ns(),
        speaker_active=True,
        legal_actions=speaking_legal,
    ))
    assert first_speak.action is ControlAction.LISTEN
    assert changed_state.action is ControlAction.CONTINUE
    assert changed_state.diagnostics["native_control6_hysteresis_count"] == 1

    missing = policy.step(_state(
        frame_id=14,
        native_control_hidden=None,
        native_control_hidden_frame_id=None,
        native_control_hidden_at_ns=None,
        native_control_no_input_tick=True,
        speaker_active=True,
        legal_actions=speaking_legal,
    ))
    assert missing.action is ControlAction.CONTINUE

    first_yield = policy.step(_state(
        frame_id=15,
        native_control_hidden_frame_id=15,
        native_control_hidden_at_ns=time.monotonic_ns(),
        speaker_active=True,
        legal_actions=speaking_legal,
    ))
    assert first_yield.action is ControlAction.CONTINUE
    assert first_yield.diagnostics["native_control6_hysteresis_count"] == 1


def test_idle_physical_hold_is_listen_and_idle_replan_mass_is_speak_mass() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.5, 0.0, 0.5, -8.0, -8.0, 0.0]),
        checkpoint_path="cpu-test.pt",
        trace_scores=True,
    )
    result = policy.step(_state(
        speaker_active=False,
        legal_actions=("LISTEN", "SPEAK"),
    ))

    assert result.action is ControlAction.LISTEN
    assert result.diagnostics["native_action_effective_order"] == [
        "HOLD", "SPEAK", "YIELD", "BACKCHANNEL"
    ]
    assert len(result.diagnostics["native_action_raw_probabilities"]) == 6
    assert sum(result.diagnostics["native_action_raw_probabilities"]) == pytest.approx(1.0)
    assert result.diagnostics["native_action_effective_probabilities"]["HOLD"] == pytest.approx(
        result.diagnostics["native_action_raw_probabilities"][0]
        + result.diagnostics["native_action_raw_probabilities"][2]
    )


def test_peer_turn_activity_and_speech_floor_are_explicit_inputs() -> None:
    parameters = inspect.signature(GemmaNativeControl6Policy).parameters
    assert "speech_rms_floor" in parameters, (
        "native Control6 must expose its current-PCM speech RMS floor"
    )
    assert parameters["speech_rms_floor"].default == pytest.approx(0.001)
    for invalid_floor in (0.0, -0.1, float("nan"), float("inf"), True):
        with pytest.raises((TypeError, ValueError), match="speech_rms_floor|RMS floor"):
            GemmaNativeControl6Policy(
                FixedSixLogitAdapter([0.0] * len(ACTIONS)),
                checkpoint_path="cpu-test.pt",
                speech_rms_floor=invalid_floor,
            )


@pytest.mark.parametrize(
    ("peer_turn_active", "user_pcm"),
    [
        (True, np.zeros(1280, dtype=np.float32)),
        (False, np.full(1280, 0.002, dtype=np.float32)),
    ],
    ids=("peer-flag", "current-pcm-rms"),
)
@pytest.mark.parametrize(
    ("row", "expected"),
    [(0, "YIELD"), (2, "CONTINUE"), (3, "YIELD"), (5, "YIELD")],
)
def test_active_overlap_maps_hold_to_continue_and_yield_rows_to_yield(
    peer_turn_active: bool,
    user_pcm: np.ndarray,
    row: int,
    expected: str,
) -> None:
    assert "peer_turn_active" in PolicyInput.__dataclass_fields__, (
        "PolicyInput must expose peer_turn_active for native Control6 activity"
    )
    logits = [-100.0] * len(ACTIONS)
    logits[row] = 100.0
    decision = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt", trace_scores=True
    ).step(_state(
        speaker_active=True,
        peer_turn_active=peer_turn_active,
        user_pcm=user_pcm,
        legal_actions=("CONTINUE", "YIELD"),
    ))

    assert decision.diagnostics["native_control6_effective_candidate"] == expected
    assert decision.diagnostics["native_control6_user_activity_detected"] is True
    assert sum(decision.diagnostics["native_action_effective_probabilities"].values()) == pytest.approx(1.0)


def test_silent_active_speaker_keeps_listen_in_hold_and_omits_activity() -> None:
    logits = [-100.0] * len(ACTIONS)
    logits[ACTIONS.index("LISTEN")] = 100.0
    decision = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits), checkpoint_path="cpu-test.pt"
    ).step(_state(
        speaker_active=True,
        peer_turn_active=False,
        user_pcm=np.zeros(1280, dtype=np.float32),
        legal_actions=("CONTINUE", "YIELD"),
    ))
    assert decision.diagnostics["native_control6_user_activity_detected"] is False
    assert decision.diagnostics["native_control6_effective_candidate"] == "CONTINUE"


def test_native1540_valid_user_to_assistant_ratio_can_detect_activity_below_rms_floor() -> None:
    user_rms = 0.0009
    assistant_rms = 0.001
    ratio = math.log10(user_rms) - math.log10(assistant_rms)
    logits = [-100.0] * len(ACTIONS)
    logits[ACTIONS.index("YIELD")] = 100.0
    decision = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits),
        checkpoint_path="cpu-test.pt",
        phase_profile="native1540",
    ).step(_state(
        speaker_active=True,
        peer_turn_active=False,
        user_pcm=np.full(1280, user_rms, dtype=np.float32),
        native_control_phase_t=np.asarray(
            [math.log10(user_rms), 0.0, math.log10(assistant_rms), ratio],
            dtype=np.float32,
        ),
        native_control_assistant_pcm_prior=np.full(1280, assistant_rms, dtype=np.float32),
        native_control_assistant_prior_frame_id=11,
        legal_actions=("LISTEN", "CONTINUE", "YIELD"),
    ))

    assert decision.diagnostics["native_control6_user_rms"] == pytest.approx(user_rms)
    assert decision.diagnostics["native_control6_ratio_activity_detected"] is True
    assert decision.diagnostics["native_control6_user_activity_detected"] is True
    assert decision.diagnostics["native_control6_effective_members"]["HOLD"] == ["CONTINUE"]


@pytest.mark.parametrize(
    ("user_rms", "assistant_rms"),
    [(0.0, 0.0), (0.0005, 0.0005)],
    ids=("both-silent", "both-below-floor"),
)
def test_native1540_ratio_does_not_create_activity_from_silence_or_quiet_noise(
    user_rms: float, assistant_rms: float
) -> None:
    logits = [-100.0] * len(ACTIONS)
    logits[ACTIONS.index("LISTEN")] = 100.0
    decision = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits),
        checkpoint_path="cpu-test.pt",
        phase_profile="native1540",
    ).step(_state(
        speaker_active=True,
        peer_turn_active=False,
        user_pcm=np.full(1280, user_rms, dtype=np.float32),
        native_control_phase_t=np.asarray(
            [math.log10(max(user_rms, 1e-6)), 0.0,
             math.log10(max(assistant_rms, 1e-6)), 0.0],
            dtype=np.float32,
        ),
        native_control_assistant_pcm_prior=np.full(1280, assistant_rms, dtype=np.float32),
        native_control_assistant_prior_frame_id=11,
        legal_actions=("LISTEN", "CONTINUE", "YIELD"),
    ))

    assert decision.diagnostics["native_control6_ratio_activity_detected"] is False
    assert decision.diagnostics["native_control6_user_activity_detected"] is False
    assert decision.diagnostics["native_control6_effective_members"]["HOLD"] == [
        "LISTEN", "CONTINUE"
    ]


def test_silent_pcm_and_floored_phase_keep_legal_hold_at_listen() -> None:
    result = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([100.0, -100.0, -100.0, -100.0, -100.0, -100.0]),
        checkpoint_path="cpu-test.pt",
    ).step(_state(
        user_pcm=np.zeros(1280, dtype=np.float32),
        peer_turn_active=False,
        native_control_phase_t=(-6.0, 0.0, -6.0, 0.0),
        legal_actions=("LISTEN", "SPEAK"),
    ))

    assert result.diagnostics["native_control6_effective_candidate"] == "LISTEN"
    assert result.action is ControlAction.LISTEN


def test_single_tick_confidence_threshold_defaults_and_validates() -> None:
    parameters = inspect.signature(GemmaNativeControl6Policy).parameters
    assert "high_confidence_single_tick_threshold" in parameters, (
        "native Control6 must expose the high-confidence single-tick threshold"
    )
    assert parameters["high_confidence_single_tick_threshold"].default == pytest.approx(0.85)
    for invalid_threshold in (-0.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(
            (TypeError, ValueError),
            match="high_confidence_single_tick_threshold|threshold",
        ):
            GemmaNativeControl6Policy(
                FixedSixLogitAdapter([0.0] * len(ACTIONS)),
                checkpoint_path="cpu-test.pt",
                high_confidence_single_tick_threshold=invalid_threshold,
            )


def test_transition_confidence_floor_ignores_weak_ticks_and_requires_two_eligible_ticks() -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)),
        checkpoint_path="cpu-test.pt",
    )
    assert policy.minimum_transition_confidence == pytest.approx(0.70)

    def apply(frame_id: int, probability: float) -> tuple[str, int, bool]:
        state = _state(
            frame_id=frame_id,
            native_control_hidden_frame_id=frame_id,
            speaker_active=False,
            legal_actions=("LISTEN", "SPEAK"),
        )
        return policy._apply_hysteresis(state, "SPEAK", probability)

    # A weak candidate cannot prime the streak. Two consecutive candidates
    # inside [minimum, high-confidence) are required for the transition.
    assert apply(30, 0.099) == ("LISTEN", 0, False)
    assert apply(31, 0.711) == ("LISTEN", 1, True)
    assert apply(32, 0.711) == ("SPEAK", 2, False)

    # An intervening weak candidate clears an already pending transition.
    assert apply(40, 0.711) == ("LISTEN", 1, True)
    assert apply(41, 0.099) == ("LISTEN", 0, False)
    assert apply(42, 0.711) == ("LISTEN", 1, True)
    assert apply(43, 0.711) == ("SPEAK", 2, False)


def test_transition_confidence_floor_is_configurable_but_cannot_exceed_single_tick_threshold() -> None:
    parameters = inspect.signature(GemmaNativeControl6Policy).parameters
    assert "minimum_transition_confidence" in parameters
    assert parameters["minimum_transition_confidence"].default == pytest.approx(0.70)
    GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)),
        checkpoint_path="cpu-test.pt",
        minimum_transition_confidence=0.0,
    )
    with pytest.raises(ValueError, match="minimum_transition_confidence"):
        GemmaNativeControl6Policy(
            FixedSixLogitAdapter([0.0] * len(ACTIONS)),
            checkpoint_path="cpu-test.pt",
            high_confidence_single_tick_threshold=0.85,
            minimum_transition_confidence=0.9,
        )


def test_high_confidence_collapsed_speak_and_yield_bypass_only_two_tick_wait() -> None:
    speak_policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([-100.0, 10.0, -100.0, -100.0, -100.0, 9.0]),
        checkpoint_path="cpu-test.pt",
    )
    speak = speak_policy.step(_state(
        user_pcm=np.zeros(1280, dtype=np.float32),
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert speak.action is ControlAction.SPEAK
    assert speak.diagnostics["native_control6_effective_candidate"] == "SPEAK"


def test_early_speak_threshold_defaults_and_validates_with_transition_bounds() -> None:
    parameters = inspect.signature(GemmaNativeControl6Policy).parameters
    assert parameters["early_speak_confidence_threshold"].default == pytest.approx(0.80)
    for invalid in (1.01, float("nan"), float("inf"), True, "0.8"):
        with pytest.raises((TypeError, ValueError), match="early_speak_confidence_threshold|threshold"):
            GemmaNativeControl6Policy(
                FixedSixLogitAdapter([0.0] * len(ACTIONS)), checkpoint_path="cpu-test.pt",
                early_speak_confidence_threshold=invalid,
            )
    for bounds in (
        {"minimum_transition_confidence": 0.81},
        {"high_confidence_single_tick_threshold": 0.79},
    ):
        with pytest.raises(ValueError, match="early_speak_confidence_threshold|threshold"):
            GemmaNativeControl6Policy(
                FixedSixLogitAdapter([0.0] * len(ACTIONS)), checkpoint_path="cpu-test.pt",
                **bounds,
            )


@pytest.mark.parametrize("fused_readback", [False, True])
def test_conditional_legal_physical_probability_drives_thresholds_and_diagnostics(
    fused_readback: bool,
) -> None:
    # Legal physical mass is .60; SPEAK's pre-normalized mass is .50, or 5/6
    # after conditioning on legal physical groups.
    probabilities = [0.05, 0.25, 0.05, 0.20, 0.20, 0.25]
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(np.log(probabilities).tolist()), checkpoint_path="cpu-test.pt",
        fused_readback=fused_readback,
        trace_scores=True,
    )
    first = policy.step(_state(frame_id=30, native_control_hidden_frame_id=30))
    second = policy.step(_state(frame_id=31, native_control_hidden_frame_id=31))
    assert first.action is ControlAction.LISTEN
    assert second.action is ControlAction.SPEAK
    assert second.diagnostics["native_control6_legal_effective_mass"] == pytest.approx(0.60)
    assert second.diagnostics["native_control6_selected_effective_probability"] == pytest.approx(5 / 6)
    assert second.diagnostics[
        "native_control6_selected_effective_pre_normalization_probability"
    ] == pytest.approx(0.50)
    assert second.confidence == pytest.approx(5 / 6)
    assert second.diagnostics["native_control6_early_speak_confidence"] == pytest.approx(
        0.25 / 0.60
    )


def test_peer_active_early_speak_gate_holds_output_but_preserves_hysteresis() -> None:
    # SPEAK has .50 of .65 legal mass: .769 conditionally.
    probabilities = [0.10, 0.20, 0.05, 0.20, 0.15, 0.30]
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(np.log(probabilities).tolist()), checkpoint_path="cpu-test.pt",
    )
    first = policy.step(_state(
        frame_id=40, native_control_hidden_frame_id=40, peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    second = policy.step(_state(
        frame_id=41, native_control_hidden_frame_id=41, peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert first.action is ControlAction.LISTEN
    assert first.diagnostics["native_control6_hysteresis_count"] == 1
    assert second.action is ControlAction.LISTEN
    assert second.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert second.diagnostics["native_control6_hysteresis_count"] == 2
    assert second.diagnostics["native_control6_hysteresis_applied_action"] == "SPEAK"
    assert second.diagnostics["native_control6_decision_action"] == "LISTEN"
    assert second.diagnostics["native_control6_early_speak_gated"] is True
    assert second.diagnostics["policy_fallback_reason"] == (
        "native_control6_early_speak_peer_quiet_hold"
    )
    assert second.reason == "gemma_native_control6_early_speak_peer_quiet_hold"

    # A gated candidate below .70 still clears the pending streak.
    policy.adapter.fixed_logits.copy_(torch.tensor(np.log(
        [0.20, 0.30, 0.15, 0.10, 0.05, 0.20]
    ), dtype=torch.float32))
    weak = policy.step(_state(
        frame_id=42, native_control_hidden_frame_id=42, peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    ))
    assert weak.action is ControlAction.LISTEN
    assert weak.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert weak.diagnostics["native_control6_selected_effective_probability"] < 0.70
    assert weak.diagnostics["native_control6_hysteresis_count"] == 0

    # After endpoint the .80 peer-active gate is absent and normal two-tick
    # hysteresis still applies to this .769 conditional probability.
    endpoint_policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(np.log(probabilities).tolist()), checkpoint_path="cpu-test.pt",
    )
    endpoint_first = endpoint_policy.step(_state(
        frame_id=50, native_control_hidden_frame_id=50, peer_turn_active=False,
    ))
    endpoint_second = endpoint_policy.step(_state(
        frame_id=51, native_control_hidden_frame_id=51, peer_turn_active=False,
    ))
    assert endpoint_first.action is ControlAction.LISTEN
    assert endpoint_first.diagnostics["native_control6_early_speak_gated"] is False
    assert endpoint_second.action is ControlAction.SPEAK


@pytest.mark.parametrize("entrypoint", ["step", "step_with_logits"])
def test_active_user_replan_mass_cannot_authorize_early_speak(
    entrypoint: str,
) -> None:
    probabilities = [0.05, 0.02, 0.02, 0.02, 0.01, 0.88]
    logits = torch.tensor(np.log(probabilities), dtype=torch.float32)
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits.tolist()), checkpoint_path="cpu-test.pt",
    )

    def decide(frame_id: int, user_pcm: np.ndarray):
        state = _state(
            frame_id=frame_id,
            native_control_hidden_frame_id=frame_id,
            user_pcm=user_pcm,
            peer_turn_active=False,
            legal_actions=("LISTEN", "SPEAK"),
        )
        if entrypoint == "step":
            return policy.step(state)
        return policy.step_with_logits(
            state,
            logits,
            tick_id=(state.session_id, state.generation_id, state.frame_id),
        )

    active = decide(70, np.full(1280, 0.05, dtype=np.float32))
    assert active.diagnostics["base_action"] == "REPLAN"
    assert active.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert active.diagnostics["native_control6_selected_effective_probability"] > 0.85
    assert active.diagnostics["native_control6_user_rms"] == pytest.approx(0.05)
    assert active.action is ControlAction.LISTEN
    assert active.diagnostics["native_control6_hysteresis_count"] == 1
    assert active.diagnostics["native_control6_hysteresis_applied_action"] == "SPEAK"
    assert active.diagnostics["native_control6_early_speak_gated"] is True
    assert active.diagnostics["native_control6_early_speak_confidence"] < 0.80

    # The speech tracker remains peer-active for the rest of its endpoint
    # window. Physical SPEAK+REPLAN mass alone cannot clear the raw-SPEAK gate.
    endpoint_state = _state(
        frame_id=71,
        native_control_hidden_frame_id=71,
        user_pcm=np.zeros(1280, dtype=np.float32),
        peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    )
    endpoint = (
        policy.step(endpoint_state)
        if entrypoint == "step"
        else policy.step_with_logits(
            endpoint_state,
            logits,
            tick_id=(endpoint_state.session_id, endpoint_state.generation_id, endpoint_state.frame_id),
        )
    )
    assert endpoint.diagnostics["native_control6_user_activity_detected"] is True
    assert endpoint.diagnostics["native_control6_user_rms"] < endpoint.diagnostics[
        "native_control6_speech_rms_floor"
    ]
    assert endpoint.diagnostics["native_control6_early_speak_gated"] is True
    assert endpoint.diagnostics["native_control6_peer_turn_active"] is True
    assert endpoint.diagnostics["native_control6_early_speak_confidence_source"] == (
        "legal_normalized_raw_speak_member"
    )
    assert endpoint.diagnostics["native_control6_early_speak_confidence"] < 0.80
    assert endpoint.action is ControlAction.LISTEN


@pytest.mark.parametrize("entrypoint", ["step", "step_with_logits"])
def test_voiced_replan_evidence_is_remembered_until_two_quiet_ticks(
    entrypoint: str,
) -> None:
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)),
        checkpoint_path="cpu-test.pt",
        trace_scores=True,
    )
    speak_logits = (
        np.log([0.093, 0.001, 0.001, 0.001, 0.002, 0.902]).tolist(),
        np.log([0.123, 0.001, 0.001, 0.001, 0.002, 0.872]).tolist(),
    )
    hold_logits = np.log([0.88, 0.01, 0.02, 0.01, 0.02, 0.06]).tolist()
    results = {}

    def decide(frame_id: int, logits_values: list[float], *, voiced: bool):
        logits = torch.tensor(logits_values, dtype=torch.float32)
        policy.adapter.fixed_logits.copy_(logits)
        state = _state(
            frame_id=frame_id,
            native_control_hidden_frame_id=frame_id,
            user_pcm=(
                np.full(1280, 0.05, dtype=np.float32)
                if voiced else np.zeros(1280, dtype=np.float32)
            ),
            peer_turn_active=True,
            user_turn_onset_frame_id=700,
            legal_actions=("LISTEN", "SPEAK", "BACKCHANNEL"),
        )
        decision = (
            policy.step(state)
            if entrypoint == "step"
            else policy.step_with_logits(
                state,
                logits,
                tick_id=(state.session_id, state.generation_id, state.frame_id),
            )
        )
        results[frame_id] = decision
        return decision

    # Two consecutive voiced physical SPEAK+REPLAN ticks qualify the endpoint
    # even though the raw SPEAK member is tiny and the early gate holds output.
    first = decide(30, speak_logits[0], voiced=True)
    second = decide(31, speak_logits[1], voiced=True)
    assert first.action is ControlAction.LISTEN
    assert second.action is ControlAction.LISTEN
    assert first.diagnostics["native_control6_selected_effective_probability"] >= 0.90
    assert second.diagnostics["native_control6_selected_effective_probability"] >= 0.87
    assert second.diagnostics["native_control6_endpoint_speak_pending"] is True
    assert second.diagnostics["native_control6_endpoint_speak_evidence_count"] == 2

    # The candidate returns to HOLD for the rest of speech. The pending intent
    # survives this change and does not trigger while user RMS remains voiced.
    for frame_id in range(32, 40):
        held = decide(frame_id, hold_logits, voiced=True)
        assert held.action is ControlAction.LISTEN
        assert held.diagnostics["native_control6_effective_candidate"] == "LISTEN"
        assert held.diagnostics["native_control6_endpoint_speak_pending"] is True

    quiet_40 = decide(40, hold_logits, voiced=False)
    quiet_41 = decide(41, hold_logits, voiced=False)
    assert quiet_40.action is ControlAction.LISTEN
    assert quiet_40.diagnostics["native_control6_endpoint_speak_quiet_count"] == 1
    assert quiet_41.action is ControlAction.SPEAK
    assert quiet_41.diagnostics["native_control6_endpoint_speak_released"] is True
    assert quiet_41.diagnostics["native_control6_endpoint_speak_pending"] is False
    assert [frame for frame, decision in results.items() if decision.action is ControlAction.SPEAK] == [41]

    # The gate and diagnostic use the legal-normalized raw SPEAK member even
    # when the physical group candidate has moved to HOLD.
    assert quiet_40.diagnostics["native_control6_early_speak_confidence_source"] == (
        "legal_normalized_raw_speak_member"
    )
    assert quiet_40.diagnostics["native_control6_early_speak_confidence"] == pytest.approx(
        quiet_40.diagnostics["native_action_raw_probabilities"][1]
        / quiet_40.diagnostics["native_control6_legal_effective_mass"]
    )
    assert quiet_40.diagnostics["native_control6_early_speak_confidence"] != pytest.approx(
        quiet_40.diagnostics["native_control6_selected_effective_probability"]
    )


def test_endpoint_speak_memory_requires_consecutive_evidence_and_resets_quiet_run() -> None:
    high_replan = np.log([0.093, 0.001, 0.001, 0.001, 0.002, 0.902]).tolist()
    weak_replan = np.log([0.209, 0.001, 0.001, 0.001, 0.002, 0.786]).tolist()
    hold = np.log([0.88, 0.01, 0.02, 0.01, 0.02, 0.06]).tolist()

    def run(policy, frame_id: int, probabilities: list[float], *, voiced: bool,
            turn_id: int | None = 701, speaker_active: bool = False,
            session_id: str = "s1",
            legal: tuple[str, ...] = ("LISTEN", "SPEAK", "BACKCHANNEL")):
        logits = torch.tensor(probabilities, dtype=torch.float32)
        policy.adapter.fixed_logits.copy_(logits)
        state = _state(
            session_id=session_id,
            frame_id=frame_id,
            native_control_hidden_frame_id=frame_id,
            user_pcm=(
                np.full(1280, 0.05, dtype=np.float32)
                if voiced else np.zeros(1280, dtype=np.float32)
            ),
            peer_turn_active=True,
            speaker_active=speaker_active,
            user_turn_onset_frame_id=turn_id,
            legal_actions=legal,
        )
        return policy.step(state)

    def new_policy(*, early_threshold: float = 0.80):
        return GemmaNativeControl6Policy(
            FixedSixLogitAdapter([0.0] * len(ACTIONS)),
            checkpoint_path="cpu-test.pt",
            early_speak_confidence_threshold=early_threshold,
        )

    # One high voiced tick cannot prime the latch.
    one_peak = new_policy()
    run(one_peak, 10, high_replan, voiced=True)
    run(one_peak, 11, hold, voiced=True)
    assert run(one_peak, 12, hold, voiced=False).action is ControlAction.LISTEN
    assert run(one_peak, 13, hold, voiced=False).action is ControlAction.LISTEN

    # Two consecutive physical group scores below .80 do not qualify.
    below_gate = new_policy()
    for frame_id in (20, 21):
        decision = run(below_gate, frame_id, weak_replan, voiced=True)
        assert decision.diagnostics["native_control6_selected_effective_probability"] < 0.80
        assert decision.diagnostics["native_control6_endpoint_speak_pending"] is False
    assert run(below_gate, 22, hold, voiced=False).action is ControlAction.LISTEN
    assert run(below_gate, 23, hold, voiced=False).action is ControlAction.LISTEN

    # A voiced interruption resets the quiet counter but retains same-turn intent.
    resumed = new_policy()
    run(resumed, 30, high_replan, voiced=True)
    run(resumed, 31, np.log([0.123, 0.001, 0.001, 0.001, 0.002, 0.872]).tolist(), voiced=True)
    assert run(resumed, 32, hold, voiced=False).action is ControlAction.LISTEN
    interruption = run(resumed, 33, hold, voiced=True)
    assert interruption.action is ControlAction.LISTEN
    assert interruption.diagnostics["native_control6_endpoint_speak_pending"] is True
    assert interruption.diagnostics["native_control6_endpoint_speak_quiet_count"] == 0
    assert run(resumed, 34, hold, voiced=False).action is ControlAction.LISTEN
    assert run(resumed, 35, hold, voiced=False).action is ControlAction.SPEAK

    # Assistant activity, onset replacement/consumption, and missing identity
    # clear pending intent rather than carrying it into another user turn.
    for change in (
        {"speaker_active": True, "legal": ("CONTINUE", "YIELD")},
        {"turn_id": 702},
        {"turn_id": None},
    ):
        cleared = new_policy()
        run(cleared, 40, high_replan, voiced=True)
        run(cleared, 41, np.log([0.123, 0.001, 0.001, 0.001, 0.002, 0.872]).tolist(), voiced=True)
        cleared_state = run(cleared, 42, hold, voiced=False, **change)
        assert cleared_state.diagnostics["native_control6_endpoint_speak_pending"] is False
        if change.get("speaker_active"):
            assert cleared_state.action is ControlAction.YIELD
        else:
            assert cleared_state.action is ControlAction.LISTEN
        assert run(cleared, 43, hold, voiced=False, turn_id=change.get("turn_id", 701)).action is ControlAction.LISTEN

    reset = new_policy()
    run(reset, 60, high_replan, voiced=True)
    run(reset, 61, np.log([0.123, 0.001, 0.001, 0.001, 0.002, 0.872]).tolist(), voiced=True)
    reset.reset()
    assert run(reset, 62, hold, voiced=False).action is ControlAction.LISTEN
    assert run(reset, 63, hold, voiced=False).action is ControlAction.LISTEN

    changed_session = new_policy()
    run(changed_session, 70, high_replan, voiced=True)
    run(changed_session, 71, np.log([0.123, 0.001, 0.001, 0.001, 0.002, 0.872]).tolist(), voiced=True)
    assert run(changed_session, 72, hold, voiced=False, session_id="s2").action is ControlAction.LISTEN
    assert run(changed_session, 73, hold, voiced=False, session_id="s2").action is ControlAction.LISTEN

    for threshold in (0.0, -0.1):
        disabled = new_policy(early_threshold=threshold)
        run(disabled, 80, high_replan, voiced=True)
        disabled_second = run(
            disabled, 81,
            np.log([0.123, 0.001, 0.001, 0.001, 0.002, 0.872]).tolist(),
            voiced=True,
        )
        assert disabled_second.diagnostics["native_control6_endpoint_speak_pending"] is False


@pytest.mark.parametrize("entrypoint", ["step", "step_with_logits"])
def test_quiet_peer_active_replan_mass_cannot_authorize_early_speak(
    entrypoint: str,
) -> None:
    probabilities = [0.05, 0.02, 0.02, 0.02, 0.01, 0.88]
    logits = torch.tensor(np.log(probabilities), dtype=torch.float32)
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits.tolist()), checkpoint_path="cpu-test.pt",
    )
    state = _state(
        frame_id=90,
        native_control_hidden_frame_id=90,
        user_pcm=np.zeros(1280, dtype=np.float32),
        peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    )

    result = (
        policy.step(state)
        if entrypoint == "step"
        else policy.step_with_logits(
            state,
            logits,
            tick_id=(state.session_id, state.generation_id, state.frame_id),
        )
    )

    assert result.diagnostics["native_control6_user_rms"] == pytest.approx(0.0)
    assert result.diagnostics["native_control6_peer_turn_active"] is True
    assert result.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert result.diagnostics["native_control6_selected_effective_probability"] >= 0.85
    assert result.diagnostics["native_control6_early_speak_confidence"] < 0.80
    assert result.diagnostics["native_control6_early_speak_confidence_source"] == (
        "legal_normalized_raw_speak_member"
    )
    assert result.diagnostics["native_control6_early_speak_gated"] is True
    assert result.action is ControlAction.LISTEN


@pytest.mark.parametrize("entrypoint", ["step", "step_with_logits"])
def test_a4_trace_replay_quiet_replan_does_not_speak_at_frame_82(entrypoint: str) -> None:
    # Native six-way softmax rows copied from attempt5b/A4/dialogue_events.jsonl
    # frames 78-82. The failed live response was created at frame 82 from
    # REPLAN=.9839879 while the user's RMS was zero and peer_turn_active true.
    trace = (
        (78, [0.4003932476043701, 2.0055880668223836e-05, 0.5867068767547607,
              7.35462663215003e-06, 0.010782291181385517, 0.002090202644467354],
         0.04736872148209162),
        (79, [0.204070582985878, 3.255940717069592e-17, 0.3618409037590027,
              3.742718981958326e-11, 0.42867016792297363, 0.005418273154646158],
         0.05707593801545135),
        (80, [0.01166362501680851, 3.596337781408229e-17, 0.9501950740814209,
              5.919265457562245e-14, 0.0004853169957641512, 0.03765595704317093],
         0.02255858853781834),
        (81, [0.04275840148329735, 2.3215959710221057e-10, 0.17440778017044067,
              1.0119157733967654e-09, 0.0009166387026198208, 0.7819172143936157],
         0.0006588176554345364),
        (82, [0.00020835312898270786, 2.3008081551090243e-10, 0.015784282237291336,
              5.140303113715383e-11, 1.947237797139678e-05, 0.9839878678321838],
         0.0),
    )
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter([0.0] * len(ACTIONS)), checkpoint_path="cpu-test.pt",
    )
    decisions = []
    for frame_id, raw_probabilities, user_rms in trace:
        logits = torch.tensor(np.log(raw_probabilities), dtype=torch.float32)
        policy.adapter.fixed_logits.copy_(logits)
        state = _state(
            frame_id=frame_id,
            native_control_hidden_frame_id=frame_id,
            user_pcm=np.full(1280, user_rms, dtype=np.float32),
            peer_turn_active=True,
            user_turn_onset_frame_id=75,
            legal_actions=("LISTEN", "BACKCHANNEL", "SPEAK"),
        )
        decision = (
            policy.step(state)
            if entrypoint == "step"
            else policy.step_with_logits(
                state,
                logits,
                tick_id=(state.session_id, state.generation_id, state.frame_id),
            )
        )
        decisions.append(decision)

    assert [decision.frame_id for decision in decisions if decision.action is ControlAction.SPEAK] == []
    for decision in decisions[-2:]:
        assert decision.action is ControlAction.LISTEN
        assert decision.diagnostics["native_control6_early_speak_confidence"] < 0.80
        assert decision.diagnostics["native_control6_early_speak_confidence_source"] == (
            "legal_normalized_raw_speak_member"
        )


@pytest.mark.parametrize("entrypoint", ["step", "step_with_logits"])
def test_quiet_peer_active_raw_speak_can_still_clear_early_gate(entrypoint: str) -> None:
    probabilities = [0.02, 0.88, 0.02, 0.02, 0.02, 0.04]
    logits = torch.tensor(np.log(probabilities), dtype=torch.float32)
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits.tolist()), checkpoint_path="cpu-test.pt",
    )
    state = _state(
        frame_id=92,
        native_control_hidden_frame_id=92,
        user_pcm=np.zeros(1280, dtype=np.float32),
        peer_turn_active=True,
        legal_actions=("LISTEN", "SPEAK"),
    )
    result = (
        policy.step(state)
        if entrypoint == "step"
        else policy.step_with_logits(
            state,
            logits,
            tick_id=(state.session_id, state.generation_id, state.frame_id),
        )
    )
    assert result.diagnostics["native_control6_early_speak_confidence_source"] == (
        "legal_normalized_raw_speak_member"
    )
    assert result.diagnostics["native_control6_early_speak_confidence"] >= 0.80
    assert result.diagnostics["native_control6_early_speak_gated"] is False
    assert result.action is ControlAction.SPEAK


@pytest.mark.parametrize("entrypoint", ["step", "step_with_logits"])
def test_active_user_raw_speak_probability_can_clear_early_gate(
    entrypoint: str,
) -> None:
    probabilities = [0.02, 0.88, 0.02, 0.02, 0.02, 0.04]
    logits = torch.tensor(np.log(probabilities), dtype=torch.float32)
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(logits.tolist()), checkpoint_path="cpu-test.pt",
    )
    state = _state(
        frame_id=80,
        native_control_hidden_frame_id=80,
        user_pcm=np.full(1280, 0.05, dtype=np.float32),
        peer_turn_active=False,
        legal_actions=("LISTEN", "SPEAK"),
    )

    if entrypoint == "step":
        result = policy.step(state)
    else:
        result = policy.step_with_logits(
            state,
            logits,
            tick_id=(state.session_id, state.generation_id, state.frame_id),
        )

    assert result.diagnostics["native_control6_user_rms"] == pytest.approx(0.05)
    assert result.diagnostics["native_control6_early_speak_confidence"] >= 0.80
    assert result.diagnostics["native_control6_early_speak_gated"] is False
    assert result.action is ControlAction.SPEAK


def test_peer_gate_hysteresis_tracks_physical_candidate_not_raw_hold_or_old_streak() -> None:
    # The raw LISTEN row wins, but SPEAK+REPLAN is the larger legal physical
    # group. Hysteresis must track that grouped candidate, not raw argmax.
    speak_probabilities = [0.30, 0.25, 0.05, 0.15, 0.05, 0.20]
    policy = GemmaNativeControl6Policy(
        FixedSixLogitAdapter(np.log(speak_probabilities).tolist()),
        checkpoint_path="cpu-test.pt",
        minimum_transition_confidence=0.50,
    )
    state = lambda frame_id: _state(
        frame_id=frame_id, native_control_hidden_frame_id=frame_id,
        peer_turn_active=True, legal_actions=("LISTEN", "SPEAK"),
    )

    first = policy.step(state(60))
    duplicate = policy.step(state(60))
    second = policy.step(state(61))
    assert first.diagnostics["base_action"] == "LISTEN"
    assert first.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert first.diagnostics["native_control6_hysteresis_count"] == 1
    assert duplicate.diagnostics["native_control6_hysteresis_count"] == 1
    assert second.action is ControlAction.LISTEN
    assert second.diagnostics["native_control6_hysteresis_count"] == 2
    assert second.diagnostics["native_control6_hysteresis_applied_action"] == "SPEAK"
    assert second.diagnostics["native_control6_early_speak_gated"] is True

    # A fresh physical HOLD candidate clears the SPEAK streak; a later SPEAK
    # candidate starts again at one even though the raw row is still LISTEN.
    hold_probabilities = [0.40, 0.15, 0.10, 0.10, 0.10, 0.15]
    policy.adapter.fixed_logits.copy_(torch.tensor(np.log(hold_probabilities), dtype=torch.float32))
    hold = policy.step(state(62))
    assert hold.diagnostics["native_control6_effective_candidate"] == "LISTEN"
    assert hold.diagnostics["native_control6_hysteresis_count"] == 0
    policy.adapter.fixed_logits.copy_(torch.tensor(np.log(speak_probabilities), dtype=torch.float32))
    restarted = policy.step(state(63))
    assert restarted.diagnostics["base_action"] == "LISTEN"
    assert restarted.diagnostics["native_control6_effective_candidate"] == "SPEAK"
    assert restarted.diagnostics["native_control6_hysteresis_count"] == 1
    assert restarted.action is ControlAction.LISTEN


def test_speak_is_legal_after_endpoint_without_quiet_frame_or_silence0_gate() -> None:
    # Scheduler legality does not accept quiet-frame or silence0 state.
    legal = legal_control_actions(
        active_response=False, response_pending=False, user_turn_pending=True,
        peer_turn_active=False, backchannel_available=False,
        renderer_plan_pending=False,
    )
    assert "SPEAK" in legal
