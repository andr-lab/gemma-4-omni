from __future__ import annotations

import numpy as np
import pytest
import torch

from duplex_kernel.turn_policy import (
    ControlAction,
    ControlState,
    PolicyInput,
    ProvisionalFramePolicy,
    SpeakerGate,
)
from duplex_kernel.turnhead import TurnHeadRuntime
from duplex_kernel.turnhead_adapter import TurnHeadPolicy


FRAME_SAMPLES = 1280


class _StubRuntime:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.has_thinker_hidden = False
        self.loaded = False
        self.raw: dict[str, object] = {}
        self.hidden_calls: list[object] = []
        self.pcm_calls: list[tuple[np.ndarray, bool]] = []

    def load(self, checkpoint: str) -> bool:
        self.loaded = checkpoint == "stub"
        return self.loaded

    def set_thinker_hidden(self, hidden: object) -> None:
        self.hidden_calls.append(hidden)
        shape = getattr(hidden, "shape", ())
        self.has_thinker_hidden = bool(shape and shape[-1] == 2048)

    def feed_pcm(self, pcm: np.ndarray, *, finalize_frame: bool = False) -> int:
        self.pcm_calls.append((np.asarray(pcm), finalize_frame))
        return 8

    def decide(self) -> dict[str, object]:
        return self.raw

    def reset(self) -> None:
        self.has_thinker_hidden = False


def _state(
    *, active: bool = False, hidden: object | None = None, frame_id: int = 4,
    pcm: np.ndarray | None = None,
) -> PolicyInput:
    return PolicyInput(
        session_id="policy-test",
        frame_id=frame_id,
        user_pcm=np.full(FRAME_SAMPLES, 0.01, dtype=np.float32) if pcm is None else pcm,
        speaker_active=active,
        generation_id=9,
        listener_hidden=hidden,
    )


def _policy(monkeypatch: pytest.MonkeyPatch, name: str, probability: float = 0.9) -> TurnHeadPolicy:
    monkeypatch.setattr("duplex_kernel.turnhead_adapter.TurnHeadRuntime", _StubRuntime)
    policy = TurnHeadPolicy(device="cpu", checkpoint="stub")
    policy.runtime.raw = {
        "pred_name": name,
        "probs": [probability, 0.02, 0.02, 0.02, 0.02],
        "active": True,
    }
    return policy


@pytest.mark.parametrize(
    ("name", "active", "expected"),
    [
        ("SPEAK", False, ControlAction.SPEAK),
        ("SPEAK", True, ControlAction.CONTINUE),
        ("BARGE_IN", True, ControlAction.HARD_STOP),
        ("BARGE_IN", False, ControlAction.LISTEN),
        ("YIELD", True, ControlAction.YIELD),
        ("YIELD", False, ControlAction.LISTEN),
        ("USER_BACKCHANNEL", True, ControlAction.CONTINUE),
        ("USER_BACKCHANNEL", False, ControlAction.LISTEN),
        ("EMIT_BACKCHANNEL", False, ControlAction.BACKCHANNEL),
        ("EMIT_BACKCHANNEL", True, ControlAction.CONTINUE),
    ],
)
def test_turnhead_action_mapping_is_state_aware(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    active: bool,
    expected: ControlAction,
) -> None:
    policy = _policy(monkeypatch, name)

    decision = policy.step(_state(active=active, hidden=torch.zeros(1, 1, 2048)))

    assert decision.action is expected
    assert decision.frame_id == 4
    assert decision.generation_id == 9


@pytest.mark.parametrize("name", ["BARGE_IN", "YIELD"])
def test_turnhead_never_stops_without_current_v9_hidden(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    policy = _policy(monkeypatch, name)

    decision = policy.step(_state(active=True, hidden=None))

    assert decision.action is ControlAction.CONTINUE
    assert "missing_2048_hidden" in decision.reason


def test_turnhead_never_speaks_from_missing_or_wrong_width_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK")

    missing = policy.step(_state(hidden=None))
    wrong_width = policy.step(_state(hidden=torch.zeros(1, 1, 1536), frame_id=5))

    assert missing.action is ControlAction.LISTEN
    assert wrong_width.action is ControlAction.LISTEN
    assert policy.runtime.has_thinker_hidden is False
    assert "hidden" in missing.reason


def test_missing_current_hidden_cannot_reuse_a_previous_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK")
    hidden = torch.zeros(1, 1, 2048)

    assert policy.step(_state(hidden=hidden)).action is ControlAction.SPEAK
    decision = policy.step(_state(hidden=None, frame_id=5))

    assert decision.action is ControlAction.LISTEN
    assert "missing_2048_hidden" in decision.reason
    assert policy.runtime.has_thinker_hidden is False


def test_turnhead_low_confidence_speak_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK", probability=0.384)
    policy.runtime.raw["probs"] = [0.384, 0.006, 0.271, 0.002, 0.337]

    decision = policy.step(_state(hidden=torch.zeros(1, 1, 2048)))

    assert decision.action is ControlAction.LISTEN
    assert "confidence" in decision.reason


def test_turnhead_silent_pcm_cannot_trigger_speak_with_valid_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK")

    decision = policy.step(
        _state(hidden=torch.zeros(1, 1, 2048), pcm=np.zeros(FRAME_SAMPLES, dtype=np.float32))
    )

    assert decision.action is ControlAction.LISTEN
    assert "silent_input_only_this_turn" in decision.reason


def test_turnhead_can_speak_on_end_silence_after_meaningful_input(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK")
    hidden = torch.zeros(1, 1, 2048)
    speech = np.full(FRAME_SAMPLES, 0.01, dtype=np.float32)
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    first = policy.step(_state(hidden=hidden, pcm=speech))
    end_of_turn = policy.step(_state(hidden=hidden, frame_id=5, pcm=silence))

    assert first.action is ControlAction.SPEAK
    assert end_of_turn.action is ControlAction.SPEAK


def test_turnhead_low_level_nonzero_input_is_not_discarded_by_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK")
    quiet_speech = np.full(FRAME_SAMPLES, 1e-6, dtype=np.float32)

    decision = policy.step(_state(hidden=torch.zeros(1, 1, 2048), pcm=quiet_speech))

    assert decision.action is ControlAction.SPEAK


def test_turnhead_reset_clears_meaningful_input_for_next_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(monkeypatch, "SPEAK")
    hidden = torch.zeros(1, 1, 2048)

    assert policy.step(_state(hidden=hidden)).action is ControlAction.SPEAK
    policy.reset()
    decision = policy.step(_state(hidden=hidden, frame_id=5, pcm=np.zeros(FRAME_SAMPLES, dtype=np.float32)))

    assert decision.action is ControlAction.LISTEN
    assert "silent_input_only_this_turn" in decision.reason


def test_unloaded_turnhead_fails_closed_and_provisional_does_not_synthesize() -> None:
    policy = TurnHeadPolicy(device="cpu", checkpoint=None)
    provisional = ProvisionalFramePolicy(window_frames=2)
    hidden = torch.zeros(1, 1, 2048)

    assert policy.step(_state(hidden=hidden)).action is ControlAction.LISTEN
    for frame_id in range(5):
        state = _state(frame_id=frame_id)
        assert provisional.step(state).action is ControlAction.LISTEN
    assert provisional.step(_state(active=True, frame_id=6)).action is ControlAction.CONTINUE


def test_turnhead_runtime_commits_eight_mels_per_80ms_and_bounds_pcm() -> None:
    runtime = TurnHeadRuntime(device="cpu")
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    assert runtime.feed_pcm(silence, finalize_frame=True) == 8
    assert runtime._mel_buffer.shape == (8, 128)  # noqa: SLF001
    assert runtime.feed_pcm(silence, finalize_frame=True) == 8
    assert runtime._mel_buffer.shape == (16, 128)  # noqa: SLF001
    assert runtime._next_mel_frame == 16  # noqa: SLF001
    assert runtime._pcm_buffer.size <= 320  # noqa: SLF001


def test_turnhead_runtime_supplies_rolling_three_tick_context_to_causal_head() -> None:
    class RecordingHead:
        in_dim = 2304

        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def __call__(self, features: torch.Tensor) -> torch.Tensor:
            self.inputs.append(features.detach().clone())
            return torch.zeros((*features.shape[:-1], 5), dtype=features.dtype, device=features.device)

    runtime = TurnHeadRuntime(device="cpu")
    runtime.head = RecordingHead()  # type: ignore[assignment]
    runtime._loaded = True  # noqa: SLF001

    for frame_id in range(5):
        runtime.set_thinker_hidden(torch.full((1, 1, 2048), float(frame_id)))
        runtime.decide()

    head = runtime.head
    assert isinstance(head, RecordingHead)
    assert [tuple(value.shape) for value in head.inputs] == [
        (1, 1, 2304),
        (1, 2, 2304),
        (1, 3, 2304),
        (1, 3, 2304),
        (1, 3, 2304),
    ]
    assert torch.equal(head.inputs[-1][0, :, 0], torch.tensor([2.0, 3.0, 4.0]))

    runtime.reset()
    runtime.set_thinker_hidden(torch.full((1, 1, 2048), 9.0))
    runtime.decide()
    assert tuple(head.inputs[-1].shape) == (1, 1, 2304)
    assert torch.equal(head.inputs[-1][0, :, 0], torch.tensor([9.0]))


def test_checkpoint_with_listener_width_is_rejected_before_partial_load(tmp_path) -> None:
    path = tmp_path / "wrong-width.pt"
    torch.save({"in_dim": 1536, "model_state_dict": {}}, path)
    runtime = TurnHeadRuntime(device="cpu")

    assert runtime.load(str(path)) is False
    assert runtime.head is None
    assert runtime.loaded is False
    assert "in_dim" in runtime.load_error


def test_policy_actions_have_expected_speaker_gate_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    class Session:
        def __init__(self, control_state: ControlState = ControlState.LISTENING) -> None:
            self.control_state = control_state
            self.session_generation_id = 9
            self.speaker_segment_id = None
            self.stop_requested = False
            self.stop_reason = None

        def open_control_segment(self, action: str) -> None:
            self.speaker_segment_id = action

        def request_control_segment_close(self) -> None:
            pass

        def invalidate_generation(self, _reason: str) -> None:
            self.session_generation_id += 1

    hidden = torch.zeros(1, 1, 2048)
    speak_policy = _policy(monkeypatch, "SPEAK")
    speak = speak_policy.step(_state(hidden=hidden))
    speak_event = SpeakerGate().apply(speak, Session())
    assert speak_event.accepted and speak_event.open_segment
    assert speak_event.next_state is ControlState.SPEAKING

    barge_policy = _policy(monkeypatch, "BARGE_IN")
    barge = barge_policy.step(_state(active=True, hidden=hidden))
    barge_event = SpeakerGate().apply(barge, Session(ControlState.SPEAKING))
    assert barge_event.stop_synthesis and barge_event.flush_audio_buffer
    assert barge_event.generation_id == 10

    backchannel_policy = _policy(monkeypatch, "EMIT_BACKCHANNEL")
    backchannel = backchannel_policy.step(_state(hidden=hidden))
    backchannel_event = SpeakerGate().apply(backchannel, Session())
    assert backchannel_event.trigger_backchannel and backchannel_event.open_segment

    silent_without_context = _policy(monkeypatch, "SPEAK").step(_state(hidden=None))
    silence_event = SpeakerGate().apply(silent_without_context, Session())
    assert silence_event.action is ControlAction.LISTEN
    assert not silence_event.open_segment and not silence_event.trigger_backchannel
