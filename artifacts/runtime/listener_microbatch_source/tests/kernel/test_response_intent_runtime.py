import torch
import pytest

from duplex_kernel.response_intent import (
    RESPONSE_INTENT_PREFIX_CAPACITY,
    ResponseIntentProjector,
    build_response_intent_prefix,
    pool_response_intent,
    response_intent_frame_budget,
    response_intent_termination,
)


def test_runtime_prefix_uses_only_generated_response_states_and_repeats_to_256():
    hidden = torch.zeros(1, 3, 1536)
    hidden[0, 0] = 1.0
    hidden[0, 1] = 3.0
    hidden[0, 2] = 5.0
    projector = ResponseIntentProjector()
    with torch.no_grad():
        projector.linear.weight.zero_()
        projector.linear.bias.zero_()
        projector.linear.weight[0, 0] = 1.0

    intent, prefix = build_response_intent_prefix(hidden, projector)

    assert intent.shape == (1, 1536)
    torch.testing.assert_close(intent[0, 0], torch.tensor(3.5))
    assert prefix.shape == (1, RESPONSE_INTENT_PREFIX_CAPACITY, 2048)
    assert torch.all(prefix[0, :, 0] == 3.5)
    assert torch.count_nonzero(prefix[0, :, 1:]) == 0


def test_response_intent_rejects_empty_or_wrong_width_hiddens():
    with pytest.raises(ValueError, match="response_hidden"):
        pool_response_intent(torch.empty(1, 0, 1536))
    with pytest.raises(ValueError, match="response_hidden"):
        pool_response_intent(torch.zeros(1, 2, 2048))


def test_safety_capacity_is_not_reported_as_natural_eos():
    assert response_intent_termination(
        eos_detected=True, cancelled=False, generated_frames=256
    ) == "eos"
    assert response_intent_termination(
        eos_detected=False, cancelled=True, generated_frames=256
    ) == "cancel"
    assert response_intent_termination(
        eos_detected=False, cancelled=False, generated_frames=256
    ) == "safety_cap"
    assert response_intent_termination(
        eos_detected=False, cancelled=False, generated_frames=255
    ) == "incomplete"


def test_runtime_speak_uses_safety_capacity_while_backchannel_stays_bounded():
    assert response_intent_frame_budget(ordinary_budget=83, is_backchannel=False) == 256
    assert response_intent_frame_budget(ordinary_budget=52, is_backchannel=True) == 32
    assert response_intent_frame_budget(ordinary_budget=19, is_backchannel=True) == 19
