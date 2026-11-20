from __future__ import annotations

import math

import numpy as np

from tools.evaluate_native_control6_phase_sweeps import (
    control6_phase0_grid,
    control6_phase3_grid,
    speaker_active_effective_delta,
)


def test_phase_sweep_grids_cover_native_clip_and_activity_boundaries() -> None:
    phase0 = control6_phase0_grid(max_observed=0.0)
    phase3 = control6_phase3_grid()

    base_grid = np.linspace(-6.0, -3.0, 301)
    assert all(np.isclose(phase0, value, rtol=0.0, atol=1e-12).any() for value in base_grid)
    for boundary in (-6.0, -3.0):
        assert any(np.isclose(phase0, boundary + offset, rtol=0.0, atol=1e-12).any()
                   for offset in (-1e-6, 0.0, 1e-6))
    assert math.isclose(float(phase0[-1]), 0.0)
    assert phase3.shape[0] == 104
    assert phase3[0] == -6.0 and phase3[-1] == 6.0
    assert any(np.isclose(phase3, -0.15 + offset, rtol=0.0, atol=1e-12).any()
               for offset in (-1e-6, 0.0, 1e-6))


def test_speaker_active_policy_delta_uses_runtime_activity_and_continuous_weight() -> None:
    raw = {
        "LISTEN": 0.2,
        "SPEAK": 0.2,
        "CONTINUE": 0.3,
        "YIELD": 0.1,
        "BACKCHANNEL": 0.1,
        "REPLAN": 0.1,
    }

    delta, weight, activity = speaker_active_effective_delta(
        raw,
        user_rms=0.0001,
        assistant_log_rms=-2.5,
        user_assistant_log_ratio=-2.0,
        ratio_valid=False,
        peer_active=False,
    )

    assert activity is False
    assert math.isclose(weight, 0.1, abs_tol=1e-12)
    assert math.isclose(delta, (0.22 - 0.48) / 0.70, abs_tol=1e-12)

    _, ratio_weight, ratio_activity = speaker_active_effective_delta(
        raw,
        user_rms=0.0001,
        assistant_log_rms=-2.5,
        user_assistant_log_ratio=-0.15,
        ratio_valid=True,
        peer_active=False,
    )
    assert ratio_activity is False  # Runtime activity uses a strict > -0.15 comparison.
    assert math.isclose(ratio_weight, 1.0, abs_tol=1e-12)


def test_monotone_weight_does_not_guarantee_monotone_delta_when_logits_change() -> None:
    # These two valid six-action distributions isolate the assumption under
    # discussion: the weight rises continuously, but the model's raw logits
    # also change between frames, so the effective YIELD-minus-HOLD can fall.
    first_raw = {
        "LISTEN": 0.2,
        "SPEAK": 0.0,
        "CONTINUE": 0.2,
        "YIELD": 0.45,
        "BACKCHANNEL": 0.0,
        "REPLAN": 0.15,
    }
    second_raw = {
        "LISTEN": 0.2,
        "SPEAK": 0.0,
        "CONTINUE": 0.4,
        "YIELD": 0.3,
        "BACKCHANNEL": 0.0,
        "REPLAN": 0.1,
    }

    first_delta, first_weight, first_activity = speaker_active_effective_delta(
        first_raw,
        user_rms=0.0001,
        assistant_log_rms=-2.5,
        user_assistant_log_ratio=-2.0,
        ratio_valid=False,
        peer_active=False,
    )
    second_delta, second_weight, second_activity = speaker_active_effective_delta(
        second_raw,
        user_rms=0.0002,
        assistant_log_rms=-2.5,
        user_assistant_log_ratio=-2.0,
        ratio_valid=False,
        peer_active=False,
    )

    assert not first_activity and not second_activity
    assert math.isclose(first_weight, 0.1, abs_tol=1e-12)
    assert math.isclose(second_weight, 0.2, abs_tol=1e-12)
    assert math.isclose(first_delta, 0.24, abs_tol=1e-12)
    assert math.isclose(second_delta, -0.12, abs_tol=1e-12)
    assert second_delta - first_delta < 0.0

