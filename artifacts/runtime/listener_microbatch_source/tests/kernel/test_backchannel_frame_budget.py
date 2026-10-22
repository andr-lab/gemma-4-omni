from __future__ import annotations

from duplex_kernel.engine import _estimate_talker_frame_budget


def test_one_word_backchannel_has_short_minimum_and_word_tolerant_ceiling() -> None:
    spoken_units, numeric_extra_units, frame_budget = _estimate_talker_frame_budget(
        "Capito.", 1, 2, short_streaming=True
    )

    assert spoken_units == 1
    assert numeric_extra_units == 0
    assert frame_budget == 20  # int(1 * 4.8) + 16; 80 ms per RVQ frame


def test_backchannel_frame_budget_is_capped_at_32_frames() -> None:
    _, _, frame_budget = _estimate_talker_frame_budget(
        "Una risposta breve completa.", 4, 2, short_streaming=True
    )

    assert frame_budget == 32


def test_backchannel_ceiling_wins_over_an_invalid_larger_floor() -> None:
    _, _, frame_budget = _estimate_talker_frame_budget(
        "Capito.", 1, 48, short_streaming=True
    )

    assert frame_budget == 32
