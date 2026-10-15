# -*- coding: utf-8 -*-
"""tests/test_dialogue_evaluator.py

Unit tests for the dialogue evaluation suite:
1. WER computation.
2. Physical audio metrics (clipping, RMS, crest factor, anomalies).
3. Deterministic fact checkers (name/number, contradiction, clarification, closing).
4. Full turn evaluation and markdown report generation.
5. Synthetic speech PCM generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from dialogue_eval import (
    ALL_SCENARIOS,
    SCENARIO_NAME_NUMBER,
    SCENARIO_CONTRADICTION,
    SCENARIO_BARGE_IN,
    DialogueEvaluator,
    calculate_wer,
    compute_physical_audio_metrics,
    generate_synthetic_speech_pcm,
)


def test_conversation_evidence_requires_paced_bridge_for_software_duplex() -> None:
    from dialogue_eval.evaluator import classify_conversation_evidence

    transport = dict(transport_complete=True, generated_facts_passed=True, independent_asr_available=True, audio_facts_passed=True)
    assert classify_conversation_evidence(**{**transport, "independent_asr_available": False}) == "WS_TRANSPORT_EVIDENCE_COMPLETE"
    assert classify_conversation_evidence(**{**transport, "audio_facts_passed": False}) == "WS_TRANSPORT_EVIDENCE_COMPLETE"
    assert classify_conversation_evidence(**{**transport, "paced_bridge_duplex_observed": True}) == "FULL_DUPLEX_AUDIO_VERIFIED"
    assert classify_conversation_evidence(**transport) == "WS_TRANSPORT_AND_OUTPUT_CONTENT_VERIFIED"
    assert classify_conversation_evidence(**{**transport, "transport_complete": False}) == "INCOMPLETE_EVIDENCE"


def test_italian_spoken_digit_sequence_matches_without_accepting_wrong_number() -> None:
    from dialogue_eval.evaluator import contains_italian_digit_sequence

    assert contains_italian_digit_sequence("Marco, quattro, otto, due, nove", "4829")
    assert contains_italian_digit_sequence("Marco codice 4, 8, 2, 9", "4829")
    assert not contains_italian_digit_sequence("Marco a 4.029", "4829")
    assert not contains_italian_digit_sequence("Marco quattro, zero, due, nove", "4829")


def test_barge_in_fact_check_requires_expected_population_not_generic_abitanti() -> None:
    step = next(step for step in SCENARIO_BARGE_IN.steps if step.step_id == "turn_2_assistant_recovered")
    wrong_population = "Oggi Venezia conta circa 250.000 abitanti."
    correct_population = "Oggi Venezia conta circa cinquantamila abitanti."

    assert not all(check(wrong_population) for check in step.deterministic_checks)
    assert all(check(correct_population) for check in step.deterministic_checks)


def test_turn_taking_audio_certificate_requires_alternation_rms_and_independent_asr() -> None:
    from dialogue_eval.evaluator import classify_turn_taking_audio_evidence

    args = {
        "transport_complete": True,
        "turn_sequence": ["A", "B", "A"],
        "turn_transfers": [
            {"from_speaker": "A", "to_speaker": "B", "frames_sent": 3, "frames_acknowledged": 3, "commit_accepted": True},
            {"from_speaker": "B", "to_speaker": "A", "frames_sent": 2, "frames_acknowledged": 2, "commit_accepted": True},
        ],
        "output_rms_by_speaker": {"A": 0.12, "B": 0.09},
        "output_asr_by_speaker": {"A": "codice 4829", "B": "confermo il codice 4829"},
        "independent_asr_engine": "faster-whisper",
        "generated_facts_passed": True,
        "audio_facts_passed": True,
    }
    evidence = classify_turn_taking_audio_evidence(**args)
    assert evidence["status"] == "TURN_TAKING_AUDIO_VERIFIED"
    assert evidence["alternating_turns"] is True
    assert evidence["positive_rms_all_speakers"] is True
    assert evidence["independent_asr_all_speakers"] is True

    no_asr = classify_turn_taking_audio_evidence(**{**args, "output_asr_by_speaker": {"A": "", "B": "confermo"}})
    assert no_asr["status"] == "INCOMPLETE_EVIDENCE"
    no_rms = classify_turn_taking_audio_evidence(**{**args, "output_rms_by_speaker": {"A": 0.0, "B": 0.09}})
    assert no_rms["status"] == "INCOMPLETE_EVIDENCE"
    non_alternating = classify_turn_taking_audio_evidence(**{**args, "turn_sequence": ["A", "A", "B"]})
    assert non_alternating["status"] == "INCOMPLETE_EVIDENCE"


def test_calculate_wer() -> None:
    assert calculate_wer("ciao mondo", "ciao mondo") == 0.0
    assert calculate_wer("ciao mondo", "ciao terra") == 0.5
    assert calculate_wer("ciao mondo", "") == 1.0
    assert calculate_wer("", "ciao mondo") == 1.0
    assert calculate_wer("", "") == 0.0


def test_physical_audio_metrics() -> None:
    # 1. Clean tone without clipping
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    metrics = compute_physical_audio_metrics(tone, sample_rate=sample_rate)
    assert metrics["finite"] is True
    assert metrics["clipping_pct"] == 0.0
    assert 0.35 <= metrics["rms"] <= 0.36
    # Sine wave crest factor is sqrt(2) ~ 1.414
    assert 1.4 <= metrics["crest_factor"] <= 1.43

    # 2. Saturated signal with clipping
    clipped_tone = np.clip(tone * 10.0, -1.0, 1.0)
    metrics_clipped = compute_physical_audio_metrics(clipped_tone, sample_rate=sample_rate)
    assert metrics_clipped["clipping_pct"] > 5.0

    # 3. Discontinuity check
    discontinuous_signal = np.array([0.0, 0.0, 0.0, 0.9, 0.9], dtype=np.float32)
    metrics_disc = compute_physical_audio_metrics(
        discontinuous_signal, sample_rate=sample_rate, packet_boundary_indices=[3]
    )
    assert metrics_disc["packet_discontinuities_count"] == 1
    assert metrics_disc["max_discontinuity_jump"] == 0.9


def test_synthetic_speech_generation() -> None:
    speech = generate_synthetic_speech_pcm(1.0, sample_rate=16000)
    assert speech.size == 16000
    assert speech.dtype == np.float32
    assert np.isfinite(speech).all()

    metrics = compute_physical_audio_metrics(speech, sample_rate=16000)
    assert metrics["clipping_pct"] == 0.0
    assert metrics["rms"] > 0.05
    # Modulated speech waveform should have natural crest factor > 3.0
    assert metrics["crest_factor"] >= 3.0


def test_dialogue_evaluator_fact_and_contradiction_checks() -> None:
    evaluator = DialogueEvaluator()

    # Scenario 1: Name & number memory
    scen1 = SCENARIO_NAME_NUMBER
    # User turn 1
    evaluator.evaluate_turn(
        turn_index=1,
        speaker="user",
        scenario=scen1,
        step=scen1.steps[0],
        recognized_input=scen1.steps[0].expected_text,
        generated_text="",
        recognized_output="",
    )
    # Assistant turn 1 (passing response)
    evaluator.evaluate_turn(
        turn_index=2,
        speaker="assistant",
        scenario=scen1,
        step=scen1.steps[1],
        recognized_input="",
        generated_text="Certamente Marco, ho salvato il codice ordine 4829.",
        recognized_output="Certamente Marco, ho salvato il codice ordine 4829.",
        timing_metrics={"ttfa_ms": 450.0},
    )

    # Scenario 4: Deliberate contradiction
    scen4 = SCENARIO_CONTRADICTION
    # Assistant turn catching contradiction
    evaluator.evaluate_turn(
        turn_index=4,
        speaker="assistant",
        scenario=scen4,
        step=scen4.steps[3],
        recognized_input="",
        generated_text="No, in realtà prima hai detto di chiamarti Marco e non Roberto, codice 4829.",
        recognized_output="No, in realtà prima hai detto di chiamarti Marco e non Roberto, codice 4829.",
        timing_metrics={"ttfa_ms": 500.0},
    )

    summary = evaluator.build_summary_report()
    assert summary["total_turns"] == 3
    assert summary["passed_turns"] == 3
    assert summary["all_facts_retained"] is True
    assert summary["all_contradictions_resolved"] is True

    # Test report markdown formatting
    md = evaluator.generate_markdown_report()
    assert "# Report di Valutazione Conversazione Full-Duplex" in md
    assert "Marco" in md
    assert "4829" in md
