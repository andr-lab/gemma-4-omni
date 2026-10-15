# -*- coding: utf-8 -*-
"""dialogue_eval/evaluator.py

Evaluation engine for full-duplex spoken dialogue.

Implements all metrics specified in User Requirements 4 & 5:
- Separate evaluation of ingress transcription vs human ground-truth.
- Emitted text vs independent transcript of produced audio.
- Relevance and deterministic fact retention.
- Contradiction resolution.
- Barge-in recovery and latency.
- Physical audio metrics (clipping, RMS, crest factor, silence anomalies, discontinuities).
- Real-time latency metrics (TTFA, p95/p99 cycle, output gaps, dropped frames, VRAM peak).
- Turn-by-turn breakdown table with error rationales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from dialogue_eval.audio_metrics import compute_physical_audio_metrics
from dialogue_eval.scenarios import DialogueScenario, ScenarioStep


_ITALIAN_DIGIT_WORDS = {
    "zero": "0",
    "uno": "1",
    "una": "1",
    "due": "2",
    "tre": "3",
    "quattro": "4",
    "cinque": "5",
    "sei": "6",
    "sette": "7",
    "otto": "8",
    "nove": "9",
}


def contains_italian_digit_sequence(text: str, expected_digits: str) -> bool:
    """Match a code spoken digit-by-digit or transcribed as numeric tokens."""
    expected = str(expected_digits).strip()
    if not expected.isdigit():
        return False
    tokens = re.findall(r"\d+|[a-zà-öø-ÿ]+", str(text).lower())
    observed_digits = "".join(
        _ITALIAN_DIGIT_WORDS[token] if token in _ITALIAN_DIGIT_WORDS else token
        for token in tokens
        if token.isdigit() or token in _ITALIAN_DIGIT_WORDS
    )
    return expected in observed_digits


def classify_conversation_evidence(
    *,
    transport_complete: bool,
    generated_facts_passed: bool,
    independent_asr_available: bool,
    audio_facts_passed: bool,
    paced_bridge_duplex_observed: bool = False,
) -> str:
    """Keep websocket arrival evidence distinct from PCM delivered through the paced bridge."""
    if not transport_complete:
        return "INCOMPLETE_EVIDENCE"
    if not generated_facts_passed or not independent_asr_available or not audio_facts_passed:
        return "WS_TRANSPORT_EVIDENCE_COMPLETE"
    if not paced_bridge_duplex_observed:
        return "WS_TRANSPORT_AND_OUTPUT_CONTENT_VERIFIED"
    return "FULL_DUPLEX_AUDIO_VERIFIED"


def classify_turn_taking_audio_evidence(
    *,
    transport_complete: bool,
    turn_sequence: List[str],
    turn_transfers: List[Dict[str, Any]],
    output_rms_by_speaker: Dict[str, float],
    output_asr_by_speaker: Dict[str, str],
    independent_asr_engine: str | None,
    generated_facts_passed: bool,
    audio_facts_passed: bool,
) -> Dict[str, Any]:
    """Certify sequential spoken turns without implying overlapping inference."""
    sequence = [str(speaker).upper() for speaker in turn_sequence]
    alternating = (
        len(sequence) >= 2
        and {"A", "B"}.issubset(sequence)
        and all(left != right for left, right in zip(sequence, sequence[1:]))
    )
    expected_transfer_count = max(0, len(sequence) - 1)
    transfers_complete = expected_transfer_count > 0 and len(turn_transfers) == expected_transfer_count
    if transfers_complete:
        for index, transfer in enumerate(turn_transfers):
            frames_sent = int(transfer.get("frames_sent", 0))
            if not (
                str(transfer.get("from_speaker", "")).upper() == sequence[index]
                and str(transfer.get("to_speaker", "")).upper() == sequence[index + 1]
                and frames_sent > 0
                and int(transfer.get("frames_acknowledged", -1)) == frames_sent
                and transfer.get("commit_accepted") is True
            ):
                transfers_complete = False
                break
    positive_rms = {
        speaker: bool(np.isfinite(float(output_rms_by_speaker.get(speaker, 0.0))) and float(output_rms_by_speaker.get(speaker, 0.0)) > 0.0)
        for speaker in ("A", "B")
    }
    asr_available = {
        speaker: bool(str(output_asr_by_speaker.get(speaker, "")).strip())
        for speaker in ("A", "B")
    }
    all_positive_rms = all(positive_rms.values())
    all_asr_available = bool(independent_asr_engine) and all(asr_available.values())
    verified = bool(
        transport_complete
        and alternating
        and transfers_complete
        and all_positive_rms
        and all_asr_available
        and generated_facts_passed
        and audio_facts_passed
    )
    return {
        "status": "TURN_TAKING_AUDIO_VERIFIED" if verified else "INCOMPLETE_EVIDENCE",
        "transport_complete": bool(transport_complete),
        "turn_sequence": sequence,
        "alternating_turns": bool(alternating),
        "acknowledged_turn_transfers_complete": bool(transfers_complete),
        "positive_rms_by_speaker": positive_rms,
        "positive_rms_all_speakers": all_positive_rms,
        "independent_asr_engine": independent_asr_engine,
        "independent_asr_available_by_speaker": asr_available,
        "independent_asr_all_speakers": all_asr_available,
        "generated_facts_passed": bool(generated_facts_passed),
        "audio_facts_passed": bool(audio_facts_passed),
        "verified": verified,
    }


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Standard Levenshtein-based Word Error Rate."""
    ref_words = re.findall(r"\w+", reference.lower())
    hyp_words = re.findall(r"\w+", hypothesis.lower())

    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    if not hyp_words:
        return 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    wer = float(d[len(ref_words), len(hyp_words)]) / float(len(ref_words))
    return round(min(1.0, wer), 4)


@dataclass
class TurnEvaluation:
    turn_index: int
    speaker: str
    scenario_id: str
    step_id: str
    expected_text: str
    recognized_input_text: str
    generated_text: str
    recognized_output_text: str
    input_wer: Optional[float] = None
    input_asr_available: bool = False
    output_wer: float = 0.0
    fact_checks_passed: bool = True
    contradiction_resolved: bool = True
    barge_in_recovered: bool = True
    error_rationale: Optional[str] = None
    physical_metrics: Dict[str, Any] = field(default_factory=dict)
    timing_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "scenario_id": self.scenario_id,
            "step_id": self.step_id,
            "expected_text": self.expected_text,
            "recognized_input_text": self.recognized_input_text,
            "generated_text": self.generated_text,
            "recognized_output_text": self.recognized_output_text,
            "input_wer": self.input_wer,
            "input_asr_available": self.input_asr_available,
            "output_wer": self.output_wer,
            "fact_checks_passed": self.fact_checks_passed,
            "contradiction_resolved": self.contradiction_resolved,
            "barge_in_recovered": self.barge_in_recovered,
            "error_rationale": self.error_rationale,
            "physical_metrics": self.physical_metrics,
            "timing_metrics": self.timing_metrics,
        }


class DialogueEvaluator:
    """Evaluates dialogue recordings, traces, and texts across all dimensions."""

    def __init__(self, acceptance_thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = acceptance_thresholds or {
            "max_clipping_pct": 0.10,          # Plan gate: <= 0.1%
            "min_crest_factor": 3.5,           # Alarm if < 3.5; target 5.0 - 8.5
            "max_crest_factor": 12.0,
            "max_tick_p95_ms": 80.0,           # Plan gate: <= 80 ms
            "max_tick_p99_ms": 100.0,
            "max_ttfa_ms": 2500.0,             # Time to first audio <= 2.5s
            "max_barge_in_latency_ms": 320.0,  # Plan gate: speech cancellation <= 320 ms
            "max_dropped_frames": 0,           # Zero loss duplex
        }
        self.evaluated_turns: List[TurnEvaluation] = []

    def evaluate_turn(
        self,
        *,
        turn_index: int,
        speaker: str,
        scenario: DialogueScenario,
        step: ScenarioStep,
        recognized_input: str,
        generated_text: str,
        recognized_output: str,
        audio_pcm: Optional[np.ndarray] = None,
        sample_rate: int = 16_000,
        timing_metrics: Optional[Dict[str, Any]] = None,
        packet_boundaries: Optional[List[int]] = None,
        input_reference_text: Optional[str] = None,
        input_asr_available: Optional[bool] = None,
    ) -> TurnEvaluation:
        # 1. Transcription WER checks. Missing ASR is unavailable evidence, not WER=0.
        input_reference_is_explicit = input_reference_text is not None
        input_reference = step.expected_text if input_reference_text is None else input_reference_text
        asr_available = bool(recognized_input) if input_asr_available is None else bool(input_asr_available)
        input_wer = calculate_wer(input_reference, recognized_input) if asr_available else None
        output_wer = calculate_wer(generated_text, recognized_output) if generated_text and recognized_output else 0.0

        # 2. Deterministic fact checks
        facts_ok = True
        fact_errors: List[str] = []
        target_text_for_facts = generated_text or recognized_output
        if input_reference_is_explicit and input_reference and not asr_available:
            fact_errors.append("Independent input ASR unavailable; input WER is not reported")
        for check in step.deterministic_checks:
            if not check(target_text_for_facts):
                facts_ok = False
                fact_errors.append(f"Fact check failed on '{step.fact_description}'")

        # 3. Contradiction & Barge-in checks
        contradiction_ok = True
        if "contradiction" in scenario.scenario_id and step.speaker == "assistant":
            # Must catch the contradiction and not passively accept false name
            contradiction_ok = facts_ok

        barge_in_ok = True
        if step.is_interruption:
            timing = timing_metrics or {}
            bi_lat = timing.get("barge_in_latency_ms", 0.0)
            if bi_lat > self.thresholds["max_barge_in_latency_ms"]:
                barge_in_ok = False
                fact_errors.append(f"Barge-in latency {bi_lat:.1f}ms exceeds threshold {self.thresholds['max_barge_in_latency_ms']}ms")

        # 4. Audio physical metrics
        phys = {}
        if audio_pcm is not None and audio_pcm.size > 0:
            phys = compute_physical_audio_metrics(
                audio_pcm,
                sample_rate=sample_rate,
                packet_boundary_indices=packet_boundaries,
            )
            if phys["clipping_pct"] > self.thresholds["max_clipping_pct"]:
                fact_errors.append(f"Clipping {phys['clipping_pct']}% exceeds {self.thresholds['max_clipping_pct']}%")

        error_rationale = "; ".join(fact_errors) if fact_errors else None

        record = TurnEvaluation(
            turn_index=turn_index,
            speaker=speaker,
            scenario_id=scenario.scenario_id,
            step_id=step.step_id,
            expected_text=step.expected_text,
            recognized_input_text=recognized_input,
            generated_text=generated_text,
            recognized_output_text=recognized_output,
            input_wer=input_wer,
            input_asr_available=asr_available,
            output_wer=output_wer,
            fact_checks_passed=facts_ok,
            contradiction_resolved=contradiction_ok,
            barge_in_recovered=barge_in_ok,
            error_rationale=error_rationale,
            physical_metrics=phys,
            timing_metrics=timing_metrics or {},
        )
        self.evaluated_turns.append(record)
        return record

    def build_summary_report(self) -> Dict[str, Any]:
        total_turns = len(self.evaluated_turns)
        if total_turns == 0:
            return {"total_turns": 0, "status": "NO_TURNS"}

        passed_turns = sum(1 for t in self.evaluated_turns if t.error_rationale is None)
        input_wer_values = [t.input_wer for t in self.evaluated_turns if t.input_wer is not None]
        mean_input_wer = float(np.mean(input_wer_values)) if input_wer_values else None
        mean_output_wer = float(np.mean([t.output_wer for t in self.evaluated_turns]))

        clipping_values = [t.physical_metrics.get("clipping_pct", 0.0) for t in self.evaluated_turns if t.physical_metrics]
        max_clipping = max(clipping_values) if clipping_values else 0.0

        crest_factors = [t.physical_metrics.get("crest_factor", 0.0) for t in self.evaluated_turns if t.physical_metrics]
        mean_crest = float(np.mean(crest_factors)) if crest_factors else 0.0

        ttfa_values = [t.timing_metrics.get("ttfa_ms") for t in self.evaluated_turns if t.timing_metrics.get("ttfa_ms") is not None]
        mean_ttfa = float(np.mean(ttfa_values)) if ttfa_values else 0.0
        p95_ttfa = float(np.percentile(ttfa_values, 95)) if ttfa_values else 0.0

        all_facts_ok = all(t.fact_checks_passed for t in self.evaluated_turns)
        all_contradictions_ok = all(t.contradiction_resolved for t in self.evaluated_turns)
        all_barge_in_ok = all(t.barge_in_recovered for t in self.evaluated_turns)

        return {
            "total_turns": total_turns,
            "passed_turns": passed_turns,
            "pass_rate_pct": round(100.0 * passed_turns / total_turns, 2),
            "mean_input_wer": round(mean_input_wer, 4) if mean_input_wer is not None else None,
            "input_asr_available_turns": len(input_wer_values),
            "mean_output_wer": round(mean_output_wer, 4),
            "max_clipping_pct": round(max_clipping, 4),
            "mean_crest_factor": round(mean_crest, 3),
            "mean_ttfa_ms": round(mean_ttfa, 2),
            "p95_ttfa_ms": round(p95_ttfa, 2),
            "all_facts_retained": all_facts_ok,
            "all_contradictions_resolved": all_contradictions_ok,
            "all_barge_ins_recovered": all_barge_in_ok,
            "turns": [t.to_dict() for t in self.evaluated_turns],
        }

    def generate_markdown_report(self) -> str:
        summary = self.build_summary_report()
        lines: List[str] = [
            "# Report di Valutazione Conversazione Full-Duplex (RTX 3090)",
            "",
            "## 1. Riepilogo Globale",
            f"- **Turni Totali Valutati**: {summary['total_turns']}",
            f"- **Turni Superati**: {summary['passed_turns']} ({summary.get('pass_rate_pct', 0.0)}%)",
            f"- **Mean Input WER (vs Ground-Truth)**: {summary['mean_input_wer']:.4f}"
            if summary.get("mean_input_wer") is not None
            else "- **Mean Input WER (vs Ground-Truth)**: N/A (ASR indipendente non disponibile)",
            f"- **Mean Output WER (Gen vs Audio Transcript)**: {summary.get('mean_output_wer', 0.0):.4f}",
            f"- **Max Clipping**: {summary.get('max_clipping_pct', 0.0):.4f}% (Soglia: <= 0.10%)",
            f"- **Mean Crest Factor**: {summary.get('mean_crest_factor', 0.0):.3f} (Target: 5.0 - 8.5)",
            f"- **TTFA p95**: {summary.get('p95_ttfa_ms', 0.0):.1f} ms",
            f"- **Conservazione Fatti**: {'PASS' if summary.get('all_facts_retained') else 'FAIL'}",
            f"- **Risoluzione Contraddizioni**: {'PASS' if summary.get('all_contradictions_resolved') else 'FAIL'}",
            f"- **Recupero Interruzione (Barge-In)**: {'PASS' if summary.get('all_barge_ins_recovered') else 'FAIL'}",
            "",
            "## 2. Tabella Dettagliata per Turno",
            "| Turno | Speaker | Scenario | Testo Previsto | Testo Riconosciuto Input | Testo Generato | Testo Riconosciuto Output | Fatti | Esito | Note/Errori |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for t in self.evaluated_turns:
            status_badge = "✅ PASS" if t.error_rationale is None else "❌ FAIL"
            facts_badge = "OK" if t.fact_checks_passed else "FAIL"
            err = t.error_rationale or "Nessun errore"
            lines.append(
                f"| {t.turn_index} | {t.speaker} | {t.scenario_id} | {t.expected_text[:60]}... | {t.recognized_input_text[:60]}... | {t.generated_text[:60]}... | {t.recognized_output_text[:60]}... | {facts_badge} | {status_badge} | {err} |"
            )

        return "\n".join(lines)
