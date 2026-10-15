# -*- coding: utf-8 -*-
"""dialogue_eval/scenarios.py

Predefined scenarios and evaluation specifications for full-duplex spoken dialogue.

Mandatory test cases:
1. Memoria di un nome o numero (Marco, codice 4829)
2. Domanda di chiarimento (prenotazione tavolo, persone e orario)
3. Cambiamento di istruzione (switch lingua inglese/italiano)
4. Contraddizione deliberata (affermazione falsa su nome precedente)
5. Barge-in durante la risposta (interruzione esplicita con recovery)
6. Chiusura del dialogo (ringraziamenti e commiato)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class ScenarioStep:
    step_id: str
    speaker: str  # "user" or "assistant"
    expected_text: str
    is_interruption: bool = False
    interruption_delay_s: float = 0.0
    deterministic_checks: List[Callable[[str], bool]] = field(default_factory=list)
    fact_description: str = ""


@dataclass
class DialogueScenario:
    scenario_id: str
    title: str
    description: str
    system_prompt_a: str
    system_prompt_b: str
    steps: List[ScenarioStep]
    max_turns: int = 4
    max_duration_s: float = 60.0


def check_contains_all(*keywords: str) -> Callable[[str], bool]:
    def _checker(text: str) -> bool:
        lower = text.lower()
        return all(kw.lower() in lower for kw in keywords)
    return _checker


def check_contains_any(*keywords: str) -> Callable[[str], bool]:
    def _checker(text: str) -> bool:
        lower = text.lower()
        return any(kw.lower() in lower for kw in keywords)
    return _checker


def check_contains_population_fifty_thousand(text: str) -> bool:
    """Match 50,000 exactly, not as a substring of a different population."""
    lower = text.casefold()
    return bool(
        re.search(r"(?<!\d)50(?:[., ]?000)(?!\d)", lower)
        or re.search(r"(?<!\d)50\s+mila\b", lower)
        or re.search(r"\b(?:cinquantamila|cinquanta\s+mila)\b", lower)
    )


def check_rejects_and_corrects(rejected: str, expected_correct: str) -> Callable[[str], bool]:
    def _checker(text: str) -> bool:
        lower = text.lower()
        # Must either mention the correct fact, or clarify the error
        has_correct = expected_correct.lower() in lower
        has_rejection_clue = any(w in lower for w in ["no", "non", "prima", "detto", "invece", "errore", "ricord"])
        return has_correct and has_rejection_clue
    return _checker


# Definition of the 6 canonical test scenarios:

SCENARIO_NAME_NUMBER = DialogueScenario(
    scenario_id="01_name_number_memory",
    title="Memoria di nome e numero d'ordine",
    description="L'utente fornisce il proprio nome Marco e codice ordine 4829, poi ne chiede conferma.",
    system_prompt_a="Sei Marco, un cliente che deve confermare il proprio codice ordine 4829.",
    system_prompt_b="Sei un assistente virtuale per il customer care. Rispondi con precisione e brevità.",
    steps=[
        ScenarioStep(
            step_id="turn_1_user",
            speaker="user",
            expected_text="Buongiorno, mi chiamo Marco e il mio codice ordine è 4829. Ricordalo per favore.",
            fact_description="Presentazione nome Marco e codice 4829",
        ),
        ScenarioStep(
            step_id="turn_1_assistant",
            speaker="assistant",
            expected_text="Buongiorno Marco, ho registrato il tuo codice 4829. Come posso aiutarti?",
            deterministic_checks=[check_contains_all("Marco", "4829")],
            fact_description="Riconoscimento e conferma immediata di nome e codice",
        ),
        ScenarioStep(
            step_id="turn_2_user",
            speaker="user",
            expected_text="Qual è il mio nome e qual è il mio codice d'ordine che ti ho appena detto?",
            fact_description="Domanda di verifica memoria a lungo termine del turno",
        ),
        ScenarioStep(
            step_id="turn_2_assistant",
            speaker="assistant",
            expected_text="Ti chiami Marco e il tuo codice d'ordine è 4829.",
            deterministic_checks=[check_contains_all("Marco", "4829")],
            fact_description="Conservazione esatta dei fatti tra turni",
        ),
    ],
    max_turns=4,
)

SCENARIO_CLARIFICATION = DialogueScenario(
    scenario_id="02_clarification_question",
    title="Domanda di chiarimento",
    description="L'utente esprime una richiesta ambigua, l'assistente chiede chiarimenti, l'utente specifica i dettagli.",
    system_prompt_a="Sei un cliente che desidera prenotare un tavolo per cena.",
    system_prompt_b="Sei il receptionist del ristorante. Se mancano persone o orario, chiedi sempre chiarimenti.",
    steps=[
        ScenarioStep(
            step_id="turn_1_user",
            speaker="user",
            expected_text="Vorrei prenotare un tavolo per cena per domani sera.",
            fact_description="Richiesta incompleta priva di numero di ospiti e orario",
        ),
        ScenarioStep(
            step_id="turn_1_assistant",
            speaker="assistant",
            expected_text="Certamente! Per quante persone e a che ora desiderate accomodarvi?",
            deterministic_checks=[check_contains_any("persone", "quanti", "orario", "ora", "quante")],
            fact_description="Richiesta proattiva di parametri mancanti",
        ),
        ScenarioStep(
            step_id="turn_2_user",
            speaker="user",
            expected_text="Siamo in quattro persone per le venti e trenta.",
            fact_description="Chiarimento con 4 persone alle 20:30",
        ),
        ScenarioStep(
            step_id="turn_2_assistant",
            speaker="assistant",
            expected_text="Perfetto, tavolo per quattro persone domani sera alle venti e trenta confermato.",
            deterministic_checks=[check_contains_any("quattro", "4"), check_contains_any("venti", "20", "trenta", "30")],
            fact_description="Conferma con recepimento dei dettagli forniti",
        ),
    ],
    max_turns=4,
)

SCENARIO_INSTRUCTION_CHANGE = DialogueScenario(
    scenario_id="03_instruction_change",
    title="Cambiamento dinamico di istruzione",
    description="L'utente cambia lingua chiedendo di rispondere in inglese, poi testa la risposta.",
    system_prompt_a="Sei un utente bilingue che testa la flessibilità linguistica dell'assistente.",
    system_prompt_b="Sei un assistente AI poliglotto. Adattati immediatamente alla lingua richiesta.",
    steps=[
        ScenarioStep(
            step_id="turn_1_user",
            speaker="user",
            expected_text="Da questo momento rispondimi esclusivamente in lingua inglese. Can you confirm?",
            fact_description="Richiesta di switch a lingua inglese",
        ),
        ScenarioStep(
            step_id="turn_1_assistant",
            speaker="assistant",
            expected_text="Yes, I confirm. I will answer you in English from now on.",
            deterministic_checks=[check_contains_any("yes", "confirm", "english", "sure", "certainly")],
            fact_description="Recepimento del cambio lingua",
        ),
        ScenarioStep(
            step_id="turn_2_user",
            speaker="user",
            expected_text="How is the weather typically in Rome during spring?",
            fact_description="Domanda in lingua inglese",
        ),
        ScenarioStep(
            step_id="turn_2_assistant",
            speaker="assistant",
            expected_text="In Rome during spring the weather is typically mild and sunny with pleasant temperatures.",
            deterministic_checks=[check_contains_any("mild", "sunny", "pleasant", "spring", "rome", "weather", "temperatures")],
            fact_description="Risposta coerente interamente in lingua inglese",
        ),
    ],
    max_turns=4,
)

SCENARIO_CONTRADICTION = DialogueScenario(
    scenario_id="04_deliberate_contradiction",
    title="Risoluzione di contraddizione deliberata",
    description="L'utente cerca di far cadere l'assistente in contraddizione affermando un nome falso.",
    system_prompt_a="Sei un tester che tenta di far confondere l'assistente sulle informazioni già fornite.",
    system_prompt_b="Sei un assistente rigoroso. Se l'utente contraddice fatti precedenti, correggilo cortesemente.",
    steps=[
        ScenarioStep(
            step_id="turn_1_user",
            speaker="user",
            expected_text="Mi chiamo Marco e il mio codice ordine è 4829.",
            fact_description="Fatti originali: Marco, 4829",
        ),
        ScenarioStep(
            step_id="turn_1_assistant",
            speaker="assistant",
            expected_text="Ricevuto Marco, codice 4829.",
            deterministic_checks=[check_contains_all("Marco", "4829")],
            fact_description="Acquisizione fatti originali",
        ),
        ScenarioStep(
            step_id="turn_2_user",
            speaker="user",
            expected_text="Prima ti ho detto che mi chiamo Roberto e che il codice era 9999, corretto?",
            fact_description="Contraddizione deliberata: Roberto e 9999",
        ),
        ScenarioStep(
            step_id="turn_2_assistant",
            speaker="assistant",
            expected_text="No, in realtà prima hai detto di chiamarti Marco e che il codice era 4829.",
            deterministic_checks=[
                check_rejects_and_corrects("Roberto", "Marco"),
                check_contains_any("4829"),
            ],
            fact_description="Rilevazione e correzione della contraddizione",
        ),
    ],
    max_turns=4,
)

SCENARIO_BARGE_IN = DialogueScenario(
    scenario_id="05_barge_in_interruption",
    title="Barge-in durante la risposta e recupero",
    description="Durante la generazione vocale dell'assistente, l'utente interrompe con una nuova istruzione.",
    system_prompt_a="Sei un utente frettoloso che interrompe l'assistente mentre parla.",
    system_prompt_b="Sei un assistente full-duplex. Se vieni interrotto, arrestati subito e ascolta la nuova richiesta.",
    steps=[
        ScenarioStep(
            step_id="turn_1_user",
            speaker="user",
            expected_text="Raccontami in dettaglio la storia della città di Venezia.",
            fact_description="Prompt a risposta lunga per testare l'interruzione",
        ),
        ScenarioStep(
            step_id="turn_1_assistant_interrupted",
            speaker="assistant",
            expected_text="Venezia fu fondata nel quinto secolo dagli abitanti della terraferma in fuga dalle invasioni barbariche...",
            fact_description="Risposta in corso di sintesi vocale da interrompere",
        ),
        ScenarioStep(
            step_id="turn_2_user_barge_in",
            speaker="user",
            expected_text="Aspetta! Fermati un attimo! Dimmi solo quanti abitanti ha oggi.",
            is_interruption=True,
            interruption_delay_s=1.2,
            fact_description="Interruzione acustica (barge-in)",
        ),
        ScenarioStep(
            step_id="turn_2_assistant_recovered",
            speaker="assistant",
            expected_text="Certamente! Oggi il centro storico di Venezia conta circa cinquantamila abitanti.",
            deterministic_checks=[
                check_contains_population_fifty_thousand,
                check_contains_any("abitanti"),
            ],
            fact_description="Recupero immediato senza residui della storia di Venezia",
        ),
    ],
    max_turns=4,
)

SCENARIO_CLOSING = DialogueScenario(
    scenario_id="06_graceful_closing",
    title="Chiusura del dialogo",
    description="L'utente ringrazia e si congeda; l'assistente chiude la conversazione educatamente.",
    system_prompt_a="Sei un cliente soddisfatto che intende chiudere la chiamata.",
    system_prompt_b="Sei un assistente cortese. Congedati con stile senza iniziare nuovi argomenti.",
    steps=[
        ScenarioStep(
            step_id="turn_1_user",
            speaker="user",
            expected_text="Perfetto, ho tutte le informazioni necessarie. Grazie mille per l'aiuto, buona giornata!",
            fact_description="Ringraziamento e commiato",
        ),
        ScenarioStep(
            step_id="turn_1_assistant",
            speaker="assistant",
            expected_text="È stato un piacere aiutarti! Ti auguro una splendida giornata, arrivederci!",
            deterministic_checks=[check_contains_any("piacere", "giornata", "arrivederci", "prego", "buona")],
            fact_description="Chiusura garbata e terminale",
        ),
    ],
    max_turns=2,
)

MODE_B_KICKOFF_TEXT = (
    "Buongiorno, mi chiamo Marco. Il mio codice ordine è quattro, otto, due, nove. "
    "Ripeto: quattro, otto, due, nove."
)


ALL_SCENARIOS = [
    SCENARIO_NAME_NUMBER,
    SCENARIO_CLARIFICATION,
    SCENARIO_INSTRUCTION_CHANGE,
    SCENARIO_CONTRADICTION,
    SCENARIO_BARGE_IN,
    SCENARIO_CLOSING,
]


def generate_synthetic_speech_pcm(
    duration_s: float,
    *,
    sample_rate: int = 16_000,
    base_f0: float = 140.0,
    formants: Optional[List[float]] = None,
) -> np.ndarray:
    """Generates a modulated harmonic formant waveform simulating clean human voiced speech.

    Avoids silent or flat artificial signals to exercise VAD, acoustic energy filters, and RMS metrics.
    """
    if duration_s <= 0.0:
        return np.empty(0, dtype=np.float32)

    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)

    # Pitch contour with subtle vibrato
    pitch = base_f0 + 8.0 * np.sin(2 * np.pi * 3.5 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate

    # Harmonic pulse train
    signal = np.zeros(num_samples, dtype=np.float32)
    target_formants = formants or [500.0, 1500.0, 2500.0]

    for k in range(1, 20):
        harm_freq = k * base_f0
        # Weight by proximity to formants (vocal tract resonance)
        weight = 0.0
        for f in target_formants:
            weight += np.exp(-((harm_freq - f) ** 2) / (2 * (150.0 ** 2)))
        signal += (weight + 0.1 / k) * np.sin(k * phase)

    # Apply syllabic amplitude envelope (4 Hz syllable rhythm)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2))
    # Soft ramp in / ramp out (20 ms)
    ramp_len = min(int(0.020 * sample_rate), num_samples // 2)
    if ramp_len > 0:
        envelope[:ramp_len] *= np.linspace(0.0, 1.0, ramp_len)
        envelope[-ramp_len:] *= np.linspace(1.0, 0.0, ramp_len)

    signal = signal * envelope

    # Normalize to RMS ~ 0.15 (-16 dBFS)
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 1e-6:
        signal = signal * (0.15 / rms)

    return np.clip(signal, -0.95, 0.95).astype(np.float32)


def generate_speech_for_text(text: str, *, sample_rate: int = 16_000) -> np.ndarray:
    """Convenience helper to generate harmonic synthetic speech proportional to text length."""
    words = max(1, len(text.split()))
    duration_s = max(0.48, min(6.0, words * 0.32))
    return generate_synthetic_speech_pcm(duration_s, sample_rate=sample_rate)
