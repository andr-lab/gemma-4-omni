#!/usr/bin/env python3
"""Generate explicitly synthetic Italian duplex conversations with Qwen3-TTS.

Run inside the RTX 3090 container. This writes an isolated synthetic candidate
under artifacts/; it deliberately refuses the canonical T0 benchmark tree.
Generated data is not human-annotated and is never eligible for T0 closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1_000
ACTIONS = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
EVENT_ACTIONS = {
    "backchannel": "CONTINUE",
    "interrupt": "YIELD",
    "yield": "YIELD",
    "correction": "REPLAN",
    "semantic_replan": "REPLAN",
    "false_start": "CONTINUE",
    "handover": "SPEAK",
    "assistant_backchannel": "BACKCHANNEL",
    "floor_holding": "LISTEN",
    "silence": "LISTEN",
}
_EVENT_PRIORITY = {"LISTEN": 0, "SPEAK": 1, "CONTINUE": 2, "BACKCHANNEL": 3, "REPLAN": 4, "YIELD": 5}
_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$")

# These scripts are authored synthetic test stimuli, not transcriptions of
# human conversations. Five scenarios jointly exercise all six policy actions.
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "01_backchannel_continue",
        "language": "Italian",
        "segments": [
            {
                "segment_id": "user_request",
                "speaker_id": "user",
                "at_s": 0.0,
                "text": (
                    "Devo organizzare una presentazione per il gruppo di lavoro e vorrei che mi aiutassi a "
                    "mettere in ordine le idee senza trasformare tutto in un elenco troppo rigido. Il tema è "
                    "come ridurre i tempi di attesa nel nostro servizio, partendo dai dati raccolti nell'ultimo "
                    "mese, distinguendo i problemi urgenti da quelli che possiamo affrontare con calma e "
                    "lasciando spazio alle domande dei colleghi alla fine."
                ),
            },
            {
                "segment_id": "assistant_explanation",
                "speaker_id": "assistant",
                "after": "user_request",
                "pause_s": 0.45,
                "text": (
                    "Certo. Per rendere la presentazione chiara, inizierei con una fotografia del problema: "
                    "quante richieste arrivano, quanto tempo passa prima della prima risposta e in quali fasce "
                    "orarie si formano le code più lunghe. Poi separerei i casi semplici, che si possono "
                    "risolvere con una procedura più snella, dai casi che richiedono il supporto di una persona "
                    "esperta. Infine proporrei due esperimenti piccoli, con una misura prima e dopo, così il "
                    "gruppo può giudicare i risultati sulla base dei dati e non delle impressioni."
                ),
            },
            {
                "segment_id": "user_backchannel",
                "speaker_id": "user",
                "during": "assistant_explanation",
                "offset_s": 5.0,
                "text": "Sì, chiaro.",
            },
            {
                "segment_id": "user_followup",
                "speaker_id": "user",
                "after": "assistant_explanation",
                "pause_s": 0.45,
                "text": (
                    "Mi piace l'idea dei due esperimenti. Aggiungerei anche una slide con le ipotesi, "
                    "così nessuno confonde una coincidenza con un miglioramento stabile."
                ),
            },
            {
                "segment_id": "assistant_close",
                "speaker_id": "assistant",
                "after": "user_followup",
                "pause_s": 0.45,
                "text": (
                    "Ottimo. Inseriamo quindi il problema, le fasce orarie, la distinzione tra richieste "
                    "semplici e complesse, i due esperimenti e le ipotesi da verificare. Lascerei l'ultima "
                    "slide per domande e decisioni operative."
                ),
            },
            {
                "segment_id": "user_next_question",
                "speaker_id": "user",
                "after": "assistant_close",
                "pause_s": 0.45,
                "text": (
                    "Prima di chiudere, vorrei capire chi raccoglie i dati e con quale frequenza li rivediamo. "
                    "Se il gruppo sceglie un indicatore solo, quale sarebbe il più utile per accorgerci presto "
                    "che le persone aspettano meno senza ricevere risposte più superficiali?"
                ),
            },
            {
                "segment_id": "assistant_next_response",
                "speaker_id": "assistant",
                "after": "user_next_question",
                "pause_s": 0.45,
                "text": (
                    "Sceglierei il tempo fino alla prima risposta utile, affiancandolo a un controllo sulla "
                    "qualità dei casi risolti. Una persona può aggiornare il dato ogni settimana e il gruppo "
                    "può confrontarlo con il valore iniziale dopo un mese. In questo modo il numero resta "
                    "semplice, ma non premia risposte veloci che non risolvono davvero il problema."
                ),
            },
        ],
        "events": [
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_explanation", "duration_s": 0.08},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_close", "duration_s": 0.08},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_next_response", "duration_s": 0.08},
            {"event_type": "floor_holding", "target_action": "LISTEN", "segment_id": "user_request"},
            {"event_type": "backchannel", "target_action": "CONTINUE", "segment_id": "user_backchannel", "max_duration_s": 1.0},
        ],
    },
    {
        "scenario_id": "02_barge_in_yield",
        "language": "Italian",
        "segments": [
            {
                "segment_id": "user_question",
                "speaker_id": "user",
                "at_s": 0.0,
                "text": (
                    "Puoi riepilogare il piano per domani? Mi interessa sapere l'ordine delle attività, "
                    "quale documento devo portare e se dobbiamo avvisare il team prima di spostare la "
                    "riunione. Ho poco tempo, quindi dimmi prima la cosa più importante e poi i dettagli."
                ),
            },
            {
                "segment_id": "assistant_interrupted",
                "speaker_id": "assistant",
                "after": "user_question",
                "pause_s": 0.45,
                "text": (
                    "La cosa più importante è arrivare alle nove con il riepilogo aggiornato. Per prima cosa "
                    "controlliamo i numeri, poi rivediamo le attività ancora aperte e infine decidiamo chi "
                    "contatta il gruppo. Prima di passare al documento, voglio chiarire anche come gestire "
                    "l'eventuale cambio di orario, perché non tutte le persone hanno già confermato la loro "
                    "presenza."
                ),
            },
            {
                "segment_id": "user_interrupt",
                "speaker_id": "user",
                "during": "assistant_interrupted",
                "offset_s": 3.0,
                "align_to_word_boundary": True,
                "text": "Aspetta, fermati un momento: la riunione non è domani.",
            },
            {
                "segment_id": "assistant_after_interrupt",
                "speaker_id": "assistant",
                "after": "user_interrupt",
                "pause_s": 0.45,
                "text": (
                    "Hai ragione, mi fermo. Riformulo senza dare per scontata la data: prima verifichiamo "
                    "il calendario condiviso, poi confermiamo il documento e avvisiamo il team soltanto "
                    "quando l'orario è stato approvato da tutti."
                ),
            },
            {
                "segment_id": "user_confirmation",
                "speaker_id": "user",
                "after": "assistant_after_interrupt",
                "pause_s": 0.45,
                "text": (
                    "Perfetto, controllo il calendario e poi mando l'aggiornamento. Nel messaggio specificherò "
                    "che l'orario è ancora in verifica, così nessuno prepara il documento sulla data sbagliata "
                    "prima della conferma definitiva."
                ),
            },
            {
                "segment_id": "assistant_final_summary",
                "speaker_id": "assistant",
                "after": "user_confirmation",
                "pause_s": 0.45,
                "text": (
                    "Va bene. Aspettiamo la verifica del calendario, confermiamo il nuovo orario e solo dopo "
                    "aggiorniamo invito e materiali. Se emerge un'altra discrepanza, ci fermiamo e controlliamo "
                    "la fonte condivisa prima di proseguire."
                ),
            },
        ],
        "events": [
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_interrupted", "duration_s": 0.08},
            {"event_type": "interrupt", "target_action": "YIELD", "segment_id": "user_interrupt", "truncates_segment_id": "assistant_interrupted", "truncate_delay_s": 0.08},
            {"event_type": "yield", "target_action": "YIELD", "segment_id": "user_interrupt"},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_after_interrupt", "duration_s": 0.08},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_final_summary", "duration_s": 0.08},
        ],
    },
    {
        "scenario_id": "03_semantic_replan",
        "language": "Italian",
        "segments": [
            {
                "segment_id": "user_schedule_request",
                "speaker_id": "user",
                "at_s": 0.0,
                "text": (
                    "Aiutami a preparare l'agenda per il prossimo incontro. Dovremo rivedere il budget, "
                    "assegnare le attività della settimana e lasciare almeno dieci minuti per le domande. "
                    "Per ora ho segnato mercoledì mattina, ma controlla con attenzione prima di costruire "
                    "la proposta definitiva."
                ),
            },
            {
                "segment_id": "assistant_old_plan",
                "speaker_id": "assistant",
                "after": "user_schedule_request",
                "pause_s": 0.45,
                "text": (
                    "Per mercoledì mattina potremmo iniziare con il budget, dedicare il blocco centrale "
                    "all'assegnazione delle attività e concludere con le domande. Manderei l'ordine del giorno "
                    "il giorno prima, insieme ai numeri aggiornati, così ciascuno può arrivare preparato. "
                    "Terrei inoltre un breve riepilogo finale con le decisioni e le persone responsabili."
                ),
            },
            {
                "segment_id": "user_correction",
                "speaker_id": "user",
                "during": "assistant_old_plan",
                "offset_s": 3.0,
                "align_to_word_boundary": True,
                "text": "No, intendevo per giovedì, non mercoledì.",
            },
            {
                "segment_id": "assistant_replanned",
                "speaker_id": "assistant",
                "after": "user_correction",
                "pause_s": 0.45,
                "text": (
                    "Ricevuto: sposto l'incontro a giovedì. Manteniamo lo stesso ordine, con budget all'inizio, "
                    "attività nella parte centrale e domande alla fine. Aggiorno anche il messaggio di invito "
                    "e il riepilogo, così la nuova data compare in tutti i materiali e non resta il riferimento "
                    "a mercoledì."
                ),
            },
            {
                "segment_id": "user_replan_acknowledgement",
                "speaker_id": "user",
                "after": "assistant_replanned",
                "pause_s": 0.45,
                "text": (
                    "Grazie, la distinzione è importante. Quando preparo l'invito, metterò giovedì nel titolo "
                    "e nel riepilogo, e controllerò entrambe le versioni prima di inviarlo per evitare che "
                    "resti in circolazione la prima bozza."
                ),
            },
            {
                "segment_id": "assistant_replan_confirmation",
                "speaker_id": "assistant",
                "after": "user_replan_acknowledgement",
                "pause_s": 0.45,
                "text": (
                    "Esatto. Aggiorniamo prima il calendario, poi la bozza dell'invito e infine il riepilogo. "
                    "Dopo un controllo incrociato possiamo inviare la comunicazione con la data corretta e "
                    "rimuovere il riferimento precedente."
                ),
            },
        ],
        "events": [
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_old_plan", "duration_s": 0.08},
            {"event_type": "correction", "target_action": "REPLAN", "segment_id": "user_correction", "truncates_segment_id": "assistant_old_plan", "truncate_delay_s": 0.08},
            {"event_type": "semantic_replan", "target_action": "REPLAN", "segment_id": "user_correction"},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_replanned", "duration_s": 0.08},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_replan_confirmation", "duration_s": 0.08},
        ],
    },
    {
        "scenario_id": "04_false_start_continue",
        "language": "Italian",
        "segments": [
            {
                "segment_id": "user_context",
                "speaker_id": "user",
                "at_s": 0.0,
                "text": (
                    "Sto confrontando tre modi diversi per presentare i risultati al gruppo. Il primo è "
                    "molto breve e mette in evidenza soltanto le conclusioni, il secondo mostra anche i "
                    "passaggi del calcolo e il terzo aggiunge una sezione sui limiti dei dati. Vorrei scegliere "
                    "un formato accessibile, ma senza nascondere le incertezze che possono cambiare la decisione."
                ),
            },
            {
                "segment_id": "assistant_suggestion",
                "speaker_id": "assistant",
                "after": "user_context",
                "pause_s": 0.45,
                "text": (
                    "Userei una struttura a livelli. La prima pagina contiene la decisione e due numeri "
                    "principali; nelle pagine successive mostriamo i calcoli e la qualità delle fonti; alla "
                    "fine aggiungiamo limiti, ipotesi e possibili verifiche. Così chi ha poco tempo capisce "
                    "il risultato, mentre chi deve controllarlo può seguire ogni passaggio senza dover "
                    "ricostruire da zero il ragionamento."
                ),
            },
            {
                "segment_id": "user_false_start",
                "speaker_id": "user",
                "during": "assistant_suggestion",
                "offset_s": 4.0,
                "text": "Ma...",
            },
            {
                "segment_id": "user_followup",
                "speaker_id": "user",
                "after": "assistant_suggestion",
                "pause_s": 0.45,
                "text": (
                    "Sì, è proprio il compromesso che cercavo. Possiamo aggiungere un esempio concreto "
                    "nella prima pagina e lasciare l'analisi completa in appendice?"
                ),
            },
            {
                "segment_id": "assistant_summary",
                "speaker_id": "assistant",
                "after": "user_followup",
                "pause_s": 0.45,
                "text": (
                    "Certamente. Metto l'esempio accanto ai due numeri principali e sposto i dettagli "
                    "tecnici in appendice. Nella sezione iniziale aggiungo anche un collegamento chiaro "
                    "all'analisi completa, così il lettore può passare dal riassunto alle prove quando vuole."
                ),
            },
            {
                "segment_id": "user_example_request",
                "speaker_id": "user",
                "after": "assistant_summary",
                "pause_s": 0.45,
                "text": (
                    "Per l'esempio sceglierei un caso piccolo e facile da controllare, non il risultato più "
                    "eclatante. Possiamo mostrare anche una situazione in cui il dato non basta per concludere, "
                    "così il lettore vede subito dove finisce l'evidenza e dove iniziano le ipotesi?"
                ),
            },
            {
                "segment_id": "assistant_example_response",
                "speaker_id": "assistant",
                "after": "user_example_request",
                "pause_s": 0.45,
                "text": (
                    "Sì. Aggiungiamo un esempio con un campione troppo piccolo e spieghiamo che cosa manca "
                    "per prendere una decisione. Accanto riportiamo il dato osservato, il limite e il controllo "
                    "successivo. Il lettore potrà così distinguere un risultato promettente da una conclusione "
                    "che i dati non sostengono ancora."
                ),
            },
        ],
        "events": [
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_suggestion", "duration_s": 0.08},
            {"event_type": "false_start", "target_action": "CONTINUE", "segment_id": "user_false_start", "max_duration_s": 1.0},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_summary", "duration_s": 0.08},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_example_response", "duration_s": 0.08},
        ],
    },
    {
        "scenario_id": "05_handover_and_passive_listening",
        "language": "Italian",
        "segments": [
            {
                "segment_id": "user_long_turn",
                "speaker_id": "user",
                "at_s": 0.0,
                "text": (
                    "Vorrei raccontarti con calma quello che è successo durante la prova di ieri. All'inizio "
                    "il gruppo ha seguito il piano senza difficoltà, poi abbiamo ricevuto due richieste quasi "
                    "insieme e la persona al banco ha dovuto scegliere quale gestire per prima. Non voglio "
                    "attribuire colpe a qualcuno: mi interessa capire se la procedura era abbastanza chiara, "
                    "se mancavano informazioni sul caso più urgente e se possiamo aggiungere un passaggio "
                    "semplice che aiuti tutti a coordinarsi. Dopo aver ascoltato anche gli altri colleghi, "
                    "potremo decidere se serve una modifica o soltanto una spiegazione migliore."
                ),
            },
            {
                "segment_id": "assistant_backchannel",
                "speaker_id": "assistant",
                "during": "user_long_turn",
                "offset_s": 10.0,
                "text": "Sì, capisco.",
            },
            {
                "segment_id": "user_finish",
                "speaker_id": "user",
                "after": "user_long_turn",
                "pause_s": 0.45,
                "text": (
                    "Questo è tutto. Se vuoi, posso recuperare i due orari esatti e il messaggio che ci "
                    "siamo scambiati, così confrontiamo la sequenza con la procedura scritta."
                ),
            },
            {
                "segment_id": "assistant_response",
                "speaker_id": "assistant",
                "after": "user_finish",
                "pause_s": 0.45,
                "text": (
                    "Grazie, ho ascoltato. Per prima cosa ricostruiamo la sequenza con gli orari e il "
                    "messaggio, poi confrontiamo ciò che è accaduto con la procedura. Se la regola era "
                    "ambigua, la riscriviamo con un criterio di priorità; se invece era chiara, prepariamo "
                    "un esempio pratico da condividere con il gruppo. In entrambi i casi documentiamo la "
                    "decisione e verifichiamo che la modifica risolva il problema senza crearne uno nuovo."
                ),
            },
            {
                "segment_id": "user_next_step_question",
                "speaker_id": "user",
                "after": "assistant_response",
                "pause_s": 0.45,
                "text": (
                    "Mi sembra un buon piano. Per non perderci, annoterò gli orari e confronterò il messaggio "
                    "con la procedura attuale. Poi porterò il riepilogo al gruppo, specificando quali punti "
                    "abbiamo verificato e quali restano ancora da chiarire."
                ),
            },
            {
                "segment_id": "assistant_next_steps",
                "speaker_id": "assistant",
                "after": "user_next_step_question",
                "pause_s": 0.45,
                "text": (
                    "Perfetto. Dopo il confronto possiamo condividere i fatti, chiedere ai colleghi se manca "
                    "qualche passaggio e concordare una modifica soltanto se risolve il problema osservato. "
                    "Conserverei anche la versione precedente della procedura, così sarà chiaro che cosa è "
                    "cambiato e perché."
                ),
            },
        ],
        "events": [
            {"event_type": "floor_holding", "target_action": "LISTEN", "segment_id": "user_long_turn"},
            {"event_type": "assistant_backchannel", "target_action": "BACKCHANNEL", "segment_id": "assistant_backchannel", "max_duration_s": 1.0},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_response", "duration_s": 0.08},
            {"event_type": "handover", "target_action": "SPEAK", "segment_id": "assistant_next_steps", "duration_s": 0.08},
        ],
    },
)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_scenarios(scenarios: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    """Validate authored plans and return their effective action/event coverage."""
    if len(scenarios) < 5:
        raise ValueError("at least five authored dialogue scenarios are required")
    scenario_ids: set[str] = set()
    action_coverage: set[str] = set()
    event_coverage: set[str] = set()
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        if not isinstance(scenario_id, str) or not _ID_RE.fullmatch(scenario_id) or scenario_id in scenario_ids:
            raise ValueError(f"invalid or duplicate scenario_id: {scenario_id!r}")
        scenario_ids.add(scenario_id)
        segments = scenario.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"scenario {scenario_id!r} requires segments")
        segment_ids: set[str] = set()
        for index, segment in enumerate(segments):
            if not isinstance(segment, Mapping):
                raise ValueError(f"scenario {scenario_id!r} segment[{index}] must be an object")
            segment_id = segment.get("segment_id")
            if not isinstance(segment_id, str) or not _ID_RE.fullmatch(segment_id) or segment_id in segment_ids:
                raise ValueError(f"scenario {scenario_id!r} has invalid/duplicate segment_id {segment_id!r}")
            segment_ids.add(segment_id)
            if segment.get("speaker_id") not in {"user", "assistant"}:
                raise ValueError(f"segment {segment_id!r} speaker_id must be user or assistant")
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"segment {segment_id!r} requires non-empty text")
            relations = [key for key in ("at_s", "after", "during") if key in segment]
            if len(relations) != 1:
                raise ValueError(f"segment {segment_id!r} requires exactly one of at_s, after, or during")
            if "at_s" in segment and (not _is_finite_number(segment["at_s"]) or segment["at_s"] < 0):
                raise ValueError(f"segment {segment_id!r} has invalid at_s")
            if "after" in segment:
                if segment["after"] not in segment_ids - {segment_id}:
                    raise ValueError(f"segment {segment_id!r} after reference must name an earlier segment")
                if not _is_finite_number(segment.get("pause_s", 0.0)) or segment.get("pause_s", 0.0) < 0:
                    raise ValueError(f"segment {segment_id!r} has invalid pause_s")
            if "during" in segment:
                if segment["during"] not in segment_ids - {segment_id}:
                    raise ValueError(f"segment {segment_id!r} during reference must name an earlier segment")
                if not _is_finite_number(segment.get("offset_s", 0.0)) or segment.get("offset_s", 0.0) < 0:
                    raise ValueError(f"segment {segment_id!r} has invalid offset_s")
                event_coverage.add("overlap")
        events = scenario.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"scenario {scenario_id!r} requires event annotations")
        for event in events:
            if not isinstance(event, Mapping) or event.get("segment_id") not in segment_ids:
                raise ValueError(f"scenario {scenario_id!r} has an event without a valid segment_id")
            event_type = event.get("event_type")
            target_action = event.get("target_action")
            if not isinstance(event_type, str) or not event_type:
                raise ValueError(f"scenario {scenario_id!r} event requires event_type")
            if target_action not in ACTIONS:
                raise ValueError(f"scenario {scenario_id!r} event {event_type!r} has invalid target_action")
            expected_action = EVENT_ACTIONS.get(event_type)
            if expected_action is not None and target_action != expected_action:
                raise ValueError(f"event {event_type!r} must map to {expected_action}")
            action_coverage.add(target_action)
            event_coverage.add(event_type)
            if "truncates_segment_id" in event and event["truncates_segment_id"] not in segment_ids:
                raise ValueError(f"event {event_type!r} truncates an unknown segment")
            for duration_key in ("duration_s", "max_duration_s"):
                if duration_key in event and (not _is_finite_number(event[duration_key]) or event[duration_key] <= 0):
                    raise ValueError(f"event {event_type!r} has invalid {duration_key}")
            if "truncate_delay_s" in event and (
                not _is_finite_number(event["truncate_delay_s"]) or event["truncate_delay_s"] < 0
            ):
                raise ValueError(f"event {event_type!r} has invalid truncate_delay_s")
        if any("after" in segment and float(segment.get("pause_s", 0.0)) >= 0.35 for segment in segments):
            event_coverage.add("silence")
    if action_coverage != set(ACTIONS):
        missing = sorted(set(ACTIONS) - action_coverage)
        raise ValueError(f"scenario action coverage is incomplete: {missing}")
    return action_coverage, event_coverage


def ensure_noncanonical_output_path(output_dir: str | Path, *, repository_root: str | Path) -> Path:
    """Return an absolute safe output path, refusing the canonical benchmark tree."""
    root = Path(os.path.abspath(os.fspath(repository_root)))
    candidate = Path(os.path.abspath(os.fspath(output_dir)))
    benchmark_root = root / "tools" / "training_gates" / "benchmarks" / "benchmark_data"
    if candidate == benchmark_root or benchmark_root in candidate.parents:
        raise ValueError("synthetic Qwen3-TTS output is forbidden under the canonical benchmark tree")
    if any(item.is_symlink() for item in (candidate, *candidate.parents)):
        raise ValueError(f"synthetic output path contains a symlink: {candidate}")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("synthetic candidate output must remain inside the repository") from exc
    return candidate


def _normalise_pcm(pcm: Any, *, sample_rate: int, label: str) -> np.ndarray:
    audio = np.asarray(pcm, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"{label} must be non-empty mono audio")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{label} sample rate must be exactly {SAMPLE_RATE} Hz, got {sample_rate}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{label} contains non-finite samples")
    peak = float(np.max(np.abs(audio)))
    if peak >= 1.0:
        raise ValueError(f"{label} peak {peak:.6f} reaches the clipping boundary")
    if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) <= 1e-6:
        raise ValueError(f"{label} is silent")
    return audio


def compose_stereo_timeline(
    clips: Mapping[str, np.ndarray],
    segments: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Place user on channel 0 and assistant on channel 1 without downmixing overlap."""
    if not segments:
        raise ValueError("at least one segment is required")
    last_sample = 0
    validated: list[tuple[Mapping[str, Any], np.ndarray, int, int, int]] = []
    for segment in segments:
        segment_id = segment.get("segment_id")
        speaker = segment.get("speaker_id")
        if speaker not in {"user", "assistant"}:
            raise ValueError(f"segment {segment_id!r} has unknown speaker_id")
        if segment_id not in clips:
            raise ValueError(f"missing audio clip for segment {segment_id!r}")
        start_sample, end_sample = segment.get("start_sample"), segment.get("end_sample")
        if (
            isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0
            or isinstance(end_sample, bool) or not isinstance(end_sample, int) or end_sample <= start_sample
        ):
            raise ValueError(f"segment {segment_id!r} has invalid sample bounds")
        clip = np.asarray(clips[segment_id], dtype=np.float32)
        if clip.ndim != 1 or not np.isfinite(clip).all():
            raise ValueError(f"segment {segment_id!r} audio must be finite mono")
        if float(np.max(np.abs(clip))) >= 1.0:
            raise ValueError(f"segment {segment_id!r} audio peak reaches the clipping boundary")
        audible_length = end_sample - start_sample
        if audible_length > clip.size:
            raise ValueError(f"segment {segment_id!r} end exceeds its source clip")
        last_sample = max(last_sample, end_sample)
        channel = 0 if speaker == "user" else 1
        validated.append((segment, clip, start_sample, end_sample, channel))
    total_samples = ((last_sample + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    for segment, clip, start_sample, end_sample, channel in validated:
        audible_length = end_sample - start_sample
        stereo[start_sample:end_sample, channel] += clip[:audible_length]
    if not np.isfinite(stereo).all():
        raise ValueError("composed timeline contains non-finite samples")
    if float(np.max(np.abs(stereo))) >= 1.0:
        raise ValueError("composed stereo peak reaches the clipping boundary")
    return stereo


def build_frame_annotations(
    conversation_id: str,
    total_samples: int,
    segments: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    transcript_spans: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build causal frame targets and speech/event flags on exact 80 ms boundaries."""
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("conversation_id must be a non-empty string")
    if isinstance(total_samples, bool) or not isinstance(total_samples, int) or total_samples <= 0:
        raise ValueError("total_samples must be a positive integer")
    frame_count = (total_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    segment_rows = list(segments)
    event_rows = list(events)
    spans = list(transcript_spans) if transcript_spans is not None else [
        {
            "span_id": segment["segment_id"],
            "speaker_id": segment["speaker_id"],
            "start_sample": segment["start_sample"],
            "end_sample": segment["end_sample"],
        }
        for segment in segment_rows
    ]
    for segment in segment_rows:
        if segment.get("speaker_id") not in {"user", "assistant"}:
            raise ValueError("frame segment has invalid speaker_id")
        if not (0 <= segment.get("start_sample", -1) < segment.get("end_sample", -1) <= total_samples):
            raise ValueError("frame segment bounds are invalid")
    for event in event_rows:
        action = event.get("target_action")
        if action is not None and action not in ACTIONS:
            raise ValueError(f"invalid event target_action: {action!r}")
        if not (0 <= event.get("start_sample", -1) < event.get("end_sample", -1) <= total_samples):
            raise ValueError("frame event bounds are invalid")
    for span in spans:
        if not (0 <= span.get("start_sample", -1) < span.get("end_sample", -1) <= total_samples):
            raise ValueError("transcript span bounds are invalid")
    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        start_sample = frame_index * FRAME_SAMPLES
        end_sample = min(total_samples, start_sample + FRAME_SAMPLES)
        active_segments = [
            segment for segment in segment_rows
            if segment["start_sample"] < end_sample and segment["end_sample"] > start_sample
        ]
        active_events = [
            event for event in event_rows
            if event["start_sample"] < end_sample and event["end_sample"] > start_sample
        ]
        explicit_actions = [event["target_action"] for event in active_events if event.get("target_action") is not None]
        if explicit_actions:
            target_action = max(explicit_actions, key=lambda action: _EVENT_PRIORITY[action])
        elif any(segment["speaker_id"] == "assistant" for segment in active_segments):
            target_action = "CONTINUE"
        else:
            target_action = "LISTEN"
        active_span_ids = [
            str(span["span_id"])
            for span in spans
            if span["start_sample"] < end_sample and span["end_sample"] > start_sample
        ]
        frames.append(
            {
                "conversation_id": conversation_id,
                "frame_index": frame_index,
                "timestamp_ms": frame_index * FRAME_MS,
                "window_ms": FRAME_MS,
                "target_action": target_action,
                "transcript_spans": active_span_ids,
                "user_speaking": any(segment["speaker_id"] == "user" for segment in active_segments),
                "assistant_speaking": any(segment["speaker_id"] == "assistant" for segment in active_segments),
                "events": sorted({str(event["event_type"]) for event in active_events}),
                "available_until_frame": frame_index,
            }
        )
    return frames


def _resolve_timeline(
    scenario: Mapping[str, Any],
    clip_info: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve relative segment offsets, speech truncation, events, overlap and silence."""
    segments: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    truncation_targets = {
        str(event["truncates_segment_id"])
        for event in scenario["events"]
        if event.get("truncates_segment_id") is not None
    }
    for source in scenario["segments"]:
        row = dict(source)
        segment_id = str(row["segment_id"])
        duration_samples = int(clip_info[segment_id]["duration_samples"])
        if "at_s" in row:
            start_sample = round(float(row["at_s"]) * SAMPLE_RATE)
        elif "after" in row:
            anchor = by_id[str(row["after"])]
            start_sample = anchor["end_sample"] + round(float(row.get("pause_s", 0.0)) * SAMPLE_RATE)
        else:
            anchor_id = str(row["during"])
            anchor = by_id[anchor_id]
            offset_s = float(row.get("offset_s", 0.0))
            if row.get("align_to_word_boundary") and anchor_id in truncation_targets:
                candidate_words = [
                    word for word in clip_info[anchor_id].get("words", [])
                    if _is_finite_number(word.get("start_s")) and float(word["start_s"]) >= offset_s
                ]
                if candidate_words:
                    offset_s = float(candidate_words[0]["start_s"])
            start_sample = anchor["start_sample"] + round(offset_s * SAMPLE_RATE)
        end_sample = start_sample + duration_samples
        resolved = {
            **row,
            "start_sample": start_sample,
            "generated_end_sample": end_sample,
            "end_sample": end_sample,
            "truncated": False,
        }
        segments.append(resolved)
        by_id[segment_id] = resolved
    for segment in segments:
        if "during" in segment:
            anchor = by_id[str(segment["during"])]
            if segment["start_sample"] >= anchor["generated_end_sample"]:
                raise ValueError(
                    f"segment {segment['segment_id']!r} starts after the speech it must overlap"
                )
    events: list[dict[str, Any]] = []
    for source in scenario["events"]:
        segment = by_id[str(source["segment_id"])]
        start_sample = segment["start_sample"] + round(float(source.get("offset_s", 0.0)) * SAMPLE_RATE)
        truncate_id = source.get("truncates_segment_id")
        if truncate_id is not None:
            target = by_id[str(truncate_id)]
            if start_sample <= target["start_sample"]:
                raise ValueError(f"interruption {source['event_type']!r} does not follow speech onset")
            cutoff_sample = start_sample + round(float(source.get("truncate_delay_s", 0.0)) * SAMPLE_RATE)
            if start_sample < target["end_sample"]:
                target["end_sample"] = min(target["end_sample"], cutoff_sample)
                target["truncated"] = target["end_sample"] < target["generated_end_sample"]
                target["truncated_by_event"] = source["event_type"]
        if "max_duration_s" in source:
            duration_limit = round(float(source["max_duration_s"]) * SAMPLE_RATE)
            if segment["end_sample"] - start_sample > duration_limit:
                segment["end_sample"] = start_sample + duration_limit
                segment["truncated"] = True
                segment["truncated_by_event"] = source["event_type"]
        if "duration_s" in source:
            end_sample = start_sample + round(float(source["duration_s"]) * SAMPLE_RATE)
        elif "max_duration_s" in source:
            end_sample = start_sample + round(float(source["max_duration_s"]) * SAMPLE_RATE)
        else:
            end_sample = segment["end_sample"]
        end_sample = min(end_sample, segment["end_sample"])
        if end_sample > start_sample:
            events.append(
                {
                    "event_id": f"{scenario['scenario_id']}:{source['event_type']}:{source['segment_id']}",
                    "event_type": source["event_type"],
                    "target_action": source["target_action"],
                    "segment_id": source["segment_id"],
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                }
            )
    for segment in segments:
        if segment["end_sample"] <= segment["start_sample"]:
            raise ValueError(f"segment {segment['segment_id']!r} has no audible samples after truncation")
    for left_index, left in enumerate(segments):
        for right in segments[left_index + 1:]:
            start = max(left["start_sample"], right["start_sample"])
            end = min(left["end_sample"], right["end_sample"])
            if start < end and left["speaker_id"] != right["speaker_id"]:
                events.append(
                    {
                        "event_id": f"{scenario['scenario_id']}:overlap:{left['segment_id']}:{right['segment_id']}",
                        "event_type": "overlap",
                        "target_action": None,
                        "segment_id": right["segment_id"],
                        "start_sample": start,
                        "end_sample": end,
                    }
                )
    for segment in segments:
        if "after" not in segment:
            continue
        previous = by_id[str(segment["after"])]
        start, end = previous["end_sample"], segment["start_sample"]
        if end - start >= round(0.35 * SAMPLE_RATE):
            events.append(
                {
                    "event_id": f"{scenario['scenario_id']}:silence:{previous['segment_id']}:{segment['segment_id']}",
                    "event_type": "silence",
                    "target_action": "LISTEN",
                    "segment_id": segment["segment_id"],
                    "start_sample": start,
                    "end_sample": end,
                }
            )
    return segments, events


def _validate_reference(path: Path, *, speaker: str) -> dict[str, Any]:
    import soundfile as sf

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{speaker} voice reference must be a regular local file: {path}")
    info = sf.info(str(path))
    if info.channels != 1 or info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"{speaker} voice reference must be non-empty mono audio: {path}")
    pcm, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != info.samplerate or pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError(f"{speaker} voice reference contains invalid audio samples: {path}")
    if float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64)))) <= 1e-6:
        raise ValueError(f"{speaker} voice reference is silent: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "sample_rate_hz": int(info.samplerate),
        "duration_s": info.frames / info.samplerate,
    }


def _validate_worker_clip(path: Path, *, segment_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    import soundfile as sf

    pcm, sample_rate = sf.read(str(path), dtype="float32")
    pcm = _normalise_pcm(pcm, sample_rate=int(sample_rate), label=f"Qwen clip {segment_id!r}")
    clipping = int(np.count_nonzero(np.abs(pcm) >= 32767 / 32768))
    quality = {
        "sample_rate_hz": int(sample_rate),
        "channels": 1,
        "duration_s": pcm.size / SAMPLE_RATE,
        "rms": float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64)))),
        "peak": float(np.max(np.abs(pcm))),
        "clipped_sample_count": clipping,
        "clipping_pct": 100.0 * clipping / pcm.size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if clipping:
        raise ValueError(f"Qwen clip {segment_id!r} contains clipped PCM samples")
    return pcm, quality


def _asr_transcription(asr: Any, path: Path) -> dict[str, Any]:
    result, info = asr.transcribe(
        str(path), language="it", beam_size=1, word_timestamps=True, vad_filter=False
    )
    chunks = list(result)
    transcript = " ".join(chunk.text.strip() for chunk in chunks if chunk.text.strip()).strip()
    words: list[dict[str, Any]] = []
    for chunk in chunks:
        for word in getattr(chunk, "words", None) or []:
            text = str(getattr(word, "word", "")).strip()
            start_s, end_s = getattr(word, "start", None), getattr(word, "end", None)
            if text and _is_finite_number(start_s) and _is_finite_number(end_s) and float(end_s) > float(start_s):
                words.append({"text": text, "start_s": float(start_s), "end_s": float(end_s)})
    return {
        "text": transcript,
        "language": getattr(info, "language", "it"),
        "words": words,
    }


def _wer(reference: str, hypothesis: str) -> float:
    from dialogue_eval.evaluator import calculate_wer

    return float(calculate_wer(reference, hypothesis))


def _atomic_write_audio(path: Path, pcm: np.ndarray, *, channels: int) -> dict[str, Any]:
    import soundfile as sf

    if channels <= 0 or (channels == 1 and np.asarray(pcm).ndim != 1) or (channels > 1 and (np.asarray(pcm).ndim != 2 or np.asarray(pcm).shape[1] != channels)):
        raise ValueError("audio shape does not match requested channel count")
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"refusing symlink audio output: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.wav", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            sf.write(target, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE or info.channels != channels or info.subtype != "PCM_16":
        raise RuntimeError(f"written WAV contract failed for {path}: {info}")
    saved, rate = sf.read(str(path), dtype="float32")
    expected_ndim = 1 if channels == 1 else 2
    if rate != SAMPLE_RATE or saved.ndim != expected_ndim or (channels > 1 and saved.shape[1] != channels) or not np.isfinite(saved).all():
        raise RuntimeError(f"written WAV is invalid or non-finite: {path}")
    channel_data = saved.reshape(-1, 1) if channels == 1 else saved
    peaks = np.max(np.abs(channel_data), axis=0)
    if np.any(peaks >= 1.0):
        raise RuntimeError(f"written WAV reached clipping boundary: {path}")
    clipped = int(np.count_nonzero(np.abs(channel_data) >= 32767 / 32768))
    if clipped:
        raise RuntimeError(f"written WAV contains clipped PCM samples: {path}")
    rms = np.sqrt(np.mean(np.square(channel_data, dtype=np.float64), axis=0))
    if np.any(rms <= 1e-6):
        raise RuntimeError(f"written WAV has a silent channel: {path}")
    return {
        "sample_rate_hz": rate,
        "channels": channels,
        "rms_by_channel": [float(value) for value in rms],
        "peak_by_channel": [float(value) for value in peaks],
        "clipped_sample_count": clipped,
        "clipping_pct": 100.0 * clipped / channel_data.size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _write_frames(path: Path, frames: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for frame in frames:
                target.write(json.dumps(frame, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _find_snapshot(patterns: Sequence[str], description: str) -> Path:
    import glob

    matches = sorted(path for pattern in patterns for path in glob.glob(pattern) if Path(path).is_dir())
    if not matches:
        raise FileNotFoundError(f"No cached {description} snapshot found; downloads are disabled")
    return Path(matches[-1]).resolve()


def _run_phase1_worker(
    *,
    jobs: list[dict[str, str]],
    model_snapshot: Path,
    reference_audio: Path,
    output_dir: Path,
    phase1_python: Path,
    worker: Path,
    max_new_tokens: int,
) -> None:
    if not phase1_python.is_file():
        raise FileNotFoundError(f"isolated Qwen Phase 1 Python not found: {phase1_python}")
    if not worker.is_file():
        raise FileNotFoundError(f"Qwen Phase 1 worker not found: {worker}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen3tts_duplex_jobs_") as temp_dir:
        jobs_path = Path(temp_dir) / "jobs.json"
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        command = [
            str(phase1_python), str(worker),
            "--model-snapshot", str(model_snapshot),
            "--reference-audio", str(reference_audio),
            "--jobs-json", str(jobs_path),
            "--output-dir", str(output_dir),
            "--max-new-tokens", str(max_new_tokens),
        ]
        print(f"[duplex-qwen3tts] generating {len(jobs)} clips for reference {reference_audio.name}", flush=True)
        subprocess.run(command, cwd=worker.parent.parent, env=environment, check=True)
    missing = [item["filename"] for item in jobs if not (output_dir / item["filename"]).is_file()]
    if missing:
        raise RuntimeError(f"Qwen worker did not create requested clips: {missing}")


def _build_transcript_spans(
    segments: Sequence[Mapping[str, Any]],
    clip_info: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        audible_length = int(segment["end_sample"] - segment["start_sample"])
        relative_words = clip_info[segment_id].get("words", [])
        added_word = False
        for word_index, word in enumerate(relative_words):
            start_s, end_s = word.get("start_s"), word.get("end_s")
            if not (_is_finite_number(start_s) and _is_finite_number(end_s)):
                continue
            start = round(float(start_s) * SAMPLE_RATE)
            end = round(float(end_s) * SAMPLE_RATE)
            # For an interrupted utterance, omit words whose ASR alignment reaches beyond the cut.
            if start < 0 or end <= start or end > audible_length:
                continue
            spans.append(
                {
                    "span_id": f"{segment_id}:w{word_index:04d}",
                    "segment_id": segment_id,
                    "speaker_id": segment["speaker_id"],
                    "text": word["text"],
                    "start_sample": segment["start_sample"] + start,
                    "end_sample": segment["start_sample"] + end,
                    "alignment_source": "faster-whisper-word-timestamps",
                }
            )
            added_word = True
        if not added_word and not segment.get("truncated"):
            transcript = str(clip_info[segment_id]["asr_transcript"])
            if transcript:
                spans.append(
                    {
                        "span_id": f"{segment_id}:asr",
                        "segment_id": segment_id,
                        "speaker_id": segment["speaker_id"],
                        "text": transcript,
                        "start_sample": segment["start_sample"],
                        "end_sample": segment["end_sample"],
                        "alignment_source": "faster-whisper-segment",
                    }
                )
    return spans


def _safe_write_candidate(args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel

    repository_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_noncanonical_output_path(args.output_dir, repository_root=repository_root)
    if output_dir.exists() and not getattr(args, "resume", False):
        raise FileExistsError(f"refusing to overwrite existing candidate directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS and Faster-Whisper candidate generation requires CUDA on the RTX 3090")
    validate_scenarios(SCENARIOS)
    reference_paths = {
        "user": Path(os.path.abspath(os.fspath(args.speaker_a_reference))),
        "assistant": Path(os.path.abspath(os.fspath(args.speaker_b_reference))),
    }
    references = {
        role: _validate_reference(path, speaker=role)
        for role, path in reference_paths.items()
    }
    if references["user"]["sha256"] == references["assistant"]["sha256"]:
        raise ValueError("speaker A and B must use distinct reference audio bytes")
    if args.max_new_tokens < 8:
        raise ValueError("max-new-tokens must be at least 8")
    if not 0.0 <= args.max_wer <= 1.0:
        raise ValueError("max-wer must be between zero and one")

    model_snapshot = args.model_snapshot.resolve() if args.model_snapshot else _find_snapshot(
        ["/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*"], "Qwen3-TTS"
    )
    if not model_snapshot.is_dir():
        raise FileNotFoundError(f"Qwen3-TTS model snapshot is not a directory: {model_snapshot}")
    asr_snapshot = args.asr_snapshot.resolve() if args.asr_snapshot else _find_snapshot(
        [
            "/workspace/.hf_cache/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*",
            "/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*",
            "/workspace/.hf_cache/hub/models--Systran--faster-whisper-small/snapshots/*",
        ], "Faster-Whisper",
    )
    if not asr_snapshot.is_dir():
        raise FileNotFoundError(f"Faster-Whisper snapshot is not a directory: {asr_snapshot}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = output_dir / ".work"
    temp_root.mkdir(parents=True, exist_ok=True)
    generated_paths: dict[str, Path] = {}
    jobs_by_speaker: dict[str, list[dict[str, str]]] = {"user": [], "assistant": []}
    source_segments: dict[str, Mapping[str, Any]] = {}
    for scenario in SCENARIOS:
        for segment in scenario["segments"]:
            segment_id = str(segment["segment_id"])
            key = f"{scenario['scenario_id']}__{segment_id}"
            if key in source_segments:
                raise ValueError(f"duplicate generated clip id: {key}")
            source_segments[key] = {**segment, "scenario_id": scenario["scenario_id"]}
            filename = f"{key}.wav"
            jobs_by_speaker[str(segment["speaker_id"])].append(
                {"filename": filename, "text": str(segment["text"]), "language": "Italian"}
            )
            generated_paths[key] = temp_root / filename

    try:
        for role in ("user", "assistant"):
            role_jobs = jobs_by_speaker[role]
            if role_jobs:
                role_output = temp_root / f"worker_{role}"
                all_exist = role_output.is_dir() and all(
                    (role_output / item["filename"]).is_file() for item in role_jobs
                )
                if getattr(args, "resume", False) and all_exist:
                    print(f"[duplex-qwen3tts] reusing existing {len(role_jobs)} clips for {role}", flush=True)
                else:
                    _run_phase1_worker(
                        jobs=role_jobs,
                        model_snapshot=model_snapshot,
                        reference_audio=reference_paths[role],
                        output_dir=role_output,
                        phase1_python=args.phase1_python,
                        worker=args.worker,
                        max_new_tokens=args.max_new_tokens,
                    )
                for item in role_jobs:
                    generated_paths[
                        Path(item["filename"]).stem.replace(".wav", "")
                    ] = role_output / item["filename"]
    except Exception:
        _atomic_write_json(output_dir / "generation_failure.json", {"status": "FAILED", "stage": "qwen_tts", "synthetic": True})
        raise

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    asr = WhisperModel(str(asr_snapshot), device="cuda", compute_type="float16")
    clip_arrays: dict[str, np.ndarray] = {}
    clip_info: dict[str, dict[str, Any]] = {}
    segment_output: dict[str, dict[str, Any]] = {}
    for clip_key, source in source_segments.items():
        clip_path = generated_paths[clip_key]
        pcm, quality = _validate_worker_clip(clip_path, segment_id=clip_key)
        asr_result = _asr_transcription(asr, clip_path)
        wer = _wer(str(source["text"]), asr_result["text"])
        quality.update({"asr_transcript": asr_result["text"], "wer": wer, "asr_passed": wer <= args.max_wer})
        if wer > args.max_wer:
            _atomic_write_json(
                output_dir / "generation_failure.json",
                {"status": "FAILED", "stage": "asr_quality", "segment_id": clip_key, "wer": wer, "max_wer": args.max_wer, "synthetic": True},
            )
            raise RuntimeError(f"ASR WER {wer:.4f} exceeds {args.max_wer:.4f} for {clip_key}: {asr_result['text']!r}")
        clip_arrays[clip_key] = pcm
        clip_info[clip_key] = {
            "duration_samples": int(pcm.size),
            "words": asr_result["words"],
            "asr_transcript": asr_result["text"],
            "quality": quality,
        }
        segment_output[clip_key] = {
            "scenario_id": source["scenario_id"],
            "segment_id": source["segment_id"],
            "speaker_id": source["speaker_id"],
            "script_text": source["text"],
            "generated_clip_sha256": quality["sha256"],
            "quality": quality,
        }
        saved_path = output_dir / "segments" / f"{clip_key}.wav"
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clip_path, saved_path)
        if hashlib.sha256(saved_path.read_bytes()).hexdigest() != quality["sha256"]:
            raise RuntimeError(f"copied segment hash mismatch for {clip_key}")

    dialogue_reports: list[dict[str, Any]] = []
    all_frames = 0
    try:
        for scenario in SCENARIOS:
            scenario_id = str(scenario["scenario_id"])
            clip_info_for_scenario = {
                str(segment["segment_id"]): clip_info[f"{scenario_id}__{segment['segment_id']}"]
                for segment in scenario["segments"]
            }
            segments, events = _resolve_timeline(scenario, clip_info_for_scenario)
            local_clips = {
                str(segment["segment_id"]): clip_arrays[f"{scenario_id}__{segment['segment_id']}"]
                for segment in scenario["segments"]
            }
            stereo = compose_stereo_timeline(local_clips, segments)
            spans = _build_transcript_spans(segments, clip_info_for_scenario)
            frames = build_frame_annotations(
                scenario_id, stereo.shape[0], segments, events, transcript_spans=spans
            )
            media_path = output_dir / "media" / f"{scenario_id}.wav"
            final_audio_quality = _atomic_write_audio(media_path, stereo, channels=2)
            frame_path = output_dir / "frames" / f"{scenario_id}.jsonl"
            _write_frames(frame_path, frames)
            media_info = sf.info(str(media_path))
            media_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()
            if media_info.samplerate != SAMPLE_RATE or media_info.channels != 2 or media_info.subtype != "PCM_16":
                raise RuntimeError(f"final dialogue WAV contract failed: {media_path}")
            if media_info.frames <= 0:
                raise RuntimeError(f"empty dialogue WAV: {media_path}")
            dialogue_report = {
                "conversation_id": scenario_id,
                "synthetic": True,
                "sample_rate_hz": SAMPLE_RATE,
                "channels": 2,
                "channel_map": {"0": "user", "1": "assistant"},
                "duration_s": media_info.frames / SAMPLE_RATE,
                "frame_count": len(frames),
                "media_path": media_path.relative_to(output_dir).as_posix(),
                "media_sha256": media_hash,
                "audio_quality": final_audio_quality,
                "frame_annotations_path": frame_path.relative_to(output_dir).as_posix(),
                "frame_annotations_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                "events": events,
                "transcript_spans": spans,
                "segments": [
                    {
                        **segment_output[f"{scenario_id}__{segment['segment_id']}"],
                        "start_sample": segment["start_sample"],
                        "end_sample": segment["end_sample"],
                        "start_s": segment["start_sample"] / SAMPLE_RATE,
                        "end_s": segment["end_sample"] / SAMPLE_RATE,
                        "audible_duration_s": (segment["end_sample"] - segment["start_sample"]) / SAMPLE_RATE,
                        "truncated": segment["truncated"],
                        "truncated_by_event": segment.get("truncated_by_event"),
                    }
                    for segment in segments
                ],
                "actions_covered": sorted({frame["target_action"] for frame in frames}),
            }
            dialogue_reports.append(dialogue_report)
            all_frames += len(frames)

        manifest = {
            "schema_version": 1,
            "dataset_id": "DuplexIT-Qwen3TTS-Synthetic-Candidate",
            "status": "SYNTHETIC_CANDIDATE_READY",
            "synthetic": True,
            "t0_eligible": False,
            "t0_disposition": "BLOCKED: canonical suite requires manually annotated, non-synthetic held-out data",
            "generator": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "qwen_model_snapshot": str(model_snapshot),
            "asr_engine": "faster-whisper",
            "asr_model_snapshot": str(asr_snapshot),
            "sample_rate_hz": SAMPLE_RATE,
            "frame_window_ms": FRAME_MS,
            "channels": 2,
            "channel_map": {"0": "user", "1": "assistant"},
            "annotation_method": "synthetic_scripted_timeline_plus_faster_whisper_word_timestamps",
            "speaker_references": references,
            "quality_gates": {
                "max_wer": args.max_wer,
                "max_observed_wer": max(
                    item["quality"]["wer"] for item in segment_output.values()
                ),
                "all_segments_asr_passed": all(item["quality"]["asr_passed"] for item in segment_output.values()),
                "max_clipping_pct": max(item["quality"]["clipping_pct"] for item in segment_output.values()),
                "all_wavs_sha256_verified": True,
            },
            "dialogue_count": len(dialogue_reports),
            "total_audio_hours": sum(item["duration_s"] for item in dialogue_reports) / 3600.0,
            "total_frames": all_frames,
            "actions_covered": sorted({action for item in dialogue_reports for action in item["actions_covered"]}),
            "dialogues": dialogue_reports,
        }
        _atomic_write_json(output_dir / "manifest.json", manifest)
        shutil.rmtree(temp_root)
        print(json.dumps({key: manifest[key] for key in ("status", "synthetic", "t0_eligible", "dialogue_count", "total_audio_hours", "total_frames", "quality_gates")}, ensure_ascii=False), flush=True)
        return manifest
    except Exception as exc:
        _atomic_write_json(
            output_dir / "generation_failure.json",
            {"status": "FAILED", "stage": "timeline_or_write", "error": f"{type(exc).__name__}: {exc}", "synthetic": True},
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-a-reference", type=Path, required=True, help="speaker A/user mono WAV; must already exist on the RTX 3090")
    parser.add_argument("--speaker-b-reference", type=Path, required=True, help="speaker B/assistant mono WAV; must already exist on the RTX 3090")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/artifacts/duplex_qwen3tts_synthetic_candidate"))
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--asr-snapshot", type=Path)
    parser.add_argument("--phase1-python", type=Path, default=Path("/opt/venv_phase1/bin/python"))
    parser.add_argument("--worker", type=Path, default=Path("/workspace/tools/generate_scenario_audio_qwen3tts_phase1_worker.py"))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-wer", type=float, default=0.15)
    parser.add_argument("--resume", action="store_true", help="reuse existing generated clips in output_dir/.work if present")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _safe_write_candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
