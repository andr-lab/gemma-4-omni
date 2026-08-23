# RETRAKE VERDETTO TASK 11H (23 ago 2026, ore 09:15)

Il verdetto "generated v3 INTELLIGIBILE 100%" contenuto in REPORT_TASK11H_VOCI_V3.md
è **RITIRATO**: era basato su un ascolto di agy CONTAMINATO (testo atteso nel prompt
+ memoria persistente → agy ha echo-ato l'attesa, non l'audio).

## Verifica cieca (blind ASR whisper-small/faster-whisper, nessun testo atteso)
| File | Trascrizione cieca | Esito |
|---|---|---|
| oracle_003 | "Continente Pangea. Questa è volte si ben escartata." | ~70% ok |
| generated_003 | "lo che è la cosa è non è un'altra cosa non è un'altra cosa" | LOOP DEGENERATO |
| oracle_001 | "Il perizoma è dorato all'esterno e dipinto di rosso all'interno." | 100% ok |
| generated_001 | "...di foto, assolutamente, non era lontano..." | contenuto errato |
| generated_005 | (vuoto) | silenzio |

## Conclusione corretta
Il problema "voce robotica/incomprensibile" NON è risolto dall'adapter v3.
Coerente con Top-1 10,6% sui codici semantici: predizioni in gran parte errate,
l'argmax autoregressivo le amplifica in loop degenerati.

## Lezione di protocollo (vincolante per il futuro)
Il giudice percettivo NON deve MAI ricevere il testo atteso prima della
trascrizione. Il confronto avviene SOLO dopo, lato coordinatore.
