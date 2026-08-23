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

## APPENDICE 23/8 ore 12:30 — Diagnostica ibrida + ricetta originale + TurnHead v5

### 1. Ricetta originale (Task 11I, heads v2) su CV26 40.8h
cos sim val 0.9598 (> benchmark 0.7455) MA blind ASR sui generated:
silenzio o italiano fluente con contenuto allucinato. Stesso fallimento della v3.

### 2. Diagnostica ibrida (cb0 GT + livelli 1..15 da sub-talker su h_t predetto)
Voce naturale ma contenuto SBAGLIATO anche con cb0 ground truth.
→ Il guasto è nell'interfaccia h_t→sub-talker, non solo nella predizione cb0.
→ L'unica catena che produce contenuto corretto resta il decode full-GT (oracle).

### 3. TurnHead v5 (GT nativo): due run entrambi collassati
Run 1 (YIELD weight 6.0): predice YIELD al 98%. Run 2 (pesi moderati): BARGE_IN 85%.
Le feature di input (spettro mel medio/frame) non discriminano le classi native reali.

### Lezione metodologica
Le metriche latenti (cos sim, mel-l1) NON predicono la qualità percettiva
dell'inferenza libera. Gate obbligatorio: blind ASR + ascolto utente.
