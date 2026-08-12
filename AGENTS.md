# AGENTS.md — Gemma 4 Full-Duplex Omni Pipeline

## Regole operative di base

- NON eseguire mai il download locale di modelli più grandi di 1GB. Il computer di esecuzione locale è lento e questo creerebbe colli di bottiglia inutili. L'esecuzione della pipeline avverrà interamente in Kaggle.
- Genera codice come fossero celle Kaggle, quindi un singolo modulo python per la pipeline principale, con separatori netti delle varie celle, e moduli secondi a sé stanti per celle necessarie ma scollegate dalla pipeline principale.
- Il tempo GPU Kaggle è costoso: mai lanciare run di training completi senza prima validare la pipeline con smoke test e test su pochi sample.

## Principi di sviluppo obbligatori (istruzione per tutti gli agenti futuri)

Ogni agente che lavora su questo progetto DEVE applicare i tre principi seguenti. Sono vincolanti, non opzionali.

### 1. Doubt-Driven Development (DDD)

Una risposta sicura non è una risposta corretta. Ogni decisione non banale viene sottoposta a una revisione avversariale a contesto fresco PRIMA di essere considerata valida.

- Una decisione è **non banale** se: introduce o modifica logica ramificata, attraversa un confine di modulo, afferma una proprietà che il type system non può verificare, dipende da contesto che il lettore futuro non vede, o ha un raggio d'impatto irreversibile (deploy, migrazione dati, run GPU costose, cambi di API).
- Processo (per ogni ciclo): **CLAIM** (nome la decisione in 2-3 righe e perché conta) → **EXTRACT** (isolo l'artefatto minimo + il contratto, senza il mio ragionamento) → **DOUBT** (revisore avversariale a contesto fresco: "trova cosa c'è di sbagliato, NON validare"; mai passare il CLAIM al revisore, solo ARTEFATTO + CONTRATTO) → **RECONCILE** (classifico ogni finding: contratto ambiguo / valido e azionabile / trade-off valido / rumore; il revisore non è infallibile: rileggo l'artefatto e decido io) → **STOP** (solo finding triviali, 3 cicli massimi, o override dell'utente).
- In contesto interattivo: dopo ogni revisione single-model, OFFRIRE esplicitamente un secondo parere cross-model (Gemini CLI, Codex CLI, revisione esterna manuale, o skip). In contesto non interattivo: annunciare lo skip. Mai invocare CLI esterne senza autorizzazione esplicita dell'utente.
- Se un test fallisce (RED di TDD), quel test fallito È il passo di dubbio per le affermazioni comportamentali.
- Non applicare DDD a operazioni meccaniche (rinominare, formattare, spostare file, eseguire tool). Altrimenti non si spedisce nulla.

### 2. Source-Driven Development (SDD)

Ogni decisione di codice specifica per un framework/libreria DEVE essere fondata sulla documentazione ufficiale, con citazione. Mai implementare a memoria.

- Processo: **DETECT** (identifico stack e versioni esatte dal file delle dipendenze; se le versioni sono ambigue chiedo all'utente) → **FETCH** (recupero la pagina di documentazione SPECIFICA, non la homepage; gerarchia di autorità: 1. documentazione ufficiale, 2. blog/changelog ufficiale, 3. reference web standard, 4. dati di compatibilità; MAI Stack Overflow, tutorial, riassunti AI o la mia memoria come fonte primaria) → **IMPLMENT** (codice conforme ai pattern documentati per QUELLA versione; se i docs confliggono col codice esistente, lo segnalo all'utente) → **CITE** (citazione con URL completo, quote del passo rilevante, note di deprecazione).
- Se non trovo documentazione ufficiale per un pattern: lo dichiaro esplicitamente UNVERIFIED invece di fare finta di essere sicuro.
- Versioni chiave del progetto (riferimento): `qwen-tts` richiede `transformers==4.49.0` (Fase 1); Unsloth + Gemma 4 E2B richiedono `transformers>=5.x` (Fase 2, kernel riavviato). ADR-003 regola questa separazione.

### 3. Test-Driven Development (TDD)

Comportamento guidato dai test: RED → GREEN → REFACTOR.

- Prima scrivo un test che fallisce (RED) e dimostro che fallisce per il motivo giusto; poi implemento il minimo per farlo passare (GREEN); poi rifattorizzo (REFACTOR) senza cambiare comportamento.
- Il test fallito è il "dubbio materializzato": è il meccanismo che rende il dubbio verificabile e la correzione provata.
- Su Kaggle (GPU a ore): ogni run che consuma GPU deve essere preceduto da uno smoke test. Gli smoke test sono OBBLIGATORI prima di qualsiasi run di training completo.

## Regole di esecuzione su Kaggle (valide per tutti gli agenti)

- **Smoke test Fase 1**: ridurre `DUPLEXCONV_MAX_HOURS` (es. 1-2h → pochi segmenti) e `MDC_MAX_SAMPLES` (es. 8-16) per validare download, tokenizzazione RVQ reale (Qwen3-TTS 12Hz), estrazione mel numpy (bit-pari col Gemma4AudioFeatureExtractor), estrazione Code2Wav + sub-talker e scrittura parquet/sidecar. Controllare che `TOKENIZER_REAL_ENCODING_ACTIVE=True` e che la distribuzione dei target NON sia degenerata (niente loss collassata ~0.002).
- **Smoke test Fase 2**: usare `PHASE2_SMOKE_BATCHES` (es. 2-4) per validare forward/backward, coupling (sub-talker congelato), loss testo, validation split e checkpoint prima di un run completo.
- **Gate**: si passa al run completo SOLO quando la pipeline non crasha, è corretta metodologicamente e teoricamente nelle operazioni che compie (verificato con DDD+SDD+TDD). In caso di dubbi sulla correttezza: fermarsi e chiedere.
- **Debugging**: usare `debugging-and-error-recovery` per root-cause sistematico; mai fix a tentativi (niente "guess and pray").
- **Checkpoint & metriche**: ogni run salva checkpoint in `checkpoints/`, wav ascoltabili in `samples/` e report. Confrontare i risultati col report precedente (`gemma4_training_report.md`). Loss iniziale attesa ~13-17 (log 4096 + log 2048), NON ~0.002.
- **Comunicazione**: alla fine del lavoro informare l'utente via email (estensione email-status / send_status_email), con stato (fatto/in corso/bloccato), file toccati, test eseguiti, prossimi passi.
