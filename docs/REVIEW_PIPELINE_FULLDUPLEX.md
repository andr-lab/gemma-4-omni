# REVIEW PIPELINE FULL-DUPLEX — TurnHead (DuplexConv) + Adapter (MDC/CV26)

**Data**: 22 agosto 2026 | **Oggetto**: verifica dell'approccio composito per Gemma 4 E2B full-duplex
**Domanda**: l'accoppiata "TurnHead allenata su DuplexConv + Adapter allenato su MDC" può produrre un agente che **ascolta, capisce quando intervenire, si interrompe (barge-in/yield), emette backchannel**?

---

## 1. VERDETTO ESECUTIVO

```
====================================================================================
VERDETTO: L'APPROCCIO COMPOSITO È DIREZIONALMENTE CORRETTO MA ATTUALMENTE NON
FUNZIONA COME SISTEMA FULL-DUPLEX, PER 4 MOTIVI STRUTTURALI:

  R1. I DUE COMPONENTI NON SONO INTEGRATI: serve_duplex.py NON usa TurnHead.
      Sono due artefatti disgiunti addestrati su distribuzioni diverse, mai
      connessi a inferenza. Oggi il sistema È half-duplex request/response.

  R2. L'ADAPTER È UN MONOLOGO TTS: addestrato SOLO su finestre con parlato
      attivo agente (MDC/CV26 = lettura single-speaker). Non sa produrre
      silenzio, non sa ascoltare, non ha alcuna nozione di turno. Può solo
      "parlare continuamente" — il muting deve avvenire ESTERNAMENTE.

  R3. TURNHEAD NON È ABASTANZA ACCURATA PER CONTROLLARE NULLA: Balanced Acc
      41,24%, YIELD recall 40,11% con PRECISION ~19%. Un controllore che
      sbaglia il 60% delle cessioni e ha l'81% di falsi allarmi su YIELD
      spezzerebbe il flusso continuamente. Sotto ~85-90% balanced acc una
      head di controllo non è utilizzabile in linea.

  R4. LE ETICHETTE DI TURN-TAKING SONO EURISTICHE, NON UMANE: YIELD = ultimi
      3 frame prima del silenzio; BARGE_IN = overlap ≥5 frame; USER_BC =
      overlap <5 frame; EMIT_BC = cadenza periodica artificiale (t%12∈{0,1,2}).
      Il modello non sta imparando la conversazione umana: sta imparando un
      energy-VAD con regole scritte da noi. Le metriche v4 misurano quanto
      bene riproduce le NOSTRE euristiche, non la competenza dialogica.
====================================================================================
```

La composizione "controller esterno + generatore" resta però **l'architettura pragmaticamente giusta** per questo progetto (vedi §6): è il pattern di Moshi/LSLM semplificato ed è quello verso cui convergere anche i sistemi cascaded recenti (DuplexCascade). Il problema non è l'idea, sono i dettagli di supervisione e integrazione.

---

## 2. STATO ATTALE — INVENTARIO DEI FATTI

### 2.1 Componenti esistenti

| Componente | Checkpoint | Dati | Metrica | Ruolo previsto |
|---|---|---|---|---|
| Gemma 4 E2B | frozen (QLoRA solo in fase sperimentale) | — | — | backbone semantico |
| TemporalResidualAdapter (~59M) | `gemma4_duplex_sota_v2` (cos sim val 0.9366) | MDC v2 15h (CV17 it mono) | mel cos-sim | proiezione hidden→h_t per sub-talker |
| Sub-talker Qwen3-TTS (181M) | frozen | — | — | h_t → codici RVQ 1..15 |
| Code2Wav | decoder ufficiale via subprocess (Task 11E) | — | gate 18/18 PASS | codici → wav |
| TurnStateHead v4 | `turnhead_v4` | DuplexConv 20h stereo | acc 66,76%, bACC 41,24%, YIELD recall 40,11% | classificatore 5 stati/frame |
| Serving | `serve_duplex.py` (215 righe) | — | — | REST /talk half-duplex |

### 2.2 Il problema percettivo aperto (feedback utente 22/8)
Oracle (codici GT) = voce chiara ✅; Generated (codici predetti dall'adapter) = robotica ❌.
L'adapter predice male i codici nonostante mel cos-sim alta → mismatch di ottimizzazione
(mel-space loss, code-space usage) + error accumulation autoregressiva. In corso: Task 11G.

---

## 2.3 STRUTTURA REALE DI DUPLEXCONV (README ufficiale HuggingFace)

Fonte: https://huggingface.co/datasets/qualialabsAI/DuplexConv (README, letto integralmente 22/8).

**Fatti dichiarati dal README** (divergono dal nostro doc interno `ANALISI_DUPLEXCONV.md`, che risulta errato su scala e durate):

| Proprietà | Valore ufficiale | Nota vs nostro uso attuale |
|---|---|---|
| Lingua | **Cinese (zh)** | ⚠️ conferma il rischio lingua: nessun trasferimento diretto a it |
| Durata totale | **2.000,21 ore** | noi ne usiamo 20h (1%) — enorme margine di scala |
| File audio | 93.709 | multi-party (non solo diadi) |
| Durata media | 76,84 s (range 8–618 s) | il doc interno diceva "5-30 s": ERRATO |
| Domini | Tutoring, Social Chat | cartelle `Edu*/` = tutoring, `none_Edu*` = chat |
| Annotazione | LLM-assisted | companion SmoothConv = **100h con ANNOTAZIONE UMANA ESPERTA** |
| Licenza | CC BY-NC 4.0 | solo ricerca, ok per noi |

**Schema annotazioni** (`wavInfo` per ogni wav multi-canale):
```
wavInfo
├── nTrack, timeLenInSec, fs          # canali, durata, sample rate
├── vadFrmLenInMs                     # granularità VAD (ms)
├── vadFlagPerFrmPerTrack (nFrm×nTrk) # ★ VAD frame-level PER CANALE — non usato da noi
└── asr[track][utterance]             # usato parzialmente da noi
    ├── txt, startInMs, endInMs, LID          # ← usati (testo + timing)
    ├── state: <|complete|>/<|incomplete|>    # ★ stato del turno — non usato
    ├── speaker.segments (diarizzazione)      # ★ non usato
    ├── labels: gender, age, emotion,
    │          accent, paralinguistic, txt    # ★ emotion/paralinguistic non usati
    └── snr, mos, privacyFlag, AED_dasheng    # ★ filtri qualità — non usati
```

**Cosa cambia per la pipeline**:
1. **TurnHead rieducabile subito con GT nativo**: VAD frame-level per canale → SPEAK/LISTEN/OVERLAP senza soglie energetiche; `state=incomplete` durante silenzio agente → KEEP_WAITING; `complete` + fine utterance utente → YIELD/response. Le 5 classi diventano derivate da annotazioni reali invece che da regole.
2. **SmoothConv come validation set**: le 100h con annotazione umana esperta sono il benchmark ideale per misurare quanto le label LLM-assisted (e le nostre derivate) sono affidabili.
3. **Scala**: da 20h a centinaia di ore disponibili se serve più dato per la policy.
4. **Vincolo lingua confermato**: tutto il dialogo è cinese → la policy appresa va adattata all'italiano (dialoghi it sintetici dual-channel col nostro stack o dati futuri); la struttura di supervisione però resta identica.

| Requisito | Cosa serve | Cosa c'è oggi | Gap |
|---|---|---|---|
| **Ascoltare** (mentre l'utente parla, senza parlare) | generazione h_t → silenzio o muting controllato | adapter addestrato SOLO su frame agent-attivi; finestre di ascolto scartate dal training (`_mel_active_mask`) | **TOTALE**: il modello non ha mai visto lo stato "ascolto" |
| **Capire quando intervenire** (end-of-turn dell'utente) | rilevare fine turno utente con semantica (esitazione vs fine) | TurnHead classe SPEAK/YIELD ma su euristiche energetiche; nessuna discriminazione esitazione/pausa sintattica | alto: servono label semantiche (punteggiatura ASR + timing reale) |
| **Interrompersi** (barge-in dell'utente) | arresto generazione <200ms dal subentrare voce competitiva | TurnHead classe BARGE_IN (recall non riportato come gate; confusione forte); serving senza loop streaming decisionale | alto: decisione frame-level causale assente in inferenza |
| **Non interrompersi** (backchannel/tosse utente ≠ turno) | distinguere overlap collaborativo vs competitivo | classe USER_BC separata, ma deriva da durata overlap (<5 frame), non da contenuto semantico/prosodico | alto: richiede comprensione semantica del contributo breve |
| **Backchannel propri** ("mhm", "sì") | emissione micro-turno 300-600ms durante ascolto | classe EMIT_BC = cadenza periodica fittizia (t%12); sub-talker non addestrato a micro-emissioni | **artificiale**: la label non corrisponde a comportamento umano reale |
| **Yield** (cedere il turno a fine risposta) | EOS acustico / transizione a silenzio | YIELD = "ultimi 3 frame prima del silenzio" (definizione circolare: si impara a prevedere il silenzio imminente dall'energia) | medio-alto |

Conclusione della tabella: **nessun requisito full-duplex è oggi soddisfatto end-to-end**, e due di essi (ascolto, backchannel) non hanno nemmeno una supervisione valida nel dataset attuale.

---

## 4. BENCHMARK LETTERATURA (docs/sources + FireRedTTS3)

### 4.1 Tassonomia del Survey (A Survey of Full-Duplex Spoken Dialogue Systems)
Tre generazioni:
- **Gen-2 (L2 token-level)**: duplex codificato NELLA sequenza token — **Moshi** (Mimi 12.5Hz 8 RVQ, RQ-Transformer multi-stream, stream utente+agente interleavati temporalmente), LSLM, OmniFlatten, SyncLLM, Mini-Omni 2.
- **Gen-3**: Moshi/LSLM hanno resettato lo standard "learned" del turn-taking.
- **dGSLM**: due stream parlato paralleli, primo dimostratore VAD-free.

Punto chiave del survey: i sistemi funzionanti codificano la decisione duplex **al livello dei token** (multi-stream o token di controllo), NON con head ausiliarie esterne. Nessun sistema SOTA usa una "head di stato" sganciata dal flusso generativo principale come facciamo noi.

### 4.2 DuplexCascade (2026) — il pattern pragmatico
Cascaded ASR-LLM-TTS reso full-duplex tramite:
1. conversione turn lunghi in **micro-turni chunk-wise**;
2. **token conversazionali espliciti** che regolano il comportamento LLM (VAD-free turn-taking);
3. **LoRA text-only su 50k dialoghi, 5k step** — l'adattamento testuale evita i problemi di cross-modal alignment.

Risultato: SOTA full-duplex su Full-Duplex-Bench + intelligenza conversazionale preservata. **Rilevantissimo per noi**: ottiene il duplex SENZA ritoccare il modulo vocale, spostando tutta la policy sui token di controllo — esattamente la separazione controller/generatore che vogliamo, ma con la policy DENTRO l'LLM (che vede testo+timing) invece che in una head acustica esterna.

### 4.3 FireRedTTS3 (arXiv:2608.17492, 18 ago 2026)
Continuous AR-TTS con **Audio Encoder congelato come "semantic teacher"** che regolarizza lo spazio delle feature: migliora text-speech alignment e **mitiga l'error accumulation autoregressiva** direttamente a livello di rappresentazione.

Trasposizione al nostro problema: la voce robotica del generated (feedback utente) è esattamente error accumulation + mancanza di ancoraggio semantico nella predizione dell'adapter. Un teacher semantico congelato (es. encoder ASR multilingue, o il livello semantico RVQ-0 del tokenizer stesso) come regularizzatore su h_t/predizioni è un fix supportato dalla letteratura recente — da integrare nel Task 11G oltre al CE multi-codebook.

### 4.4 Mini-Omni 2 / SyncLLM / LM-Enhanced DM
- **Mini-Omni 2**: duplex per intercaltazione di canali (user input + agent output + moduli) su schema temporale condiviso.
- **SyncLLM** (Beyond Turn-Based Interfaces): sincronizzazione token-level tra stream utente/agente.
- **LM-Enhanced Dialogue Management**: l'LM decide esplicitamente gli stati dialogici (approccio ibrido cascata+policy).

Sintesi trasversale: **tre famiglie** — (A) multi-stream nativo (Moshi), (B) token di controllo su cascata (DuplexCascade, LM-Enhanced DM), (C) head/policy esterna (la nostra). La (C) non ha rappresentanti forti in letteratura proprio perché separare la policy dal generatore perde le informazioni semantiche che servono per decidere.

---

## 5. DIAGNOSI DEI 4 PROBLEMI RADICE (con evidenza)

### R1 — Non-integrazione (fatto verificabile)
`serve_duplex.py`: zero occorrenze di `turn_head`/`YIELD`/`BARGE_IN`. Endpoint `/talk`: riceve wav utente → risponde wav agente. Mezzo turno per richiesta. Anche se TurnHead fosse perfetta, oggi non influenzerebbe nulla.

### R2 — Adapter monologo (fatto verificabile)
Proposal interna (docs/PROPOSAL_FULLDUPLEX_DINAMICHE.md §2.1, confermata nel codice): `_mel_active_mask` scarta le finestre senza energia agente → la loss vede SOLO parlato agente. MDC/CV26 sono inoltre dataset di **lettura single-speaker**: zero sovrapposizioni, zero cambi turno reali, zero backchannel. L'adapter è un compressore TTS, non un modello dialogico. Questo È per progetto (il compito dell'adapter è la qualità vocale), ma implica che **tutta la dinamica deve stare nel controller** — e il controller attuale non basta (R3/R4).

### R3 — Accuratezza insufficiente per il controllo
TurnHead v4: bACC 41,24%; YIELD recall 40,11% / precision ~18,9% (confusion matrix Task10). Per un policy gate servono: FALSI YIELD quasi nulli (altrimenti l'agente si zitta a metà frase continuamente) e recall alto sui BARGE-IN veri. Allo stato attuale l'uso in linea peggiorerebbe l'esperienza rispetto a un semplice VAD con isteresi.

### R4 — Supervisione euristica
Le 5 classi derivano da regole scritte a mano su energia/durata/cadenza (`turnhead_v3_cell2.py`, `_extract_window_features_and_labels_v3`, ramo fallback; anche dove esistono turnlabels precomputati da ASR, la SEMANTICA delle classi resta definita da quelle regole). Due conseguenze:
1. il ceiling di accuratezza è il grado in cui le regole approximano il comportamento umano (basso: la cadenza periodica degli EMIT_BC e la definizione "ultimi 3 frame" di YIELD non corrispondono alla fenomenologia reale);
2. anche un modello 100% accurato sulle regole NON sarebbe un buon dialogante, perché le regole sono sbagliate rispetto all'obiettivo.

Aggiunta (scoperta Task 11E): DuplexConv contiene lunghissime finestre di silenzio/ascolto (USER_BACKCHANNEL=0, 14 token distinti CB0 = codici di silenzio) → il bilanciamento dei bucket è ulteriormente distorto.

### R4-bis — AGGRAVANTE: le euristiche erano INUTILI, il dataset fornisce già la verità a livello frame

L'analisi del README ufficiale HF (`qualialabsAI/DuplexConv`, vedi §2.3) rivela che il dataset contiene **annotazioni native** che coprono esattamente ciò che abbiamo simulato con regole energetiche:

| Ciò che abbiamo euristico | Cosa fornisce DuplexConv e NON usiamo |
|---|---|
| VAD per energia (`u_energy > percentile`) | `wavInfo.vadFlagPerFrmPerTrack`: **VAD frame-level per canale**, già pronto `(nFrm × nTrack)` |
| YIELD = "ultimi 3 frame prima del silenzio" | `asr[t][i].state = <|complete|>` / `<|incomplete|>`: **stato di completamento del turno per utterance** — la discriminazione fine-turno vs esitazione che l'energia non può fare |
| USER_BC/BARGE_IN per durata overlap | `speaker.segments` (diarizzazione per utterance, numSpeakers/multiSpeaker) |
| EMIT_BC a cadenza fittizia t%12 | `labels.paralinguistic`, `labels.emotion` (feedback vocale etichettato dall'LLM annotatore) |
| — | `snr`, `mos`, `AED_dasheng` (filtri qualità/eventi acustici) |

Conclusione: R3 (accuratezza insufficiente) e R4 (supervisione artificiale) hanno la **stessa radice evitabile**: la head è stata addestrata su proxy sintetici quando il ground-truth nativo esisteva negli stessi JSON che già scarichiamo (~1GB estratti). Rifare le label da `vadFlagPerFrmPerTrack` + `state` è un compito di estrazione, non di raccolta dati.

---

## 6. VERIFICA DELL'APPROCCIO PROPOSTO E RACCOMANDAZIONI

### 6.1 L'idea "controller + generatore" va mantenuta, ma con la policy dentro l'LLM

La letteratura dice: la decisione di turno ha bisogno di semantica (cosa sta dicendo/ha detto l'utente, se la pausa è sintatticamente completa), non solo prosodia. Una MLP su feature spettrali medie (e_user = media mel per frame!) non può vedere la semantica. Gemma 4 invece sì — è lì che risiede l'intelligenza del sistema.

**Architettura raccomandata (Path C+ — "DuplexCascade-like sul nostro stack")**:

```
┌────────────────────────────────────────────────────────────────────┐
│  LOOP STREAMING FRAME-LEVEL (80ms)                                 │
│                                                                    │
│  audio utente ──► [ASR streaming leggero o codici RVQ-0 utente]    │
│       │                    │                                       │
│       ▼                    ▼                                       │
│  Gemma 4 E2B (QLoRA leggera) ◄── testo parziale + TOKEN CONTROLLO  │
│       │            <yield> <keep_waiting> <backchannel_mhm>        │
│       │            <barge_in_ack> <continue>                       │
│       │                                                            │
│       ├──► stato: SPEAK ──► adapter → sub-talker → codici → wav    │
│       └──► stato: LISTEN/YIELD ──► MUTING (silenzio forzato)       │
│                                                                    │
│  TurnHead degrada a FEATURE ausiliaria (prior prosodica soft)      │
│  fornita a Gemma, NON a gate duro                                  │
└────────────────────────────────────────────────────────────────────┘
```

Perché questa e non le alternative:
- **Path A (Moshi multi-stream nativo)**: richiederebbe riaddestrare Gemma su sequenze dual-stream interleaved con codici RVQ — fuori scala per 3090/24GB con QLoRA e fuori budget-tempo; è il salto di generazione giusto ma non ora.
- **Path B (tenere TurnHead come unico controller)**: bloccata da R3+R4; anche con label migliori, una head acustica non vedrà mai la differenza tra "pausa per esitazione" e "fine turno", che è IL problema centrale del turn-taking (documentato anche in DuplexCascade §1: VAD/energia falliscono proprio lì).
- **Path C (policy a token nell'LLM)**: provata SOTA su Full-Duplex-Bench con solo LoRA testuale (50k dialoghi, 5k step); riusa la forza di Gemma; mantiene l'adapter/sub-talker come puro motore vocale (il loro miglioramento resta ortogono, Task 11G); i token di controllo mappano 1:1 ai 5 stati già modellati.

### 6.2 Piano d'azione concreto (ordinato)

1. **[11G — in corso] Fix adapter (qualità vocale)**: CE multi-codebook o ri-encode corretto + **regularizzatore semantic-teacher stile FireRedTTS3** (encoder congelato sulla h_t/predizione). Senza voce comprensibile, ogni dinamica è inutile. Gate: ascolto utente.
2. **[Nuovo task] Etichette turn-taking REALI dai campi nativi DuplexConv** (aggiornato post-lettura README): NON serve più inventare supervisione —
   - `vadFlagPerFrmPerTrack` → SPEAK/LISTEN/OVERLAP frame-level esatti (niente soglie energetiche);
   - `state` (<|complete|>/<|incomplete|>) per utterance utente durante silenzio agente → YIELD (fine turno vero) vs KEEP_WAITING (esitazione) — la discriminazione che l'energia non può fare;
   - `speaker.segments` → USER_BC vs BARGE_IN da diarizzazione reale;
   - validare le label LLM-assisted su SmoothConv (100h annotate da umani esperti);
   - EMIT_BC: se i dati reali non contengono abbastanza backchannel agente, ELIMINARE la classe e gestirla come micro-turno generato (vedi punto 4).
3. **[Nuovo task] Addestramento duplex dell'adapter — modalità silenzio**: includere le finestre di ascolto con target = codici di silenzio (già osservati nei dati: CB0 con token dedicati). L'agente deve poter emettere silenzio "vero" quando in LISTEN, così il muting diventa una scelta del modello e non solo un taglio server-side.
4. **[Nuovo task] Policy a token (Path C)**: dataset micro-turni da DuplexConv (transizioni di stato per frame → token di controllo allineati), QLoRA leggera di Gemma sui soli token di controllo inseriti nel contesto; valutare con Full-Duplex-Bench (usato da DuplexCascade, metriche standard).
5. **[Integrazione serving]** riscrivere `serve_duplex.py` come loop streaming 80ms con: buffer circolare (ADR-002 già pronto), chiamata policy per frame, muting istantaneo su YIELD/BARGE_IN, latenza target <200ms misurata.
6. **TurnHead**: riciclata come prior/feature prosodica soft (input aggiuntivo a Gemma) oppure tenuta come fallback VAD semantico; NON più gate duro finché bACC < 85%.

### 6.3 Rischi e mitigazioni
- **Qualità ASR per la policy testuale**: DuplexConv ha annotazioni ASR già pronte; per l'inferenza live serve uno streaming ASR leggero (whisper-small/streaming o i codici RVQ-0 semantici utente come input diretto — opzione preferibile perché già nel nostro spazio).
- **Token control vs capacità E2B (2B)**: DuplexCascade dimostra che bastano 5k step LoRA testuali; rischio basso.
- **Dataset italiano**: DuplexConv è **cinese (confermato dal README)** → la policy appresa va validata/adattata su dialoghi it (dialoghi sintetici dual-channel it generati col nostro stesso stack, con auto-verifica percettiva, o CV26 mono come sorgente di voci); la STRUCTURE di supervisione (VAD frame + state + diarizzazione) resta però direttamente riutilizzabile.

### 6.2bis SCOPERTA CHIAVE POST-README — la scala c'è già

DuplexConv dichiara **2.000 ore** (noi usiamo 20h = 1%) con VAD frame-level e stati di turno nativi. Questo cambia la fattibilità del Path C+: il dataset micro-turni per la QLoRA della policy (punto 4) può essere costruito in poche ore di estrazione dagli stessi JSON già presenti su disco (~1GB estratti), con qualità di supervisione molto superiore alle euristiche attuali. Inoltre il companion **SmoothConv** (100h, annotazione umana esperta, stessi domini/formato) fornisce il validation set di alta qualità che ci manca per misurare l'affidabilità delle label.

---

## 7. CONCLUSIONE

Alla domanda "TurnHead(DuplexConv) + Adapter(MDC) può dare il full-duplex?", la risposta onesta è: **non nella forma attuale**. L'adapter (per progetto) e TurnHead (per limiti di supervisione e accuratezza) coprono insieme meno della metà dei requisiti, e i due pezzi non sono nemmeno collegati a inference. La direzione giusta — coerente con Moshi/DuplexCascade/FireRedTTS3 — è:

1. voce comprensibile (Task 11G, con semantic teacher),
2. adapter che sa tacere (silenzio come target),
3. policy di turno DENTRO Gemma tramite token di controllo addestrati su label reali da ASR,
4. serving streaming integrato con loop di decisione frame-level,
5. TurnHead retrocessa a segnale ausiliario finché non ridiventa abbastanza accurata.

Con questi 5 passi il sistema diventa realmente full-duplex sull'architettura che abbiamo già, senza ricostruzioni revolutionary.

---

## APPENDICE (22/8 sera) — Risultati empirici Task 11H

### Dataset THv5 con GT nativo: distribuzione classi reale
5997 segmenti / 6.099.319 frame etichettati da vadFlagPerFrmPerTrack + state:
SPEAK 76,87% | USER_BC 15,70% | BARGE_IN 7,11% | YIELD 0,33% | **EMIT_BC 0%**

### Scoperta strutturale su DuplexConv
Analisi di 400 JSON campionati: le utterance del canale agente NON scendono mai
sotto ~1,76s (p10) → il backchannel dell'agente come "emissione breve <0,8s"
NON ESISTE nei dati. Inoltre l'81% delle utterance agente sovrappone tratti
utente >2s → i due canali non sono turni alternati puliti ma conversazioni
multi-party ad alta sovrapposizione (coerente col README: multi-party).

### Conseguenze per TurnHead v5
1. Classi effettive: 4 (eliminare EMIT_BC o ridefinirlo su intervalli brevi relativi)
2. YIELD 0,33% richiede weighted sampling/focal loss (come fatto in v4)
3. La supervisione resta SUPERIORE alle euristiche v4 (VAD frame-level reale),
   ma il backchannel agent dovrà essere insegnato con dati dedicati (dialoghi it
   sintetici dual-channel col nostro stack) o micro-turni generati (Path C).
