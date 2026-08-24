# Ruolo dei Dataset: DuplexConv vs MDC26 (evidenze dalla letteratura)

**Data**: 2026-08-24 · **Domanda**: servono davvero entrambi? per quali componenti e in che modo?

## 1. Il caso di riferimento: J-Moshi (arXiv:2506.02979)

Adattare un full-duplex model (Moshi, inglese) a una lingua non-inglese (giapponese) —
**lo stesso identico problema del nostro italiano**. Ricetta documentata:

| Fase | Dati | Ore | Ruolo |
|---|---|---|---|
| Pre-training | J-CHAT (mono, diarizzato→stereo finto) | 69.000 | fonetica/lingua, capacità di base |
| Fine-tuning | Stereo reale (Callhome, CSJ, Travel, in-house) | 344 | **dinamiche full-duplex vere** (overlap, backchannel) |
| Augmentation | Stereo sintetico (multi-stream TTS da dialoghi testuali, selezione WER) | 602 | diversità dialogica |
| Codec (Mimi) | **CONGELATO** | — | *"Mimi has a certain level of ability to encode and resynthesize Japanese speech without additional training, so we froze Mimi's parameters"* |

**Tre evidenze decisive per noi:**

1. **Il codec/voce è lingua-agnostico e va congelato.** Mimi (equivalente del nostro
   sub-talker Qwen3-TTS) parla giapponese senza ri-addestramento. **→ Il timbro e la
   fonetica italiana di Qwen3-TTS NON si addestrano: si ereditano.** MDC26 non serve
   "per dare la voce italiana" — Qwen3-TTS ce l'ha già.

2. **Mono ≠ stereo.** I dati mono (anche 69.000h) NON contengono overlap/backchannel
   naturali: servono solo per lingua/fonetica. Le dinamiche full-duplex si apprendono
   SOLO da stereo reale (344h) o sintetico (602h). **→ DuplexConv (stereo, con VAD
   frame-level, turn state, interruzioni) è l'unico dato che può insegnare il
   turn-taking; MDC26 (mono, non conversazionale) non può.**

3. **Loss ratio semantic:acoustic = 100:1.** Moshi pesa i token SEMANTICI 100× più
   degli acustici (i PAD -50%). **→ La predizione del codebook semantico (il nostro
   cb0) è il componente da priorizzare nel training, non i residui.**

4. **Il turn-taking varia per lingua**: J-Moshi genera più overlap del Moshi inglese
   (5.0s vs 1.2s/min, coerente col GT giapponese 8.1s). → Le dinamiche apprese dal
   cinese (DuplexConv) NON trasferiscono automaticamente le statistiche di
   turn-taking italiane; servono dati stereo italiani (sintetici se non esistono).

## 2. Verdetto per i nostri dataset

### MDC26 (= Common Voice 26 italiano, mono, audio+trascrizioni)
- **RUOLO CORRETTO**: dati di *lingua/fonetica italiana* per:
  - **adapter acustico** (mapping Gemma→spazio Qwen su contenuto italiano) — già fatto (CV26 40.8h)
  - **predittore cb0** (distribuzione dei cb0 italiani) — già usato (cache ARS)
  - **ancora testuale**: le trascrizioni MDC = testo allineato per l'inner-monologue (P3)
- **RUOLO SBAGLIATO (da evitare)**: usarlo per "migliorare la voce italiana" del
  sub-talker (congelato, già italiano) o per il turn-taking (non ci sono conversazioni).
- **Conferma**: come J-CHAT in J-Moshi → serve, ma è il livello 1 (lingua), non il livello 2 (dinamiche).

### DuplexConv (HF qualialabsAI/DuplexConv, 2.000h cinesi, stereo, LLM-annotated)
- **COSA È**: 2.000h conversazioni cinesi multi-canale (Tutoring, Social Chat),
  VAD frame-level per canale, turn state (`<|complete|>`), interruzioni,
  backchannel, speaker segments, emotion — il dato di fine-tuning stereo ideale.
- **RUOLO CORRETTO**: insegnare le **dinamiche full-duplex** (come Callhome/CSJ per J-Moshi):
  - TurnHead / dialogue manager: supervisione reale su overlap, barge-in, yield (il v5
    usava etichette euristiche — qui c'è il GT nativo: VAD frame-level + turn state)
  - struttura di turno per il training multi-stream (se adottiamo il pattern Moshi)
- **LIMITE**: cinese, non italiano. Le statistiche di turn-taking NON trasferiscono
  automaticamente (evidenza J-Moshi §5). Uso ottimale:
  1. turn-taking a livello di struttura/policy (componente "quando parlare"),
  2. **sintesi stereo italiana** dal suo schema: dialoghi testuali italiani +
     Qwen3-TTS multi-stream → stereo italiano sintetico (pattern J-Moshi 602h /
     Moshi 20.000h sintetiche), selezionando con WER.
- **NOTA OPERATIVA**: il progetto l'ha GIÀ usato per TurnHead v4 (`train_tokens_20.0h_ext`,
  901k frame, z_t∈R^2304) — coerente con questo verdetto.

### Cosa manca (gap reale)
- **Stereo italiano reale o sintetico** per il turn-taking italiano. Opzioni:
  a) sintesi: dialoghi testuali italiani (PersonaChat-it, ecc.) → Qwen3-TTS multi-stream
     → stereo, selezione WER (J-Moshi §3.4: 602h, WER 24.6%);
  b) DuplexConv per la policy + italiano per la voce (ibrido);
  c) raccolta propria (fuori scope).

## 3. Implicazione per il design

```
Componente               Dati richiesti              Fonte
─────────────────────────────────────────────────────────────────────
Gemma (ragionamento)     testo/prompt                esistente (congelata)
Adapter acustico         italiano mono (lingua)      MDC26 ✅ (già fatto)
Predittore cb0           italiano mono (fonetica)    MDC26 ✅ (in uso)
Ancora testuale (P3)     trascrizioni allineate      MDC26 ✅ (in parquet)
Turn-taking (policy)     stereo con turn state       DuplexConv ⚠️ (cinese:
                                                     ok per struttura/policy,
                                                     serve stereo IT sintetico
                                                     per le statistiche IT)
Sub-talker / voce        NIENTE (congelato, IT)      ✅ Qwen3-TTS
```

**Conclusione**: MDC26 serve (per lingua/fonetica/adapter/cb0, NON per il timbro —
Qwen parla già italiano); DuplexConv serve (per le dinamiche di turn-taking, NON per
la fonetica — è cinese). Il gap da colmare per il full-duplex italiano è lo **stereo
italiano sintetico** (pattern J-Moshi: dialoghi testuali → Qwen3-TTS multi-stream → WER-selection).

---
Fonti: J-Moshi arXiv:2506.02979 (letto per intero); README DuplexConv HF
qualialabsAI/DuplexConv; report interni REPORT_TASK10_*.md; survey full-duplex
2606.19453 (integrazione subagent).
