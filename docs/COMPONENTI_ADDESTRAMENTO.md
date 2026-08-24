# Matrice di Addestramento — Componenti, Metodo, Dati (definitivo)

**Data**: 2026-08-24 · **Base**: docs/DESIGN_FULLDUPLEX_LETTERATURA.md, docs/DATASET_RUOLI_EVIDENCE.md
**Principio guida**: ogni componente si allena SOLO con i dati che possono insegnargli il
suo ruolo; la voce italiana NON si addestra (J-Moshi: Mimi congelato è lingua-agnostico).

---

## 1. LA MATRICE

| # | Componente | Stato | COSA impara | COME (ricetta) | CON COSA (dati) | Evidenza |
|---|---|---|---|---|---|---|
| 1 | **Gemma-4-E2B** | 🔒 CONGELATA | — (preserva ragionamento, tool calling, testo) | nessun gradiente; usata come estrattore hidden + generatore testo | — | vincolo utente; Qwen2.5-Omni Thinker congelato |
| 2 | **Adapter acustico** (TemporalResidualAdapter) | ⚙️ GIÀ ADDESTRATO (CV26 40.8h) — **ritrain opzionale P4** | mappare hidden Gemma → spazio talker Qwen (h_sub) | cos_loss + MSE + coupling (già fatto); **P4: + loss ausiliaria predizione cb0** (SemBridge: supervisione semantica sugli stati, training-only) | **MDC26/CV26** (italiano mono: lingua/fonetica) | 2608.07462 SemBridge; nostri probe (h_sub 3.7% info cb0 = collo) |
| 3 | **Predittore cb0** (AR-SemanticDecoder) | 🔴 IN TRAINING v3 (100k step) | sequenza cb0 (codebook semantico 12.5Hz) dal contenuto di Gemma | **P1 input corruption** (ε=0.15) + **P2 hybrid TF/free-running** (k=8, w=0.5, prompt protection, EOS-adaptive, no gate) + label smoothing + log-prior bias; **P3: + ancora testuale h_text** | cache `cache_ars_full.pt` (h_sub + cb0 da MDC26); **P3: + h_text** (hidden Gemma sul blocco testo, già nel prompt) | 2607.29363 (corruption), 2509.17021 (hybrid), 2410.00037 Moshi (inner monologue), 2508.15827 (thinking-in-speaking) |
| 4 | **Token di controllo turn-taking** (`<backchannel>`, `<yield>`, `<barge_in>`, `<interrupt>`, `<speak>`) | 🆕 DA IMPLEMENTARE (sostituisce TurnHead) | quando parlare/cedere/interrompere (decision state machine) | special tokens nel vocabolario Gemma, CE loss **pesata 10× eventi rari**, predetti nel flusso testo→cb0 | **DuplexConv** (VAD frame-level + turn state `<|complete|>` = GT nativo, NON euristiche) + **stereo IT sintetico** (dialoghi testuali IT → Qwen3-TTS multi-stream → WER-selection, pattern J-Moshi) | 2603.09180 DuplexCascade, 2410.00037 Moshi multi-stream, 2606.19453 survey (state machine nel modello), prova negativa TurnHead v5 (bACC 41%) |
| 5 | **Sub-Talker Qwen3-TTS** (MTP cb1..15) | 🔒 CONGELATO | — (voce italiana già acquisita) | nessun gradiente | — | 2506.02979 J-Moshi (Mimi congelato parla giapponese senza training); Qwen3-TTS report |
| 6 | **Code2Wav** (decoder ufficiale) | 🔒 CONGELATO | — | nessun gradiente | — | Qwen3-TTS report |

---

## 2. FLUSSO DI CONDIZIONAMENTO (chi legge cosa)

```
Utente audio ──► Gemma (congelata) ──► hidden h_raw (1536)
                                        │  ▲
              [TESTO risposta agente]───┘  │ (già nel prompt, text-instructed)
                                        │  │
              [token di turno] <───┬────┘  │ (P5: nel flusso, condizionano tutto)
                                   │       │
                                   ▼       │
                          Adapter acustico │ (P4: + cb0-pressure)
                                   │       │
                                   ▼       │
                        h_sub (2048) ──────┼──► Predittore cb0 (v3: P1+P2+P3)
                                   │       │        │ ancora testuale h_text
                                   ▼       ▼        ▼
                        Sub-Talker ◄── cb0 ──┘  (sequenza completa)
                                   ▼
                              Code2Wav → audio
```

**Punti chiave del flusso:**
- Il **testo della risposta agente è già nel prompt di Gemma** (TEXT_BLOCK_LEN=48, left-pad)
  → l'ancora testuale (P3) è un pooling degli hidden sul blocco testo, non un nuovo dato.
- Il **predittore cb0 è l'UNICO generatore semantico addestrabile**: è lì che si gioca
  Task 12 (free ≈ ancorato). Tutti i fix P1-P3 agiscono su di lui.
- I **token di turno** (P5) vivono nel flusso di Gemma: condizionano sia il testo sia
  (di conseguenza) cb0 — niente componente esterno.

---

## 3. DATI: chi usa cosa (riassunto del verdetto)

| Dato | Lingua | Formato | Usato da | NON usato da |
|---|---|---|---|---|
| **MDC26 / CV26** (40.8h IT) | IT | mono, audio+trascrizioni | Adapter (fatto), Predittore cb0 (cache), **ancora testuale P3** (trascrizioni) | Sub-Talker (voce già IT), turn-taking (non conversazionale) |
| **DuplexConv** (2.000h) | ZH | stereo, VAD frame-level, turn state, interruzioni | **Token di turno P5** (etichette GT), validazione policy | Predittore cb0 (fonetica cinese ≠ IT), adapter (lingua sbagliata) |
| **Stereo IT sintetico** (da creare) | IT | stereo sintetico (dialoghi testuali IT → Qwen3-TTS multi-stream → WER-selection) | **Token di turno P5** (statistiche turno IT), futuro full-duplex | — |
| **Qwen3-TTS** (voce) | multilingue | modello | Sub-Talker (congelato) | — |

---

## 4. ORDINE DI ESECUZIONE (stato attuale)

| Passo | Azione | Stato |
|---|---|---|
| 1 | **v3 training** (P1 input corruption + P2 hybrid) 100k step | 🔴 IN CORSO (~step 15.5k/100k, val_tf 7.1%) |
| 2 | **Free-gen v3** (finestre val + blind ASR + mel cos) | ⏳ dopo training |
| 3 | **Cache v2** (h_raw + h_text) + probe h_raw vs h_sub | ⏳ script pronto (`extract_ars_cache_v2.py`) |
| 4 | **P3 ancora testuale** (v4 trainer: condiziona su h_text) | ⏳ se probe h_raw mostra info semantica |
| 5 | **P4 cb0-pressure adapter** (SemBridge) | ⏳ se h_sub resta il collo |
| 6 | **P5 token di turno** (DuplexConv etichette + stereo IT sintetico) | ⏳ dopo catena vocale stabile |
| 7 | **Opzione B** (backbone 1.7B + proiezione) | ⏳ solo se top-1 < 15% dopo 1-5 |

---

## 5. REGOLE VINCOLANTI (dalla letteratura + lezioni interne)

1. **Mai ri-addestrare la voce** (sub-talker, Code2Wav): J-Moshi congela Mimi → la nostra
   voce italiana è Qwen3-TTS, già pronta.
2. **Il turn-taking non è un classificatore esterno**: è predizione di token nel LM
   (DuplexCascade), con loss pesata per gli eventi rari.
3. **La semantica si pesa 100:1 rispetto all'acustica** (Moshi loss ratio) → il predittore
   cb0 è il componente critico; i residui (sub-talker) sono già ottimi e congelati.
4. **Il gap TF/free si chiude nel TRAINING** (hybrid + corruption), non con euristiche
   d'inferenza (repetition-penalty = palliativo).
5. **Ancora testuale = canale diretto Gemma→cb0** (inner monologue): il contenuto viene
   da Gemma (testo), cb0 lo realizza foneticamente.
6. **Metriche**: mai top-1 isolato (tetto intrinseco, 2509.17021); sempre qualità del
   parlato campionato (blind ASR + mel cos-sim) + ascolto utente.
