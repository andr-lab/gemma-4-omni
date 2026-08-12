# Verifica architetturale integrata — Gemma 4 E2B → adapter → sub-talker Qwen3-TTS

Data: 2026-02 (post v14-kill, post CHECK-B/B2/B3/C, **CHECK-D COMPLETATO: KO — diagnosi CHIUSA**)
Fonti: Technical Report ufficiali (Gemma 4, Qwen3-TTS) + 8 paper primari + risultati empirici CHECK-B/B2/B3/C/D.

---

## 1. Cosa dicono i technical report ufficiali (fonte primaria)

### 1.1 Gemma 4 Technical Report — "E2B e E4B usano un audio encoder 305M USM"

> "E2B and E4B Gemma models use a 305M audio encoder that processes audio in 40ms chunks
> with Mel filterbank inputs. The encoder architecture is based on the Universal Speech
> Model (USM)... twelve Conformer layers... **We do not use vector quantization; the LLM
> ingests the continuous representations produced by the audio encoder.**"

Punto architetturale A: **Gemma 4 E2B ingesta audio come rappresentazioni CONTINUE (non tokenizzate, non VQ)**. Le hidden di Gemma codificano il contenuto *semantico/paralinguistico* che serve a capire il parlato — ma NON sono mai state addestrate per essere il condizionamento di un decoder audio.

Punto architetturale B (report §2.2, §2.3): Gemma 4 12B è encoder-free (raw audio proiettato), ma **E2B (quello che usiamo noi) ha l'encoder USM 305M** — audio → 40ms chunks → mel → conformer → continua → LLM. L'h_t di Gemma è l'ultimo hidden del LLM, in uno spazio che la letteratura di PRIME-Speech (CKA su layer intermedi) mostra NON essere il più adatto per condizionare un talker.

### 1.2 Qwen3-TTS Technical Report — "backbone predice cb0, MTP genera i residui"

> "Qwen3-TTS-12Hz... adopts a hierarchical prediction scheme: **the backbone ingests
> aggregated codebook features to predict the zeroth codebook, and an MTP (Multi-Token
> Prediction) module then generates all residual codebooks.**"

> "For real-time synthesis, Qwen3-TTS employs a **dual-track representation by concatenating
> textual and acoustic tokens along the channel axis.**"

Punto architetturale C: **Nel teacher Qwen3-TTS, l'input al backbone è il testo + i codebook acustici aggregati** (dual-track). Il backbone (1.7B, hidden 2048) produce cb0 e le hidden da cui l'MTP (sub-talker, 5 layer, hidden 1024) genera cb1-15. L'MTP è stato addestrato per ricevere in input: **cb0 embedding + h_t del backbone + i codebook precedenti** — con h_t = hidden di un backbone che HA VISTO IL TESTO.

Punto architetturale D: **Il nostro sub-talker (estratto: code_predictor + cb0_embedding) è identico all'MTP del teacher**. Il suo contratto di input è `[h_t(2048) + cb0_embed + cb1..cb14_embeds]` (verificato in qwen_subtalker.py righe 205-260, replica esatta del sorgente qwen-tts 0.1.1). Il contratto di addestramento dell'MTP era: h_t = hidden del backbone Qwen3-TTS **condizionato su testo + audio precedente**.

### 1.3 Il disallineamento (la radice del problema)

| | Teacher (Qwen3-TTS originale) | Nostro design |
|---|---|---|
| Generatore di h_t | Backbone Qwen3-TTS 1.7B, addestrato su 5M ore con testo+audio | Adapter Linear(1536→2048)+LayerNorm su Gemma 4 E2B (input solo audio) |
| Cosa vede il generatore di h_t | Testo GT + codebook aggregati (dual-track) | Solo mel/audio continuo (nessun testo esplicito nel coupling) |
| Addestramento del generatore di h_t | Joint con MTP (fine-tuned insieme) | Congelato (Gemma) / solo adapter trainato |
| Capacità del generatore | 1.7B params, 30+ layer | 6.3M params (adapter), 1 layer |
| Manifold atteso dall'MTP | h_t del backbone Qwen3-TTS (2048) | h_t proiettato da Gemma (deve entrare nello stesso manifold) |

**La verifica**: l'MTP/sub-talker Qwen è una funzione rigida addestrata su 5M ore per mappare `h_t_backbone_Qwen → cb1-15`. Il nostro adapter deve produrre h_t che cada in quel manifold partendo da hidden di Gemma che (a) non hanno mai visto testo nel coupling, (b) sono continue (no codebook), (c) vengono da un backbone mai addestrato per la generazione audio.

---

## 2. Evidenze empiriche (CHECK-B, CHECK-B2)

1. **CHECK-B (maschera codici, 15m)**: speech_frac = 1.0 con `(tgt==0).all(-1)` → il silenzio Qwen3-TTS NON è 16-zeri → la maschera codici era inutile → il monkeypatch v14 era no-op.
2. **CHECK-B2 (maschera energia, come v13)**: CE parlato ≈ **7.0-7.4 in TRAIN e VAL** (casuale = 7.62 su vocab 2048) → **H1 confutata**: il modello non ha MAI imparato il parlato, nemmeno in train. Il silenzio: train CE 3.5-6.2, val 0.45-0.63 → impara i token di silenzio ma nessun codice di parlato.
3. **La loss 0.49 era un'illusione**: dominata dal silenzio (CE val ~0.5 su ~82-92% dei frame), il parlato contribuiva ~zero.

Interpretazione: il gradiente non era "annegato" (H1) — il modello **non riesce a mappare h_t → codici parlato nemmeno overfittando** (o il training non glielo permette). CHECK-B3 (probe di capacità, Linear vs MLP su una finestra) discrimina: capacità insufficiente (H3) vs informazione assente in h_t (H2).

---

## 3. Verifica architetturale vs letteratura (8 paper)

### Il pattern vincente nella letteratura (TUTTI i sistemi reali)

| Sistema | Backbone | Decoder audio | Cosa si addestra |
|---|---|---|---|
| **PRIME-Speech** (Microsoft) | Frozen | Post-decoder 2B, 10 layer causali | **Il decoder** (sincronizzato su hidden intermedi ~2/3 profondità) |
| **Freeze-Omni** (Tencent) | Frozen LLM | NAR+AR decoder 120M + prefix decoder | **I decoder** (3 stage) |
| **CosyVoice 2** (Alibaba) | LLM | Flow-matching | LLM predice solo cb0; flow matching genera i residui |
| **PersonaPlex** (NVIDIA, Moshi-based) | Intero modello (Moshi init) | — | **Tutto**, con downweight 0.02 sui token non-semantici |
| **VALL-E 2** | Intero | — | Grouped code modeling |
| **Qwen3-TTS (teacher)** | 1.7B | MTP (sub-talker) | Backbone+MTP **joint**, 5M ore, con testo |

**Nessun sistema usa: decoder congelato + adapter piccolo come ponte verso un backbone mai addestrato per quello.** Il nostro design è l'inverso del pattern PRIME-Speech/Freeze-Omni: lì il decoder si adatta al backbone (trainabile); qui il decoder è rigido e il ponte è un singolo layer lineare.

### Dettagli che colpiscono la nostra architettura

1. **PersonaPlex downweighta i token audio NON-semantici a 0.02** — il contenuto sta in cb0 (semantico); i residui (cb1-15) sono dettaglio acustico. Noi predichiamo 15 codebook con coupling weight 0.6: gerarchia di importanza invertita.
2. **PRIME-Speech usa hidden da un layer INTERMEDIO (~2/3 profondità, scelto con CKA)** — non l'ultimo. Noi usiamo hidden_states[-1]. Le ultime hidden di un LLM sono "testuali/astratte", quelle intermedie più vicine all'audio.
3. **PRIME-Speech condiziona con h_mix = wh·h_mid + wτ·e_text(prev) + wa·r_audio(prev)** — include testo E audio. Nel nostro coupling il testo non entra mai nel sub-talker (solo h_t da audio).
4. **Freeze-Omni addestra il decoder in 3 stage con embedding congelati del LLM** — il ponte è addestrato CON il decoder, non contro un decoder congelato.
5. **FitNets (Hint Learning)**: `L_HT = ½||u_h(x) − r(v_g(x))||²` — il piano B: regressione diretta verso h_t del teacher, più semplice che forzare attraverso CE indiretta.

---

## 4. Verdetto finale (post CHECK-B3 + CHECK-C + CHECK-D) — diagnosi CHIUSA

**CHECK-B3 (probe, 1 finestra, 24m):** l'adapter PUO' overfittare una finestra a CE act ~0.01 (Adam lr 1e-3, h_t fisso) → capacità e informazione per-finestra presenti.

**CHECK-C (generalizzazione, 12 win train + 12 val, 16m):** condizioni ideali (h_t fisso, lr 1e-3, CE pura, adapter-only) NON generalizzano:

| | TRAIN ce_act | VAL ce_act |
|---|---|---|
| Baseline (adapter ep2) | 7.310 | 6.807 |
| @400 step | 3.376 | **10.819** (peggio del casuale 7.62) |

**CHECK-D (decoder adattivo, knob #2 ADR-005: LoRA sul sub-talker r=16 + adapter, 16m):** anche dando al decoder 1.8M params di adattamento, la firma e' IDENTICA:

| | ce_act TRAIN | ce_act VAL | ce_all VAL |
|---|---|---|---|
| Baseline (adapter ep2, LoRA zero) | 7.310 | 6.807 | 3.711 |
| @400 step (adapter 1e-3 + LoRA sub-talker 1e-4) | 0.110 | **13.630** | 7.270 |
| **Ablation h_t=0** | 12.497 | 13.247 | 9.206 |

**Lettura CHECK-D**: (1) train 0.11 → il decoder MEMORIZZA le 12 finestre (usa h_t intensamente: senza h_t il train crolla a 12.5); (2) val 13.63 → peggiora oltre il caso, in salita fino all'ultimo step; (3) ablation h_t=0 in VAL quasi non cambia (13.6 vs 13.2) → la mappatura appresa e' una regola per-finestra che spara classi sbagliate sugli h_t nuovi. NON e' coupling fittizio (in train h_t conta moltissimo): e' INFORMAZIONE NON GENERALIZZABILE.

**VERDETTO FINALE (4 test brevi, ~75 min GPU totali):** il problema NON e' l'ottimizzazione (H1 confutata: condizioni ideali + decoder adattivo → stesso overfit, val oltre il caso). E' un problema di INFORMAZIONE: h_t (ultimo hidden di Gemma, semantico) NON contiene il residuo acustico RVQ in forma generalizzabile; le associazioni h_t→codici sono per-finestra incompatibili. Il contratto "adapter → sub-talker congelato" e' **non-generalizzabile per costruzione** — coerente con la letteratura (nessun sistema reale usa decoder congelato + adapter piccolo; tutti addestrano il decoder con dati massivi).

### Conseguenze operative

Nessun tweak di ottimizzazione (lr, loss, LoRA sul decoder, piu' step) puo' risolvere: il collo di bottiglia e' l'informazione in h_t. Le strade residue (da decidere CON l'utente, nessuna run lunga senza teoria approvata):

1. **Piano B FitNets (piu' fondato)**: distillazione h_t verso il teacher Qwen main talker; l'adapter impara a riprodurre h_t nativo del teacher, il sub-talker accetta input nel suo manifold. Costo: download main talker (~3-4GB, in Kaggle) + estrazione offline target h_t + training di regressione.
2. **Generatore audio ridisegnato**: decoder RVQ addestrato da zero condizionato da Gemma (pattern PRIME-Speech/CosyVoice 2) — costo GPU alto, dati 1000+ ore.
3. **Ridimensionamento del progetto**: Gemma = planner semantico (testo + cb0), generatore residui separato addestrato.

**Regola**: nessuna run lunga finche' l'utente non decide la direzione (tutte richiedono settimane di lavoro e GPU; nessuna e' un semplice "fix").

---

## Aggiornamento CHECK-E (learning congiunto, proposta utente) — 10/08/2026

**Contesto**: l'utente ha proposto l'architettura "Gemma-4 + QLoRA, poi adapter, poi QLoRA sul subtalker" addestrati insieme (learning congiunto, pattern Moshi/PersonaPlex/Qwen3-TTS). Valutazione teorica: **valida** — attacca la causa (informazione in h_t) rimuovendo il vincolo del decoder congelato. Validazione con test breve CHECK-E.

**Esecuzione CHECK-E** (kernel `andrlab/gemma4-fix-checke`):
- v1 ERROR: `Inference tensors cannot be saved for backward` nel primo step di training — pesi del `causal_conv` (compilato da unsloth) tracciati in inference_mode dalla baseline eseguita prima del training.
- v2 ERROR: fix insufficiente (`is_inference` = 0 → il problema era il grafo compilato, non i pesi).
- v3 ERROR: `NameError: name 'device' is not defined` — mio bug banale (device usato prima della definizione nel blocco di sostituzione del conv).
- **v4 pronto** (locale, `kernel_src/checke_cell2.py` md5 `dc15a2b4`): Conv1d fresco fp32 al posto del conv compilato + rimozione cache `unsloth_compiled_cache` + `.clone()` su agent_hidden.
- **Blocco esterno**: quota GPU settimanale Kaggle esaurita (30h, reset lun 17/08 00:00 UTC). Push v4 impossibile fino al reset.

**Stato**: CHECK-E IN SOSPESO (fix pronto, ~45 min GPU al reset). Baseline già validata nei run v1/v2: TRAIN ce_act=7.263, VAL ce_act=6.808 (coerente con CHECK-D baseline 6.807).
