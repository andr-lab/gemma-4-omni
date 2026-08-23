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

## Aggiornamento 17/08/2026 — ripresa dopo quota, CHECK-E smoke in corso

- **Quota GPU resettata** (lun 17/08 00:00 UTC). Ripresa operativa.
- **Pulizia repo**: storico archiviato in `_archive_20260817/` (fix_kernel_*, commit_ep2, kernel diag/eval, samples ep2, script one-off, report vecchi), log spazzatura v14 rimossi, `.gitignore` aggiornato. Commit `c67c143`.
- **NEXT_STEP.md (protocollo R0–R4) ricevuto dall'utente**: ipotesi co-adattamento Gemma↔adapter↔sub-talker; R1 adapter-only con latent alignment; loss L = L_acoustic + λ·L_latent + μ·L_text + ν·L_turn; turn head ausiliaria; modifiche obbligatorie (clipping 2048 non cieco, DuplexConv/MDC, preflight REAL_ENCODING per contenuto).
- **Validazione agy ×2 (DDD cross-model)**:
  - claude-opus-4-6-thinking: CHECK-E va completato PRIMA come ceiling test (~45min GPU, v4 pronto); adapter 1-layer lineare 3072→2048 strutturalmente sottodimensionato → MLP 2-layer come fallback se rank/varianza bassi; hidden_states[-1] discute con la letteratura (PRIME-Speech usa layer ~2/3 profondità); R1: target h_t dal main talker Qwen offline (opzione i), loss cosine + variance reg, gate cos_sim VAL>0.5 / CE_act<6.5 / rank>100 / test audio C.
  - gemini-3.7-flash-high: stesso verdetto (CHECK-E prima); progetta R1 in 5 step (estrazione offline h_t da Qwen/Qwen3-TTS-12Hz-1.7B-Base su ~10h DuplexConv → preflight dimensioni/range → training solo adapter con L_latent 500 step → gate loss+CE<5.2 → bivio R2/R3 vs CHECK-E se gap informativo); rischi: gap informativo (Gemma frozen senza testo risposta non può ricostruire pitch/fase dei codebook 11–15), illusione da alignment (cosine alta ma audio non intelligibile), off-by-one frame rate 12.5Hz + padding sinistro F.pad(2,0).
- **CHECK-E v4 rivisto (fix banali)**: aggiunto `import shutil` (era usato in `shutil.rmtree` senza import — stesso tipo del NameError v3). `frame_weights` confermato nel dataset (kaggle_pipeline.py:2676). py_compile OK, 0 char>127, round-trip notebook OK.
- **Smoke kernel `andrlab/gemma4-checke-smoke` v1 PUSHATO** (17/08 15:27 UTC): STEPS=6, EVAL_EVERY=2, N_TRAIN=3, N_VAL=2 (≈10-20 min GPU) per validare che il learning congiunto PARTE senza crash (lezione: mai run lunghe con errori banali). Monitor attivo. Al verdetto smoke → run completa CHECK-E (800 step) → verdetto ceiling test → decisione direzione.

## Esito SMOKE CHECK-E (17/08 15:40 UTC) — run completa in corso

- **Smoke `andrlab/gemma4-checke-smoke` v1 COMPLETE in 12m52s, ZERO errori.** Il fix v4 funziona: cache unsloth rimossa, causal_conv → Conv1d fresco fp32 (in=1536, out=1536, k=3), inference tensors 0, adapter fresco fp32 3.15M params, sub-talker fresco fp32, LoRA sub-talker 1.8M params, niente più "Inference tensors cannot be saved for backward".
- Baseline coerente: TRAIN ce_act=7.247 | VAL ce_act=6.995 (atteso ~6.81-7.26, finestre diverse dal run v1/v2).
- 6 step: loss 2.03→1.55, TRAIN ce_act 7.82→6.62, VAL stabile ~7.1 (nessun apprendimento VAL in 6 step — atteso).
- Ablation h_t=0: TRAIN 12.33 / VAL 13.13 → gap VAL ≈ 6.0 vs con-h_t → **il decoder usa davvero h_t** (coupling reale, non fittizio).
- Verdetto automatico "KO" NON significativo con 6 step (il criterio delta≥3 richiede apprendimento VAL); la run completa (800 step, ~45 min) darà il verdetto vero.
- **Run completa `andrlab/gemma4-checke-full` PUSHATA e in esecuzione** (17/08 ~15:45 UTC): cell2 = checke_cell2.py (md5 bcc89f51, default 800 step / 24+12 finestre / eval ogni 50). Monitor `433a1a5c` attivo (timeout 90 min).
- Nota log: `kaggle kernels logs` NON accetta `-f <file>` (è `--follow`); il log va scaricato con redirezione `> file` + `PYTHONIOENCODING=utf-8`.

## VERDETTO CHECK-E (run completa, 17/08 ~16:20 UTC) — KO → Piano B FitNets

- Run `andrlab/gemma4-checke-full` COMPLETE in 36m35s. 800 step, 24+12 finestre, sub-talker LoRA + Gemma QLoRA + adapter tutti trainable, h_t differenziabile.
- **Pattern IDENTICO a CHECK-D**: TRAIN collassa (ce_act 7.71→0.04, memorizzazione totale su 24 finestre), VAL PEGGIORA monotonamente (6.81→13.40, sopra casuale 7.62 e sopra baseline 6.81). ce_all VAL 3.71→7.07.
- **Ablation h_t=0**: TRAIN 12.57 (vs 0.04 → h_t è l'unico veicolo dell'apprendimento in train), VAL 13.65 (vs 13.40 → ≈ identico: in VAL h_t NON porta informazione utile).
- **Interpretazione (DDD, confronto con CHECK-B2/C/D)**:
  - B2 (50h, sub congelato): train 7, val 7 → niente segnale.
  - C (h_t fisso, sub congelato): overfit, val 10.8.
  - D (h_t fisso, sub LoRA): overfit totale, val 13.6.
  - E (learning congiunto, tutto trainable): overfit totale, val 13.4.
  - → Nessuna combinazione di (origine h_t) × (adattabilità decoder) generalizza. Il denominatore comune: l'hidden di Gemma (parlato utente + testo risposta) NON contiene l'informazione acustica/prosodica/lessicale necessaria ai codebook residui Qwen in distribuzione generalizzabile. **CONFUTATA l'ipotesi "mancanza di co-adattamento"**: con h_t libero di riorganizzarsi il risultato non cambia. Il problema è il GAP INFORMATIVO strutturale (come previsto da agy: "Gemma frozen/semantica grezza non può ricostruire pitch/fase dei codebook 11-15").
- **Prossimo: PIANO B FitNets (R1)**: estrazione offline h_t nativo dal teacher Qwen (main talker) su ~10h DuplexConv → training adapter-only con L_latent (cosine + variance regularizer, secondo le 2 revisioni agy) → gate cos_sim VAL>0.5 / CE_act<6.5 / rank adapter>100 → poi R2/R3 (sblocco progressivo LoRA) o bivio.
- Lezione monitor: `kaggle kernels logs <kernel> > file` con PYTHONIOENCODING=utf-8 (il flag -f è --follow).

## Valutazione agy (gemini-3.7-flash-high) su scelta FitNets (17/08 16:20 UTC)

- a) Verdetto gap informativo CONFERMATO (B2 vs D/E; decoder usa h_t come hash-key della memoria train; USM Gemma = semantica, no micro-struttura acustica).
- b) FitNets = mossa giusta (rientrare nel manifold del code_predictor pre-addestrato) ma con punti deboli: mismatch conditioning (teacher: testo GT BPE Qwen; runtime: testo Gemma), regressione fuori manifold; adapter 1-layer lineare non basta da solo.
- c) TARGET = **layer 28 (output finale del backbone post-RMSNorm)** — OBBLIGATORIO: il code_predictor (nostro sub-talker) è stato addestrato su QUEL formato. PRIME-Speech (layer ~2/3) non si applica (decoder addestrato da zero).
- d) Loss: **non solo cosine** (rischio sotto-spazio fittizio: cosine 0.7-0.8 allineando 2-3 componenti principali). L = α·SmoothL1(h_ada, h_tea) + β·(1−CosSim) + λ·L_CE(cb1-15) — supervisione mista con ancoraggio sul task.
- e) Step 0: SANITY 1-SAMPLE (costo ~0): h_t nativo + cb0 GT → sub-talker locale → Code2Wav → audio. Perfetto → problema era h_t. Metallico → bug interno sub-talker (rotary/mask/vocabolario), stop.
- f) Dimensionamento: estrazione 1.5-2h (~1000 finestre) = 15-20 min GPU (non 10h); R1 500-800 step adapter-only < 5 min; gate CosSim_VAL>0.55 + CE_VAL<6.2 + test audio C.
- g) Mitigazioni: adapter 2-layer (3072→4096→2048+LN) se rank effettivo <100; R2 LoRA r=8 lr=1e-5 sul sub-talker.
- h) Alternative: distillazione CE pura = fallita (CHECK-C/D/E); ibridazione PRIME-Speech (h_t = W_h·h_Gemma + W_e·Emb(text) + W_s·e_spk) = fallback se R1 non passa il gate.
- Operativo: Step 0 sanity → estrazione pilota 2h → training R1 → gate. Fattibilità alta su T4 (25 min totali GPU).

## Estrazione target FitNets — implementazione (17/08 17:00 UTC)

- **Studio SDD del teacher completato** (fonte: repo ufficiale QwenLM/Qwen3-TTS, finetuning/dataset.py + sft_12hz.py + modeling_qwen3_tts.py, scaricati e analizzati):
  - h_t nativo = `hidden_states[codec_mask[:, :-1]]` = output layer 28 post-RMSNorm (talker_hidden 2048, 28 layer, vocab cb0 3072, residui 2048) ai frame audio 12.5Hz — ESATTAMENTE l'input del code_predictor (nostro sub-talker).
  - Formato input dual-track (collate_fn ufficiale, batch 1): t = 8 + T_text + T_audio; canale testo (ChatML `<|im_start|>assistant\n...`, primi 3 token in testa, pad 3-6, tts_bos in 7, testo in 8.., tts_eos, pad) + canale codec (nothink/think_bos/think_eos/0/pad in 3-8, pad nel testo, codec_bos in 8+Tt-2, cb0 GT, codec_eos); codec_embedding_mask[6]=False (slot speaker, voce neutra); codec_ids (1,t,16) per i residui.
  - Forward (sft_12hz.py): inputs_embeds = text_emb(text) * text_mask + codec_emb(cb0) * codec_emb_mask + Σ_{i=1..15} codec_emb_i(cb_i) * codec_mask (speaker embedding omesso → voce neutra, coerente col nostro sistema senza voce target); talker(inputs_embeds[:, :-1], attn[:, :-1], output_hidden_states=True, use_cache=False).
  - Token speciali: tts_pad 151671, tts_bos 151672, tts_eos 151673; codec_nothink 2155, think_bos 2156, think_eos 2157, pad 2148, bos 2149, eos 2150. text_vocab 151936 (Qwen2Tokenizer).
- **Script `kernel_src/fitnets_extract_cell2.py`**: clona il repo Qwen3-TTS (qwen_tts) in Kaggle (git clone --depth 1, internet), carica teacher bf16 eager forward-only (3.5GB VRAM su T4), replica collate+forward, estrae h_t fp16 per finestra (`ht_{idx:05d}.npy`) + manifest.json in `/kaggle/output/fitnets_targets/`. SANITY (1 finestra): CE sub-talker con h_t nativo + cb0 GT + sample_acoustic_codes greedy → wav + accuratezza cb1-15 vs oracle. Scan finestre attive via mel (proxy energia, niente decode per finestra).
- **Asimmetria documentata**: h_t teacher[p] vede cb0[p] in input (futuro-informato), il nostro h_t[p] no; mitigazione = loss mista (L_latent + λ·L_CE) nel training R1.
- **Kernel**: `andrlab/gemma4-fitnets-smoke` (FITEX_N=1, SANITY=1, ~5-10 min: validazione percorso completo) → poi `andrlab/gemma4-fitnets-extract` (FITEX_N=1000, ~20-25 min). Output scaricabili via `kaggle kernels output` → dataset `andrlab/gemma4-fitnets-targets` per R1.
