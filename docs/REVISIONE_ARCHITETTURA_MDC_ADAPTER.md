# Revisione Architetturale: "Solo MDC per l'Adapter" e Separazione delle Responsabilita
## Valutazione Tecnica, Analisi Mismatch Input-Target e Verdetto Operativo

- **Progetto**: Gemma 4 Full-Duplex Omni Pipeline (C:\Users\andre\Documents\gemma-4-omni)
- **Ruolo**: Ingegnere ML Senior / Lead Architect
- **Data**: 21 Agosto 2026
- **Oggetto**: Revisione architetturale per l'adozione esclusiva di Mozilla Common Voice (MDC) 26.0 Italiano nello Stage 2 (Adapter Latente) e disaccoppiamento da DuplexConv (TurnHead).

---

## 1. Executive Summary & Verdetto Architetturale

```
========================================================================================
                              VERDETTO ARCHITETTURALE
========================================================================================
[STATO]:        VALIDA E FORTEMENTE CONSIGLIATA (Opzione A).

[PRINCIPIO]:    SEPARAZIONE RIGOROSA DELLE RESPONSABILITA:
                1. GENERAZIONE VOCALE & TIMBRICA ITALIANA (Stage 2) -> 100% MDC Italiano.
                2. TURN-TAKING & DINAMICHE FULL-DUPLEX (Stage 3)     -> DuplexConv + MDC.

[MOTIVAZIONE]:  L'adapter FitNets e un regressore neurale che allinea lo spazio semantico
                di Gemma 4 allo spazio latente acustico h_t (2048D) del sub-talker.
                Poiche Qwen3-TTS supporta nativamente l'italiano al SOTA mondiale
                (WER 0.948, SIM 0.817), alimentare h_t con il 100% di parlato italiano
                elimina alla radice il bias mandarino senza rompere il coupling acustico.
========================================================================================
```

---

## 2. Analisi dell'Ipotesi "Solo MDC per l'Adapter" (Fase 2.1)

### 2.1. Trade-Off Analitico: Cosa Perde vs Cosa Guadagna il Modello

| Dimensione | Dataset Misto (DuplexConv 93.5% + MDC 6.5%) | Solo MDC Italiano (100% MDC 50-80h) |
| :--- | :--- | :--- |
| **Purezza Fonetica e Timbrica** | **GRAVE COMPROMISSIONE**: l'adapter modella fonemi cinesi, toni mandarini e formanti non italiane. | **100% ITALIANO NATURALE**: fonotassi italiana pura, vocali aperte/chiuse, consonanti doppie e prosodia corretta. |
| **Bias Linguistico / Contaminazione** | **93.5% Bias Mandarino**: campioni sintetizzati incomprensibili per un ascoltatore italiano. | **0.0% Bias Cinese**: eliminazione completa di qualsiasi residuo fonetico non italiano. |
| **Contesto Conversazionale Reale** | Presente (dialoghi spontanei cinesi con sovrapposizioni e interruzioni naturali). | Assente (MDC e read speech di frasi singole lette da volontari). |
| **Accoppiamento (Coupling) con Sub-Talker** | Funzionante ma calibrato sui centroidi acustici del Mandarino. | **PERFETTO**: Qwen3-TTS e preaddestrato su >5M ore multilingue e mappa h_t italiano sui codici RVQ con WER SOTA (0.948). |
| **Saturazione Dati per l'Adapter** | 50 ore di parlato. | **50 - 80 ore di parlato italiano validato** (40,000 - 65,000 clip), ampiamente sufficienti per saturare i 59.55M parametri dell'adapter. |

### 2.2. Il Canale Target Codes RVQ puo essere 100% MDC?
**SI', senza alcuna riserva.**
1. **Ruolo Matematico dell'Adapter**:
   L'adapter $f_\theta: \mathbb{R}^{1536} \to \mathbb{R}^{2048}$ ha l'obiettivo di proiettare gli stati nascosti $h_{\text{gemma}}$ dell'LLM (congelato) verso lo spazio latente $h_t$ che condiziona il sub-talker neurale:
   $$\mathcal{L}_{\text{Stage2}} = (1 - \text{cos\_sim}(f_\theta(h_{\text{gemma}}), h_t^{\text{target}})) + 0.5 \cdot \text{MSE}(f_\theta(h_{\text{gemma}}), h_t^{\text{target}}) + 0.2 \cdot \mathcal{L}_{\text{coupling}}(y_{1..15}, \text{SubTalker}(f_\theta(h_{\text{gemma}}), y_0))$$
2. **Natura Agnostica del Sub-Talker**:
   Il sub-talker `Qwen3TTSSubTalker` e' una rete neurale Transformer a 5 layer che esegue la decodifica condizionata:
   $$P(y_{1..15} \mid h_t, y_0) = \prod_{k=1}^{15} \text{Softmax}(\text{MTP}_k(h_t, y_0, y_{<k}))$$
   Il teacher `Qwen3-TTS-12Hz-1.7B-Base` possiede gia la conoscenza acustica e fonetica dell'italiano. Fornire come target $h_t^{\text{target}}$ estratto da frasi italiane di MDC guida l'adapter a generare le esatte traiettorie continue che attivano i fonemi italiani nel sub-talker.

---

## 3. Analisi del Mismatch Linguistico Utente-Agente (Fase 2.2)

### 3.1. Il Problema del Mismatch nei Run Precedenti
Nei run precedenti (`Temporal`, `Temporal 2`, `RankFix`), il pipeline presentava una discrepanza strutturale:
- **Input Utente (Mel 100Hz)**: 93.5% parlato mandarino di DuplexConv.
- **Target Agente (Codici RVQ 12.5Hz)**: 93.5% parlato mandarino di DuplexConv.
- Quando si cercava di valutare il modello su un prompt italiano, Gemma 4 riceveva un embedding utente non allineato e l'adapter proiettava verso $h_t$ cinese, generando audio incomprensibile.

### 3.2. Soluzione: Dialogo Dual-Channel Simulato con MDC
Costruendo il dataset di training con `create_dual_channel_dialogue` applicato a clip MDC Italiani:
1. **Canale 0 (Utente)**: Clip vocale italiano letto da uno speaker MDC (es. Speaker A: *"Quali sono le caratteristiche del lago?"*).
2. **Canale 1 (Agente)**: Clip vocale italiano letto da un altro speaker MDC (es. Speaker B: *"Si tratta del lago naturale piu esteso dello stato."*).
3. **Risultato sul Flusso di Informazione**:
   - L'audio encoder Conformer di Gemma 4 riceve mel log-spettrale a 100Hz di parlato **ITALIANO**.
   - I token BPE di testo di Gemma 4 sono in **ITALIANO**.
   - L'hidden state $h_{\text{gemma}}$ esprime semantica e prosodia **ITALIANA**.
   - L'adapter mappa $h_{\text{gemma}} \to h_t$ su target **ITALIANO**.
   - **Zero mismatch di lingua**: perfetta coerenza end-to-end su tutta la catena.

---

## 4. Architettura Target e Piano di Disaccoppiamento Funzionale

```
??????????????????????????????????????????????????????????????????????????????????????????
?                   ARCHITETTURA DISACCOPPIATA A DUE STADI (TASK 6B)                    ?
??????????????????????????????????????????????????????????????????????????????????????????
?                                                                                        ?
?  [FASE 1: PREP DATASET MDC ITALIANO (50-80 ORE)]                                       ?
?   ??? Input: Clip Validati Mozilla Common Voice 26.0 / 17.0 Italiano                   ?
?   ??? Simulazione Dual-Channel: Canale 0 Utente (Mel 100Hz) + Canale 1 Agente          ?
?   ??? Tokenizzazione RVQ: 16 Livelli a 12.5Hz con Tokenizer Qwen3-TTS                  ?
?   ??? Output: train_tokens_mdc_50h_mel.parquet, _codes.parquet, _text.parquet         ?
?                                                                                        ?
?  [STAGE 2: RETRAIN TEMPORAL RESIDUAL ADAPTER SU VOCE ITALIANA]                         ?
?   ??? Input: Gemma 4 E2B Frozen 4-bit (Mel Utente IT + Prompt IT)                      ?
?   ??? Modulo Trainable: TemporalResidualAdapter (Conv1D + 4 Blocchi MHSA Causali)      ?
?   ??? Target: 100% h_t estratto dal Teacher Qwen3-TTS su parlato MDC Italiano          ?
?   ??? Risultato: Modello vocale con pronuncia e timbro 100% ITALIANO                   ?
?                                                                                        ?
?  [STAGE 3: TURNSTATEHEAD (GESTIONE DINAMICHE DI TURNO)]                                ?
?   ??? Input: z_t = [h_t_temp (2048) || e_user (128) || d_e_user (128)]                ?
?   ??? Dataset: DuplexConv (2000h dinamiche reali) + MDC (dialoghi simulati)            ?
?   ??? Risultato: Gestione Barge-in, Yield e Backchannel universale e indipendente      ?
?                                                                                        ?
??????????????????????????????????????????????????????????????????????????????????????????
```

---

## 5. Specifiche dei Componenti per lo Stage 2 Italiano

1. **Backbone LLM**: `unsloth/gemma-4-E2B-it` (congelato in 4-bit, zero parametri trainable, forward `no_grad()`).
2. **Adapter Architettura**: `TemporalResidualAdapter` con:
   - `ResidualAdapter` (21.24M params): `CausalConv1d` ($k=3$, left-pad 2,0) + Skip `Linear(1536, 2048)` + MLP `1536 -> 3072 -> SiLU -> 2048` + `LayerNorm(2048)`.
   - `CausalTemporalModule` v2 (38.30M params): 4 blocchi Causal Multi-Head Self-Attention Pre-LN ($d=1024$, 8 teste, MLP 2048 con SiLU) + Positional Embedding ($512 \times 1024$) + `out_proj(1024 -> 2048)` con ReZero / Identity-Init.
   - Parametri trainable totali: **$59.55\text{M}$ parametri**.
3. **Funzione di Loss**:
   $$\mathcal{L} = (1 - \text{cos\_sim}(h_t^{\text{pred}}, h_t^{\text{target}})) + 0.5 \cdot \text{MSE}(h_t^{\text{pred}}, h_t^{\text{target}}) + 0.2 \cdot \mathcal{L}_{\text{coupling}}$$
4. **Ottimizzatore e Schedule**:
   - Optimizer: AdamW con weight decay 0.01 e gradient clipping 1.0.
   - Learning rate: $1.5 \times 10^{-4}$ con linear warmup (100 step su 1200) e cosine decay a 10% del picco.
   - Batch size: 8 con grad accum 2 (batch effettivo 16 finestre = 1024 frame = 81.92s di parlato per step).

---
*Documento di revisione architetturale redatto in conformita con i principi di Source-Driven Development (SDD) e Doubt-Driven Development (DDD).*
