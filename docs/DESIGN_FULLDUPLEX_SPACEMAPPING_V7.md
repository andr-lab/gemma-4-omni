# Architettura Full-Duplex: Mapping degli Spazi Latenti Gemma 4 ↔ Qwen3-TTS (Post-V7)

## 1. Visione Architetturale e Principio Fondante

Il principio guida dell'architettura è il paradigma **Decoupled Thinker-Talker con Spazi Latenti Allineati via Projector Specializzato**, formalizzato e validato dalla letteratura scientifica recente:

* **"Projector Is All You Train" (Iskandar et al., Agosto 2026, [arXiv:2608.19726](https://arxiv.org/abs/2608.19726))**: Addestrare unicamente il modulo di proiezione preserva al 100% l'intelligenza nativa, il ragionamento e il tool-calling del backbone LLM (Gemma 4), prevenendo il degrado da *capability drift* o *catastrophic forgetting* tipico del fine-tuning congiunto.
* **"ASPIRin — Action Space Projection" (Aprile 2026, [arXiv:2604.10065](https://arxiv.org/abs/2604.10065))**: Disaccoppia rigorosamente *quando parlare* (Action / Gating Space) da *cosa dire* (Speech Generation Space), eliminando i collassi generativi da regressione alla media.
* **"Freeze-Omni" (Novembre 2024, [arXiv:2411.00774](https://arxiv.org/abs/2411.00774))**: Dimostra la convergenza ottimale di dialoghi speech-to-speech a bassissima latenza con LLM e neural vocoder congelati.

---

## 2. Lo Spazio Latente Multimodale di Gemma 4 vs Spazio Acustico di Qwen3-TTS

La pipeline connette due spazi continui eterogenei:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             GEMMA 4 E2B (FROZEN)                            │
│                                                                             │
│  1. Spazio Acustico Conformer (USM 305M):                                   │
│     - h_raw[t] in R^1536 a 12.5 Hz (80ms/frame)                             │
│     - Riflette la prosodia, energia, pause ed espressione dell'audio utente │
│                                                                             │
│  2. Spazio Semantico Decoder LLM:                                           │
│     - h_tx in R^1536 (Hidden state del pensiero/risposta)                   │
│     - Riflette il contenuto lessicale, logico e la risposta generata        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ (Mapping & Gating a 12.5 Hz)
┌─────────────────────────────────────────────────────────────────────────────┐
│                      QWEN3-TTS 1.7B TALKER (FROZEN)                         │
│                                                                             │
│  Spazio di Condizionamento Acustico Recurrente:                             │
│  - h_cond[t] in R^2048 a 12.5 Hz                                            │
│  - Iniettato nel loop: inputs_t = agg(codes_{t-1}) + h_cond[t]              │
│  - Pilota il Codec Head (CB_0) e il Sub-Talker MTP (CB_1..CB_15)            │
│  - Sintetizzato a 24 kHz via Code2Wav Official Vocoder                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Il Modulo di Mapping: Struttura a Doppio Ramo (ADR-014)

Per garantire la massima purezza acustica senza collassi statistici, il mapping da $\mathbb{R}^{1536 \times 2} \to \mathbb{R}^{2048}$ viene scisso in due rami funzionali interni:

```
                            [ h_raw ]  ||  [ h_tx ] (3072D)
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           │                                                         │
           ▼                                                         ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│      RAMO A: SINTESI VOCALE V7      │   │     RAMO B: DINAMICHE & GATING      │
│            [CONGELATO]              │   │            [ADDESTRABILE]           │
│                                     │   │                                     │
│ • Audio MLP: 1536 -> 2048 (Temporal)│   │ • Causal Conv1D (k=3, left-padded)  │
│ • Text MLP:  1536 -> 2048 (ReZero)  │   │ • Pre-LN Deep Residual MLP          │
│ • Gain Calibrato: gamma = 0.25      │   │ • Logits Azioni: s_t in R^5         │
│                                     │   │ • Vettori Latenti di Stato:         │
│ => h_speech in R^2048               │   │   e_silence, e_yield, e_bc in R^2048│
│    (Fonetica e Dizione Italiana)    │   │ • Gate Continuo: g_t in [0.0, 1.0]  │
└──────────────────┬──────────────────┘   └──────────────────┬──────────────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │     GatedStateModulator       │
                        │                               │
                        │   h_cond[t] = gamma * Norm(   │
                        │     g_t * h_speech +          │
                        │     (1 - g_t) * e_silence +   │
                        │     p_yield * e_yield +       │
                        │     p_bc * e_bc               │
                        │   )                           │
                        └───────────────┬───────────────┘
                                        │ h_cond (2048D)
                                        ▼
                           Qwen3-TTS Talker Loop (Frozen)
```

### Regole Matematiche del Gating Continuo (VoiceChat-TTS & ASPIRin):
1. **Calcolo del Gate di Attivazione Parlato**:
   $$g_t = \sigma\left(\text{logit}_{\text{SPEAK}}[t] - \text{logit}_{\text{LISTEN}}[t]\right) \in [0.0, 1.0]$$
2. **Iniezione del Silenzio Pulito & Interruzione Fluida**:
   Quando l'utente parla o l'agente è in ascolto passivo ($g_t \to 0.0$), il segnale $h_{\text{speech}}$ viene azzerato e sostituito integralmente da $\mathbf{e}_{\text{silence}}$, impedendo qualsiasi riverbero o allucinazione acustica del Talker. In caso di interruzione (`BARGE_IN`), il vettore $\mathbf{e}_{\text{interrupt}}$ assicura una transizione morbida al silenzio senza crash di gradiente.
3. **Emissione di Backchannel Senza Interruzione Testuale**:
   Quando l'encoder $h_{\text{raw}}$ rileva una pausa prosodica dell'utente su una frase prolungata, il gate attiva per 160–240ms la componente $p_{\text{bc}} \cdot \mathbf{e}_{\text{bc}}$, producendo un assenso (*"mhm"*, *"sì"*) col medesimo timbro vocale dell'agente senza impegnare il decoder testuale di Gemma.
4. **Zero-Reset KV-Cache su Barge-In (VoiceChat-TTS Pattern)**:
   In caso di interruzione da parte dell'utente, lo `StreamingDuplexController` svuota immediatamente il buffer audio destinato alla scheda audio del client ($<10\text{ms}$), ma **non cancella la KV-cache del Talker**, che transiziona a silenzio tramite $\mathbf{e}_{\text{interrupt}}$, preservando la memoria di contesto acustico per il turno successivo.
5. **One-Token Lookahead Text Buffer (2-Subword Minimum Window)**:
   Per garantire la corretta intonazione e dizione italiana alle giunzioni di parola (*"l'amico"*, *"un'altra"*), il buffer di testo trattiene 1 token di lookahead prima di rilasciare la sequenza al proiettore, introducendo appena ~30-50ms di ritardo impercettibile.

---

## 4. Il Loop di Runtime e Serving Full-Duplex (12.5 Hz Synchronous Clock)

Il runtime opera a clock fisso sincrono di **80 ms per frame (12.5 Hz)** in un unico processo GPU monolitico:

```
TICK (t = n * 80ms)
│
├── 1. Ingestion: Mel Spectrogram Utente -> Conformer USM -> h_raw[t]
├── 2. Ragionamento: Gemma 4 aggiorna la KV-cache e genera token -> h_tx
├── 3. Arbitraggio: TurnActionHead valuta l'azione conversazionale a_t
│     ├── Se a_t == BARGE_IN:
│     │     Lo StreamingDuplexController invia {"op": "FLUSH"} al socket
│     │     La scheda audio svuota il buffer hardware (<10ms)
│     │     g_t = 0.0 (silenzio istantaneo)
│     ├── Se a_t == SPEAK:
│     │     g_t = 1.0 -> h_cond = gamma * Norm(h_speech)
│     │     Talker predice CB_0 + Sub-Talker predice CB_1..15
│     │     Code2Wav emette 1920 campioni PCM (24kHz) sul socket
│     └── Se a_t == EMIT_BACKCHANNEL:
│           Attivazione per 2 frame di e_bc -> micro-assenso vocale
└── 4. Output: Streaming chunk su WebSocket (:8000)
```

---

## 5. Roadmap di Integrazione Post-V7

Non appena il training `v7_thinker_talker_run` giungerà al termine (500 step):

1. **Salvataggio & Congelamento V7**:
   - Salvare `checkpoints/proj_v7_final.pt`.
   - Eseguire la sonda diagnostica di allineamento manifold (`tools/diagnostics_manifold_oracle.py`).
2. **Addestramento Leggero del Modulo Dinamico (`TurnDynamicsModule`)**:
   - Mantenere Gemma, Qwen e `proj_v7_final.pt` **rigorosamente congelati**.
   - Addestrare unicamente i ~1.8M parametri della `TurnActionHead` + `LearnableStateEmbeddings` sui 16 dialoghi stereo sintetici bilanciati con `CalibratedTurnFocalLoss` ($w = [1.0, 1.2, 4.0, 3.0, 2.5]$, $\gamma=1.2$).
   - Tempo stimato di addestramento: **< 3 minuti** su RTX 3090.
3. **Deployment Streaming Full-Duplex**:
   - Avvio del serving monoprocesso su porta 8000 via `./docker_run_3090.sh serve`.
   - Validazione interattiva del barge-in e della fluidità bidirezionale in tempo reale.
