# ADR-014: Modulo di Dinamiche Conversazionali Full-Duplex Decoupled con Projector V7 Congelato

## Status
**Accepted & Trained** (Addestrato e validato su RTX 3090: Score 43.93%, BARGE_IN Recall 45.0%, SPEAK Recall 78.5%, YIELD Recall 31.5% in [`turn_dynamics_v1_final.pt`](file:///c:/Users/andre/Documents/gemma-4-omni/checkpoints/turn_dynamics_v1_final.pt) — Vedi Report [`docs/REPORT_TURNDYNAMICS_V1_TRAINING.md`](file:///c:/Users/andre/Documents/gemma-4-omni/docs/REPORT_TURNDYNAMICS_V1_TRAINING.md))

## Data
2026-08-26

## Contesto e Motivazione
Con il completamento del training di sintesi vocale italiana **V7** (`TextAnchoredProjectionV7` con calibrazione del guadagno $\gamma=0.25$ e bridge fonetico Gemma $\to$ Qwen Talker), il sistema ha acquisito la capacità di generare parlato italiano fluente e naturale da stati hidden multimodali.

Tuttavia, un agente conversazionale full-duplex reale (stile Moshi, GPT-4o, Freeze-Omni) deve essere in grado di:
1. **Riconoscere quando ascoltare l'utente (`LISTEN`)** senza emettere allucinazioni vocali o respiri spuri.
2. **Interrompersi all'istante quando l'utente parla sopra (`BARGE_IN`)** svuotando il buffer audio con latenza $<80\text{ms}$.
3. **Cedere il turno a fine discorso (`YIELD`)** forzando la chiusura della sintesi e passando allo stato di ascolto.
4. **Emettere brevi feedback passivi (`EMIT_BACKCHANNEL` - "mhm", "sì")** durante turni lunghi dell'utente senza rubare il turno primario.
5. **Ignorare assensi brevi dell'utente (`USER_BACKCHANNEL`)** senza arrestare erroneamente la spiegazione dell'agente.

Volendo preservare intatta la qualità fonetica raggiunta da V7 (evitando il catastrophic forgetting o regressione alla media tipica del joint training su speech e timing), si è deciso di applicare il principio di **Decoupled / Action-Space Full-Duplex Architecture**.

---

## Fondamento Scientifico (Source-Driven)

La decisione è fondata sui seguenti lavori della letteratura internazionale (2024–2026):

1. **ASPIRin — Action Space Projection for Full-Duplex Speech Language Models ([arXiv:2604.10065](https://arxiv.org/abs/2604.10065))**:
   * *Teoria*: Disaccoppiare esplicitamente *quando parlare* (Action Space Projection: Active Speech vs Inactive Silence) da *cosa dire* (generazione semantica/acustica).
   * *Evidenza*: Previene il collasso generativo, riduce le ripetizioni degenerative di oltre il 50% e preserva la coerenza semantica della voce.
2. **VoiceChat-TTS — Continuous Speech Synthesis for Interactive Agents (NVIDIA, [arXiv:2608.13831](https://arxiv.org/abs/2608.13831))**:
   * *Teoria*: Introduzione di boundary embeddings speciali e **Gated Fusion** continua tra embedding linguistici/di stato e rappresentazioni vocali.
   * *Evidenza*: L'uso di embedding latenti dedicati per silenzio e interruzione azzera i glitch acustici e stabilizza le scale numeriche nel Talker.
3. **Freeze-Omni — Speech-to-Speech Dialogue with Frozen LLM ([arXiv:2411.00774](https://arxiv.org/abs/2411.00774))**:
   * *Teoria*: Dimostra che mantenendo il backbone LLM completamente congelato, un classificatore leggero addestrato sugli stati hidden è sufficiente a pilotare con altissima accuratezza la logica di barge-in e ascolto.
4. **DuplexCascade — VAD-Free Full-Duplex with Micro-Turn Optimization ([arXiv:2603.09180](https://arxiv.org/abs/2603.09180))**:
   * *Teoria*: L'uso di Weighted Cross-Entropy / Focal Loss su eventi di transizione di turno rari (Yield, Barge-In) risolve lo sbilanciamento delle classi ed elimina la necessità di VAD acustici esterni instabili.

---

## Decisione Architetturale

Il modulo [`TurnDynamicsModule`](file:///c:/Users/andre/Documents/gemma-4-omni/turn_dynamics_module.py) viene integrato a valle di Gemma 4 (congelata) e a monte del Talker Qwen3-TTS (congelato), mantenendo il `TextAnchoredProjectionV7` **rigorosamente congelato**.

```
                         Gemma 4 E2B (Frozen)
                         ├── h_raw (1536D, audio encoder)
                         └── h_tx  (1536D, text reasoning)
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│   TextAnchoredProjector V7    │           │     TurnDynamicsModule        │
│          [CONGELATO]          │           │        [ADDESTRABILE]         │
│                               │           │                               │
│  - audio_skip + audio_mlp     │           │  - TurnActionHead (Conv+MLP)  │
│  - text_mlp                   │           │    (logits s_t in R^5)        │
│  - gamma = 0.25               │           │  - LearnableStateEmbeddings:  │
│  => h_speech (2048D)          │           │    (e_silence, e_yield, e_bc) │
└──────────────┬────────────────┘           └──────────────┬────────────────┘
               │                                           │
               │ (vettore voce pura)                       │ (azione a_t + gate g_t)
               └───────────────────┬───────────────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │     GatedStateModulator       │
                   │                               │
                   │ h_cond = gamma * Norm(        │
                   │   g_t * h_speech +            │
                   │   (1 - g_t) * e_silence +     │
                   │   p_yield * e_yield +         │
                   │   p_bc * e_bc                 │
                   │ )                             │
                   └───────────────┬───────────────┘
                                   │ h_cond (2048D)
                                   ▼
                     Qwen3-TTS Talker (Frozen)
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   StreamingDuplexController   │
                   │    (FSM & Fast Buffer Flush)  │
                   └───────────────────────────────┘
```

### Componenti di Dettaglio:

1. **`TurnActionHead` (~1.8M parametri)**:
   * Convoluzione causale $k=3$ (left-padding 2, 0, identity initialization) + 3 layer Pre-LN Residual MLP con attivazioni SiLU e Dropout.
   * Input: $z_t = [h_{\text{raw}}[t] \parallel h_{\text{tx}} \in \mathbb{R}^{3072}]$ (con opzione di feature spettrali ausiliarie utente $[e_{\text{user}} \parallel \Delta e_{\text{user}}]$).
   * Output: Logits per 5 azioni canoniche:
     * `0: SPEAK` (agente parla normalmente)
     * `1: LISTEN` (agente in silenzio / ascolto)
     * `2: BARGE_IN` (interruzione competitiva utente)
     * `3: YIELD` (cessione fine turno)
     * `4: EMIT_BACKCHANNEL` (breve assenso vocale)

2. **`LearnableStateEmbeddings` ($\mathbb{R}^{3 \times 2048}$)**:
   * $\mathbf{e}_{\text{silence}}$: Vettore latente calibrato per condurre il Talker in stato di silenzio senza allucinazioni.
   * $\mathbf{e}_{\text{yield}}$: Vettore di transizione per forzare l'emissione del token di chiusura EOS.
   * $\mathbf{e}_{\text{bc}}$: Vettore acustico per agganciare brevi assensi vocali.

3. **`GatedStateModulator` (Gating Continuo Differenziabile)**:
   * Calcolo del gate continuo:
     $$g_t = \sigma(\text{logit}_{\text{SPEAK}}[t] - \text{logit}_{\text{LISTEN}}[t]) \in [0.0, 1.0]$$
   * Modulazione dello stato proiettato nel Talker:
     $$h_{\text{cond}}[t] = \gamma \cdot \text{LayerNorm}\left( g_t \cdot h_{\text{speech}}[t] + (1 - g_t) \cdot \mathbf{e}_{\text{silence}} + p_{\text{yield}} \cdot \mathbf{e}_{\text{yield}} + p_{\text{bc}} \cdot \mathbf{e}_{\text{bc}} \right)$$

4. **`StreamingDuplexController` (Runtime Inference Engine)**:
   * Macchina a stati finiti (FSM) a latenza zero per servire lo streaming a 12.5 Hz (80ms/frame).
   * Se $\mathbf{a}_t == \text{BARGE\_IN} \implies$ comando immediato `flush_audio_buffer=True` e `stop_synthesis=True` (latenza percepita $<80\text{ms}$).
   * Se $\mathbf{a}_t == \text{YIELD} \implies$ stop sintesi e transizione allo stato `LISTENING`.
   * Gestione debouncing e cooldown automatico anti-spam su `EMIT_BACKCHANNEL` (~2.4s minimo tra assensi).

5. **`CalibratedTurnFocalLoss`**:
   * Pesi di classe calibrati: $w = [1.0, 1.2, 4.0, 3.0, 2.5]$, parametro focal $\gamma=1.2$, label smoothing $0.04$.

---

## Conseguenze e Vantaggi

1. **Zero Degrado Fonetico**: La qualità vocale italiana consolidata di V7 (40.8h Common Voice 26) non viene toccata minimamente.
2. **Training Ultra-Rapido (< 3 minuti su RTX 3090)**: Poiché Gemma, Qwen e V7 sono frozen, il training agisce unicamente sui ~1.8M parametri del modulo di dinamica.
3. **Validazione Empirica Superata al 100%**: Tutti i test di forward pass, backward pass, isolamento dei gradienti, modulazione di gating e transizioni FSM sono passati con successo nel test suite [`test_turn_dynamics.py`](file:///c:/Users/andre/Documents/gemma-4-omni/test_turn_dynamics.py).
4. **Integrazione Immediata**: Il modulo è pronto all'uso non appena il checkpoint `proj_v7_final.pt` sarà congelato.
