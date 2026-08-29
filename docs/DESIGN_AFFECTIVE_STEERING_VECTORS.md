# DESIGN: Affective Prosody & Emotional Steering Vectors for Gemma4-Omni

## 1. Executive Summary & Vision

L'architettura **Gemma4-Omni (Strada B / Thinker-Talker)** genera parlato italiano fluente collegando gli stati latenti cognitivi di Gemma 4 E2B (`h_raw` e `h_tx`) al loop autoregressivo del Talker Qwen3-TTS tramite il `TextAnchoredProjectionV7` (congelato) e il modulo di gestione turni `TurnDynamicsModule` (ADR-014).

Questo documento definisce l'estensione per dotare il sistema di **controllo fine-grained della prosodia affettiva, del tono emotivo e dell'espressività vocale** mediante **Steering Vectors** (Activation Steering / Representation Engineering), senza riaddestrare né alterare i pesi del backbone linguistico (Gemma 4), del proiettore (`Projector V7`) né del vocoder neurale (`Code2Wav`).

---

## 2. Fondamenti Teorici & Riferimenti Scientifici (2025–2026)

Le scelte architetturali si fondano sui seguenti risultati empirici e teorici della letteratura recente:

1. **Sito Primario di Generazione Prosodica (Language Model vs Flow-Matching)**:
   * *CoCoEmo* ([arXiv:2602.03420](https://arxiv.org/abs/2602.03420), 2026) dimostra che la prosodia affettiva (traiettoria di $F_0$, contorni di energia, durata sillabica) è interamente governata dal modulo autoregressivo linguistico (**Talker / SLM**), mentre i moduli acustici a valle (flow-matching, MTP, vocoder) eseguono mero rendering armonico e tessitura timbrica.
2. **Disentanglement Geometrico (Speaker vs Emozione)**:
   * *A Geometric Perspective on Composable Emotion Steering* ([arXiv:2607.00946](https://arxiv.org/abs/2607.00946), Luglio 2026) dimostra tramite analisi di dimensionalità intrinseca locale (LID) che il Talker racchiude un sottospazio emotivo a bassa dimensionalità linearmente separabile e disaccoppiato dall'identità del parlante. Questo consente la combinazione lineare di emozioni multiple senza alterare l'identità vocale.
3. **Localizzazione Layer-Wise Ottimale**:
   * Negli strati intermedi-alti del Transformer del Talker (layer 12–16 su 28 in Qwen3-TTS 1.7B) e negli output di auto-attenzione (`attn_output`), la separabilità lineare delle emozioni raggiunge il picco massimo.
4. **Preservazione della Norma Euclidea**:
   * L'iniezione diretta di vettori $\mathbf{h} + \alpha \mathbf{v}$ altera la norma delle attivazioni causando degradazione della dizione (aumento del WER). La riscalatura euclidea $\tilde{\mathbf{h}} \leftarrow \frac{\|\mathbf{h}\|}{\|\tilde{\mathbf{h}}\|} \tilde{\mathbf{h}}$ garantisce zero artefatti fonetici e intelligibilità invariata.

---

## 3. Architettura del Sistema di Steering

Il sistema opera su due livelli coordinati:

```
                      ┌──────────────────────────────────────────────┐
                      │             Gemma 4 E2B (Cognizione)         │
                      │  h_raw (1536D)                 h_tx (1536D)  │
                      └───────┬───────────────────────────────┬──────┘
                              │                               │
                              ▼                               │
                   [TextAnchoredProjection V7]                │
                              │ (Frozen)                      │
                              ▼                               │
                       h_cond [t] (2048D)                     │
                              │                               │
                              ▼                               ▼
                   ┌───────────────────────┐       ┌──────────────────────┐
                   │  TurnDynamicsModule   │       │  GemmaToneExtractor  │
                   │  (e_silence, e_yield) │       │  (w_emo o v_latent)  │
                   └──────────┬────────────┘       └──────────┬───────────┘
                              │                               │
                              ▼                               ▼
                   ┌──────────────────────────────────────────────────────┐
                   │            AffectiveSteeringModule                   │
                   │  1) Ingress Steering: h_cond + α · v_emo             │
                   │  2) Deep Attention Hooking: Layers 12-16             │
                   │  3) Euclidean Norm Preservation                      │
                   └──────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                   ┌──────────────────────────────────────────────────────┐
                   │            Qwen3-TTS Talker 1.7B (Loop)              │
                   │     inputs_t = codec_agg_{t-1} + h_cond_steered      │
                   │     H_t (2048D)  ──> Codebook 0 (Prosodia Emotiva)   │
                   └──────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                   ┌──────────────────────────────────────────────────────┐
                   │             Hierarchical MTP Module                  │
                   │          Codebooks 1..14 (Timbro & Dettaglio)        │
                   └──────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                   ┌──────────────────────────────────────────────────────┐
                   │             Code2Wav Official Decoder                │
                   │                  Audio 24 kHz                        │
                   └──────────────────────────────────────────────────────┘
```

---

## 4. Formulazione Matematica

### 4.1. Estrazione dei Vettori di Steering (Contrastive Mean-Difference)
Dato un dataset di coppie controllate $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{a}_i, y_i)\}$ dove $y_i \in \{\text{neutral}, \text{empathetic}, \text{enthusiastic}, \text{calm}, \text{curious}, \text{assertive}\}$:

$$\mathbf{v}_e^{(l)} = \frac{1}{|\mathcal{D}^{(e)}|} \sum_{i \in \mathcal{D}^{(e)}} \mathbf{h}_i^{(l)} - \frac{1}{|\mathcal{D}^{(0)}|} \sum_{j \in \mathcal{D}^{(0)}} \mathbf{h}_j^{(l)}$$

Il vettore risultante viene normalizzato a norma unitaria:
$$\hat{\mathbf{v}}_e^{(l)} = \frac{\mathbf{v}_e^{(l)}}{\|\mathbf{v}_e^{(l)}\|_2}$$

### 4.2. Composizione Lineare di Emozioni Miste
Per esprimere toni composti (es. 70% Calma + 30% Empatia):
$$\mathbf{v}_{\text{mix}}^{(l)} = \sum_{k=1}^K w_k \cdot \hat{\mathbf{v}}_k^{(l)}, \quad \text{con } \sum_{k=1}^K w_k = 1.0, \; w_k \ge 0$$

### 4.3. Iniezione in Inferenza & Preservazione della Norma
Dato lo stato di attivazione $\mathbf{h} \in \mathbb{R}^d$ al punto di iniezione (sia in $h_{\text{cond}}$ che all'uscita dell'attenzione del layer $l$):

$$\mathbf{h}' = \mathbf{h} + \alpha \cdot \mathbf{v}_{\text{mix}}^{(l)}$$
$$\tilde{\mathbf{h}} = \frac{\|\mathbf{h}\|_2}{\|\mathbf{h}'\|_2 + \epsilon} \cdot \mathbf{h}'$$

dove $\alpha \in [0.0, 3.5]$ è il fattore scalare continuo di intensità affettiva:
* $\alpha = 0.0$: tono puramente neutro/canonico (comportamento baseline V7).
* $\alpha = 1.0 - 1.5$: sfumatura emotiva naturale e colloquiale.
* $\alpha = 2.0 - 3.0$: espressività marcata (enfasi teatrale o forte intonazione).

---

## 5. Dati Necessari per l'Estrazione e la Validazione

L'approccio basato su steering vectors è **training-free per i backbone**: richiede solo dataset mirati per l'estrazione e la calibrazione direzionale.

### 5.1. Dataset di Parlato Emotivo Richiesti
1. **EMOVO (Italian Emotional Speech Corpus)**:
   * **Lingua**: Italiano nativo (6 attori, 3M/3F, 14 frasi per 6 emozioni: neutro, gioia, sorpresa, tristezza, rabbia, paura, disgusto).
   * **Uso**: Estrazione diretta dei vettori acustici di contrasto per il Talker Qwen3-TTS a parità di frase e speaker.
2. **ESD (Emotional Speech Dataset)**:
   * **Lingua**: Bilingue Inglese/Cinese (350 frasi parallele per 10 speaker su 5 emozioni).
   * **Uso**: Benchmark incrociato e validazione delle direzioni prosodiche universali ($F_0$ shift, duration).
3. **Synthetic Italian Contrastive Dialogue Pairs (Gemma-Generated)**:
   * **Descrizione**: 100-200 coppie di prompt/risposte generate con Gemma 4 forzando direttive di stile (es. *"Rispondi con tono empatico e rassicurante"* vs *"Rispondi con tono asettico e neutro"*).
   * **Uso**: Estrazione delle direzioni di tono cognitivo $\mathbf{v}_{\text{Gemma}} \in \mathbb{R}^{1536}$.

---

## 6. Punti di Integrazione & Roadmap Post-V7

L'attivazione del modulo avverrà in modo strettamente sequenziale:

1. **Step 1 (In Corso)**: Completamento training Projector V7 (`v7_thinker_talker_run`) e convergenza loss semantica/acustica.
2. **Step 2**: Validazione del `TurnDynamicsModule` in ambiente interattivo (FSM di ascolto, barge-in e yield).
3. **Step 3**: Estrazione del banco di vettori affettivi tramite `tools/extract_affective_steering_vectors.py`.
4. **Step 4**: Attivazione del `AffectiveSteeringModule` con sweep $\alpha \in [0.0, 3.0]$ su campioni audio di verifica (`tools/eval_affective_steering.py`).
