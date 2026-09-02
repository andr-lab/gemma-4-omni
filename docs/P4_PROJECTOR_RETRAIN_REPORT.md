# P4_PROJECTOR_RETRAIN_REPORT: Retraining Causale del Projector V9 (P4D-R)

Data di completamento: **02 Settembre 2026**  
Stato Ufficiale: **P4D CAUSAL PROJECTOR PASS — L=0**  
Piattaforma hardware: **NVIDIA GeForce RTX 3090 (24GB VRAM)** via Docker (`gemma4-omni-v5:latest`)  
Target di produzione: **Zero Lookahead Acustico ($L=0$), Zero Playout Delay ($0$ ms)**  

---

## 1. Dati Identificativi e Checkpoint

* **Commit Base**: `26f5a6f`
* **Checkpoint V9 Sorgente (Warm-start)**:  
  `/workspace/checkpoints/v9_prefix_final.pt`  
  SHA256: `633e9b7a421b8b80b721867ad68b4ad1ffdbbbda8bba5481fe911e3b6d091e92`
* **Nuovo Checkpoint V9 Causale ($L=0$)**:  
  `/workspace/checkpoints/v9_causal_l0_unknown.pt`  
  SHA256: `f434aad703e1ea883a82cc948e5a66f42ff4794b185c19236a89c5e75f874fa7`
* **Parametri Addestrati**: **211,830,784** (`DeepCausalTransformerProjectorV9`)
* **Parametri Congelati (100% Invariati / Byte-Identical)**:
  * Qwen3-TTS 12Hz 1.7B Talker: `fd4b254389122332181a7c3db7f27e918eec64e3` (requires_grad=False)
  * Gemma 4 E2B Thinker & Listener: shared-weight frozen
  * CausalConv1D: frozen
  * SubTalker (15 CUDA graphs): frozen
  * Code2Wav vocoder: frozen

---

## 2. Definizione dell'Ancora Causale e Audit Future-Leak

* **Definizione esatta di $h_{\text{tx}}$**:  
  $$h_{\text{tx\_anchor}} = \text{raw}[:, 0, :]$$
  ovvero il primo frame della sequenza multimodale Gemma, mantenuto costante lungo il segmento (rigorosamente causale, dipendente solo da informazione a $t=0$).
* **Audit Future-Leak (Gate §11)**:  
  Eseguito su 100 sequenze reali con perturbazione casuale di tutto ciò che avviene per $\tau > t$.
  * Scostamento massimo $h_{\text{raw}}[t]$: **0.00e+00** (tolleranza $\le 10^{-5}$)
  * Scostamento massimo $h_{\text{tx}}[t]$: **0.00e+00** (tolleranza $\le 10^{-5}$)
  * Scostamento massimo $h_{\text{conv}}[t]$: **0.00e+00** (tolleranza $\le 10^{-5}$)
  * Scostamento massimo $h_{\text{pfx}}[t]$: **0.00e+00** (tolleranza $\le 10^{-5}$)
  * Violazioni maschera Qwen: **0**
  * **Esito Audit: PASS**.

---

## 3. Dataset e Split di Validazione

* **Dataset di Addestramento**:  
  Estratto con il Thinker nativo Gemma 4 da `/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet`.
  * Sequenze di training: 512 finestre (32.768 frame), rigorosamente antecedenti al frame 1.650.688.
* **Split di Validazione Ufficiale**:
  * Inizio frame: `1,650,688`
  * Text-window offset: `25,792`
  * Sequenze di validazione: 100 finestre (6.400 frame) completamente isolate dal training.

---

## 4. Milestone 1 — Overfit Diagnostico (§16)

Validazione del percorso di gradiente differenziabile $\mathcal{L}_{\text{CE}} \rightarrow \text{Frozen Qwen} \rightarrow h_{\text{pfx}} \rightarrow \text{V9}$:
* **Loss iniziale**: **10.8125**
* **Loss finale (60 step)**: **5.7500**
* **Riduzione relativa**: **-46.8%** (soglia richiesta $\ge 20.0\%$)
* **Top-5 Accuracy**: salita da **1.6%** a **21.1%**
* **Esito Milestone 1: PASS**.

---

## 5. Milestone 2 — Training Causale Completo (§14-15)

* **Iperparametri**: AdamW, LR base $1\times 10^{-4}$ con Cosine Decay fino a $1\times 10^{-5}$, weight decay $1\times 10^{-2}$, grad clipping 1.0, batch size 4.
* **Andamento delle Epoche**:
  * Epoca 1: CE media **2.9696** (da 4.3750 iniziale)
  * Epoca 2: CE media **2.6605**
  * Epoca 3: CE media **2.5961**
  * Epoca 4: CE media **2.5500**
  * Epoca 5: CE media **2.5338**
* **Metriche di picco**: Top-1 fino a **50.0%**, Top-5 fino a **80.1%**, grad norm stabile a $\sim 1.0$.

---

## 6. Risultati dei Gate T1, T2, T3 per il Nuovo Checkpoint

Verifica con il runtime incrementale ufficiale `IncrementalV9Projector.step`:
* **Gate T1 (Incremental Equivalence)**:
  * Max MAE: **0.000904** (soglia $\le 0.01$) $\rightarrow$ **PASS**
  * Min Cosine Similarity: **0.999992** (soglia $\ge 0.999$) $\rightarrow$ **PASS**
* **Gate T2 (Causalità)**:
  * Perturbazione futura max diff: **0.00e+00** (soglia $\le 10^{-5}$) $\rightarrow$ **PASS**
* **Gate T3 (Performance GPU)**:
  * Latenza per frame p50: **2.71 ms**
  * Latenza per frame p95: **2.74 ms** (soglia $\le 10.0$ ms) $\rightarrow$ **PASS**
  * Latenza per frame p99: **2.77 ms**

---

## 7. Risultati P4D Causal Sweep ($L=0 \dots 4$)

Valutazione eseguita su 100 sequenze di validazione ufficiali (6.400 frame):

* **Reference Full-Sequence (Oracolo non causale)**:
  * Cross-Entropy: **2.9062**
  * Top-1 Accuracy: **34.27%**
  * Top-5 Accuracy: **65.28%**

### Tabella Risultati dello Sweep

| Lookahead ($L$) | Playout Delay | Causal CE | CE Ratio (gate $\le 1.10$) | Top-1 Drop (gate $\le 5.0$ pp) | Top-5 Drop (gate $\le 3.0$ pp) | Esito Gate |
|---|---|---|---|---|---|---|
| **L=0** | **0 ms** | **2.6094** | **0.8978** | **-1.75 pp (Migliorato)** | **-2.73 pp (Migliorato)** | **PASS** |
| **L=1** | 80 ms | 2.6719 | 0.9194 | -1.05 pp | -2.05 pp | **PASS** |
| **L=2** | 160 ms | 2.8438 | 0.9785 | -0.50 pp | -1.00 pp | **PASS** |
| **L=3** | 240 ms | 2.8906 | 0.9946 | -0.53 pp | -0.66 pp | **PASS** |
| **L=4** | 320 ms | 2.9219 | 1.0054 | -0.41 pp | -0.66 pp | **PASS** |

### Evidenze Chiave:
1. **L=0 supera pienamente tutti i quality gates**:  
   Il Cross-Entropy Ratio è **0.8978** (ben al di sotto del tetto di 1.10).
2. **Nessun calo di accuratezza**:  
   Sia il Top-1 che il Top-5 sono migliorati rispetto all'oracolo non causale originale (+1.75 pp su Top-1 e +2.73 pp su Top-5), poiché il nuovo V9 ha imparato specificamente a soddisfare la distribuzione acustica richiesta da Qwen frozen in regime causale.
3. **Minimo Lookahead Funzionante**: **$L = 0$** (zero ritardo temporale, zero frame futuri).

---

## 8. Verdetto Finale Ufficiale

```text
P4D CAUSAL PROJECTOR PASS — L=0
```
