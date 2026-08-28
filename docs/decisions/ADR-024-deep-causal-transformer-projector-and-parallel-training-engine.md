# ADR-024: Architettura del Deep Causal Transformer Projector V9 e del Parallel 3D MRoPE Training Engine

- **Stato**: Accettato / Consolidato in Produzione
- **Data**: 27 Agosto 2026
- **Autori**: Gemma 4 Omni Team
- **Contesto**: Consolidamento definitivo dell'architettura del Projector V9 (211.83M parametri) e del motore di addestramento parallelo denso a 10.000 step che ha portato la Val CB0 Loss da 14.48 a 2.294 con accuratezza al 40.8% e sintesi vocale naturale 100% fedele.

---

## 1. Architettura del Projector V9 (Deep Causal Transformer)

Il **Projector V9** è il modulo ponte causale che converte le rappresentazioni latenti multimodali del Thinker (**Gemma 4 E2B**, $1536$-dim) nel prefisso continuo di condizionamento iniettato nella KV-Cache del Talker (**Qwen3-TTS 1.7B**, $2048$-dim).

```
  Gemma 4 Token States h_raw (1, T_pfx, 1536)     Gemma 4 Inner Monologue h_tx (1, 1536)
                     │                                              │
              [audio_in Linear]                              [text_in Linear]
              (1536 -> 2048)                                 (1536 -> 2048)
                     │                                              │
                     └──────────────────────┬───────────────────────┘
                                            │ (+) Somma broadcast
                                            ▼
                                  ┌───────────────────┐
                                  │   x_0 (T_pfx, d)  │ (d = 2048)
                                  └─────────┬─────────┘
                                            │
               ┌────────────────────────────┴───────────────────────────┐
               │              4x TRANSFORMER BLOCKS (Causali)          │
               │                                                        │
               │  ┌──────────────────────────────────────────────────┐  │
               │  │ Pre-RMSNorm (d=2048, eps=1e-6)                   │  │
               │  │ Causal Multi-Head Self-Attention (16h, RoPE)     │  │
               │  │ Residual Add (+)                                 │  │
               │  │ Pre-RMSNorm (d=2048, eps=1e-6)                   │  │
               │  │ SwiGLU FFN (d=2048 -> 5632 -> 2048)              │  │
               │  │ Residual Add (+)                                 │  │
               │  └──────────────────────────────────────────────────┘  │
               └────────────────────────────┬───────────────────────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │   Final RMSNorm   │
                                  └─────────┬─────────┘
                                            │
                                            ▼
                       h_pfx = Prefix KV-Cache Embedding (1, 16, 2048)
```

### Specifiche Tecniche e Parametri

* **Parametri Totali**: **`211.838.208`** ($211.83\text{M}$ parametri in precisione `bfloat16`).
* **Input Layer**:
  - `audio_in`: `nn.Linear(1536, 2048, bias=False)` proietta la sequenza di stati acustici interpolati $h_{\text{raw}} \in \mathbb{R}^{B \times T_{\text{pfx}} \times 1536}$.
  - `text_in`: `nn.Linear(1536, 2048, bias=False)` proietta il vettore di contesto semantico globale $h_{\text{tx}} \in \mathbb{R}^{B \times 1536}$.
* **Transformer Core (4 Strati Identici)**:
  - **Dimensione Modello ($d_{\text{model}}$)**: $2048$.
  - **Numero di Head ($n_{\text{heads}}$)**: $16$ head con dimensione $d_{\text{head}} = 128$.
  - **Positional Encoding**: **RoPE** (*Rotary Position Embedding*, $\theta = 10000.0$) applicato alle matrici $Q$ e $K$ ad ogni strato.
  - **Attenzione**: *Scaled Dot-Product Causal Self-Attention* con maschera triangolare inferiore per garantire perfetta causalità temporale.
  - **Feed-Forward**: **SwiGLU FFN** con espansione a $d_{\text{ffn}} = 5632$:
    $$\text{FFN}(x) = W_{\text{down}} \big(\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x\big)$$
    dove $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{5632 \times 2048}$ e $W_{\text{down}} \in \mathbb{R}^{2048 \times 5632}$.
  - **Normalizzazione**: Pre-Layer Normalization con **RMSNorm** ($\epsilon = 10^{-6}$) su attenzione e FFN, più `final_norm` prima dell'uscita.

---

## 2. Architettura del Training: Parallel 3D MRoPE Engine

L'addestramento di V9 è stato reso possibile dalla transizione dal vecchio loop autoregressivo sequenziale ($30.5\text{s/step}$) al motore **Dense Parallel 3D MRoPE** ($0.59\text{s/step}$, accelerazione $51.7\times$).

### 2.1 Formulazione della Sequenza Parallela

Durante ogni batch di addestramento ($B=4, T=64$), l'intera sequenza di input viene concatenata ed eseguita in un **singolo passaggio forward parallelo**:

$$\mathbf{X}_{\text{train}} = \big[\, \underbrace{\text{ICL}_{\text{Text}}}_{L_{\text{txt}}} \;\|\; \underbrace{\mathbf{h}_{\text{pfx}}}_{T=16} \;\|\; \underbrace{\text{BOS}}_{1} \;\|\; \underbrace{\text{agg\_audio}_{0\dots T-2}}_{T-1} \,\big] \in \mathbb{R}^{B \times L_{\text{tot}} \times 2048}$$

dove l'embedding audio aggregato canonico al tempo $t$ è:
$$\text{agg\_audio}_t = \mathbf{e}_{\text{CB0}}(c_{t,0}) + \sum_{g=1}^{15} \mathbf{e}_{\text{sub}, g}(c_{t,g}) + \mathbf{e}_{\text{tts\_pad}}$$

### 2.2 Allineamento Causale dei Target di Predizione

Grazie alla maschera di attenzione causale del Talker:
1. **Posizione del token $\text{BOS}$** (indice $L_{\text{txt}} + T_{\text{pfx}}$): il Talker ha accesso visivo causale al testo e a tutto il prefisso $h_{\text{pfx}}$, e predice il primo codice acustico $\widehat{CB}_0[0]$.
2. **Posizione del token $\text{agg\_audio}_t$** (indice $L_{\text{txt}} + T_{\text{pfx}} + 1 + t$): il Talker predice il codice successivo $\widehat{CB}_0[t+1]$ per tutti i frame $t = 0 \dots T-2$ simultaneamente in parallelo su GPU.

```
Posizioni Input:  [ ICL_Text ... | h_pfx_0 .. h_pfx_15 | BOS | agg_0 | agg_1 | ... | agg_T-2 ]
                                                         │       │       │             │
                                                         ▼       ▼       ▼             ▼
Target Predetti:                                       CB0[0]  CB0[1]  CB0[2]  ...   CB0[T-1]
```

### 2.3 Moshi Loss Rebalancing (100:1)

Poiché il Talker genera 16 codebook totali ($CB_0$ semantico + $CB_1 \dots CB_{15}$ acustici fini), la loss canonica non ponderata attribuisce il $93.75\%$ del gradiente ai residui acustici secondari, soffocando l'apprendimento di $CB_0$.

Abbiamo adottato il bilanciamento asimmetrico **Moshi Loss Rebalancing (100:1)**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(CB_0, \widehat{CB}_0) + \lambda_{\text{res}} \sum_{g=1}^{15} \mathcal{L}_{\text{CE}}(CB_g, \widehat{CB}_g) \quad \text{con } \lambda_{\text{res}} = 0.01$$

Questo garantisce che il gradiente del Projector V9 ottimizzi con priorità assoluta la predizione del codebook semantico principale $CB_0$.

---

## 3. Risultati Empirici della Traiettoria a 10.000 Step

Addestrato sul dataset Common Voice 26 Italiano ($40.8\text{ ore}$, $25.791$ sequenze) su host remoto con NVIDIA GeForce RTX 3090 ($24\text{GB}$ VRAM):

* **Velocità di Calcolo**: $0.59\text{s / batch}$ ($B=4$) $\implies \mathbf{10.000\text{ step in 98 minuti}}$.
* **Evoluzione della Validation Loss**:
  - Step 1: Loss iniziale $\mathbf{14.48}$ (Top-1 Accuracy $0.0\%$).
  - Step 3.000: Val CB0 Loss $\mathbf{2.391}$ (Accuracy $39.4\%$).
  - Step 7.000: Val CB0 Loss $\mathbf{2.315}$ (Accuracy $41.1\%$).
  - Step 8.100: Val CB0 Loss **`2.293`** (Minimo Globale CB0).
  - Step 8.700: Val CB0 Accuracy **`41.6%`** (Massimo Globale Accuracy, picchi batch a $60.9\%$).
  - Step 10.000: Val CB0 Loss **`2.294`**, Val Residual Loss **`10.210`** (Minimo Globale Residui).
* **Verifica di Generalizzazione**: Piattaforma asintotica perfetta (Flat Minima) tra step 8.000 e 10.000, con zero evidenza di overfitting.

---

## 4. Validazione Finale & Impatto sull'Inferenza

| Configurazione | Val CB0 Loss | Acc Top-1 | HNR | THD+N | Qualità Sintesi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Solo Testo (No Projector)** | — | — | $+0.70\text{ dB}$ | $45.96\%$ | Voce con incertezze armoniche e pause trascinate |
| **V8.0 (MLP 2-Layer Additivo)** | $3.416$ | $28.3\%$ | $+1.80\text{ dB}$ | $39.10\%$ | Manifold shift al Layer 0 del Talker |
| **V9 (Deep Causal Transformer 10k)** | **`2.294`** | **`40.8%`** | **`+3.97 dB`** | **`28.61%`** | **Voce naturale, ritmo compatto, 100% fedeltà fonetica** |

Il checkpoint `/workspace/checkpoints/v9_prefix_step10000.pt` è ratificato come **standard canonico di produzione** per Gemma 4 Omni.
