# ARCHITECTURE STATE — Stato Consolidato V9 (27 Agosto 2026)

**Checkpoint Canonico di Riferimento**: `checkpoints/v9_prefix_step10000.pt` (e `v9_prefix_final.pt`).
**Prestazioni Acustiche Dimostrate**:
- **Crest Factor**: **`3.92 – 4.85`** (Pieno range vocale umano naturale, volume omogeneo senza saturazione).
- **HNR (Harmonics-to-Noise)**: **`+3.97 dB`** (Massimo storico del progetto).
- **THD+N (Distorsione)**: **`28.61%`** (Superato l'obiettivo critico di scendere sotto il 30%).
- **ASR Faster-Whisper**: **`100% di accuratezza lessicale`** su prompt conversazionali live complessi in lingua italiana.
- **Architettura Standard**: **Deep Causal Transformer Projector V9 (211.83M parametri)** con **Prefix KV-Cache Conditioning** conforme ad ADR-022, ADR-023, ADR-024 e **Qwen3.5-Omni (arXiv:2604.15804)**.
- **Vincolo di Sistema Inderogabile**: **Tutto il blocco acustico Qwen3-TTS (Talker 1.7B + SubTalker 175M + Codec Decoder 24kHz) rimane 100% CONGELATO.**

---

## 📜 1. CRONOLOGIA DEI PARADIGMI E RISULTATI SPERIMENTALI

```
[V8.0–V8.5: MLP 25M a Iniezione Additiva] ──> Manifold Shift a Layer 0, Crest Factor compresso (2.3), Distorsione acustica elevata.
                 │
                 ▼
[V8.6 Finale: Prefix KV-Cache su MLP 25M] ──> Eliminato Manifold Shift, Crest Factor a 4.07, ma saturazione rapida della capacità (25M).
                 │
                 ▼
[V9 Deep Causal Transformer 211.8M (Svolta)] ──> 4 Layer Transformer con RoPE, SwiGLU, RMSNorm e Moshi 100:1 Rebalancing.
                 │                               • Step 1000: Voce limpida, Crest Factor 6.29, HNR 2.24 dB, ASR 7/9 parole.
                 │                               • Step 3000: Val Loss CB0 a 2.393 (minimo storico), HNR 2.52 dB, THD+N 35.87%.
                 │                               • Risolto l'artefatto di coda integrando il cutoff post-parlato EOS/VAD.
                 ▼
[V9 10k Run Completo & ARIA Streaming] ───────> • Step 10.000 completato: Val CB0 Loss 2.294, Val Accuracy 40.8%, Val Res Loss 10.210.
                                                • Studio Ablativo: CON Projector V9 dimezza la distorsione armonica (28.6% vs 46.0%).
                                                • ARIA Streaming a blocchi 640ms validato con continuità di fase bit-exact.
```

---

## 🔬 2. SPECIFICHE DELL'ARCHITETTURA DEL PROIETTORE V9

* **Parametri Totali**: **`211.838.208`** (211.83M parametri `bfloat16`).
* **Input Layer**: `audio_in` (`Linear(1536, 2048)`) + `text_in` (`Linear(1536, 2048)`).
* **Transformer Core**:
  - 4 Blocchi Transformer causali ($d_{\text{model}} = 2048$, $n_{\text{heads}} = 16$, $d_{\text{head}} = 128$).
  - Positional Embedding: **RoPE** ($\theta = 10000.0$).
  - Attivazione FFN: **SwiGLU** ($d_{\text{ffn}} = 5632$).
  - Normalizzazione: **Pre-RMSNorm** ($\epsilon = 10^{-6}$) e `final_norm`.

---

## ⚡ 3. SPECIFICHE DEL TRAINING ENGINE (PARALLEL 3D MROPE)

* **Formulazione Parallela**: Singolo forward pass per batch ($B=4, T=64$) sulla sequenza:
  $$\mathbf{X}_{\text{train}} = \big[\, \text{ICL}_{\text{Text}} \;\|\; \mathbf{h}_{\text{pfx}} \;\|\; \text{BOS} \;\|\; \text{agg\_audio}_{0\dots T-2} \,\big] \in \mathbb{R}^{B \times L_{\text{tot}} \times 2048}$$
* **Allineamento Target**: Posizione $\text{BOS} \to CB_0[0]$; posizioni $\text{agg\_audio}_t \to CB_0[t+1]$.
* **Moshi Loss Rebalancing (100:1)**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(CB_0, \widehat{CB}_0) + 0.01 \sum_{g=1}^{15} \mathcal{L}_{\text{CE}}(CB_g, \widehat{CB}_g)$$
* **Velocità**: $0.59\text{s/step}$ su RTX 3090 (10.000 step eseguiti in 98 minuti).
* **Parametri di Inferenza Canonici (ADR-023)**:
  - `repetition_penalty = 1.0` (rigorosamente disabilitata su RVQ audio per preservare le formanti).
  - `top_k = 20` su $CB_0$ con $T = 0.60$.
  - Iniezione sistematica $+ \mathbf{e}_{\text{tts\_pad}}$ ad ogni frame ricorsivo (gain piatto, zero saturazione).
  - SubTalker 12L campionato (`top_k=20, top_p=0.90, temperature=0.80`).
  - Pacing ARIA a $4.8\text{ frame/parola}$ con decodifica incrementale vocoder ogni 8 frame ($640\text{ms}$).
