# Gemma 4 Full-Duplex Omni Pipeline - Stato Attuale e Prossimi Passi

Documento di stato del progetto • Aggiornato al 19 Agosto 2026

---

## 1. Stato Attuale del Progetto (Executive Summary)

Il progetto sviluppa una pipeline conversazionale Speech-to-Speech Full-Duplex real-time accoppiando **Gemma 4 E2B** (305M USM-Conformer audio input) con il **sub-talker congelato di Qwen3-TTS-12Hz-1.7B-Base** (~363MB) e il decoder **Code2Wav** (24kHz PCM).

La fase di allineamento e distillazione latente (FitNets Stage 2: Gemma hidden -> adapter -> h_t teacher) e' stata validata empiricamente attraverso tre generazioni di adapter (R1, R2, R3) e molteplici run Kaggle su GPU T4 (16GB).

---

## 2. Cosa e' stato fatto (R1, R2, R3 e Run Completo v7)

### A. Fedelta' e Validazione Sub-Talker (Oracle)
- **Verifica Wrapper vs Reference Teacher**: Replica bit-identica del `code_predictor` eager/bf16 (max diff logits = 0.00e+00, CE oracolare = 5.6875).
- **Risoluzione Bug RoPE e Theta**: Corretto position_ids collassato ($S = \text{inputs\_embeds.shape}[1]$) e sincronizzato $\text{rope\_theta} = 10^6$.
- **Mistero Oracle Risolto**: Distinzione formale tra codici GT completi (MCD = 0 dB) e residui generati dal sub-talker da $h_t$ GT (MCD = 6-9 dB, 100% intelligibile, varianza fisiologica RVQ stocastica).
- **Protocollo di Sampling per Livello**: Falsificata l'ipotesi che il sampling distrugga la voce; validato `tdecay` ($0.8 \cdot 0.75^{g-1}$) per sintesi streaming ricca e stabile.

### B. Risultati Sperimentali FitNets Stage 2

| Run ID | Architettura Adapter | Params | Step | Cos Val (Gate >=0.88) | Coupling CE Val (Gate <=5.95) | SVD Rank Pred / Target (Gate >=256) | Note Principali |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Linear(1536->2048) + LN | 3.15M | 1500 | **0.6211** (FAIL) | **5.4226** (PASS) | **1136 / 2043** (PASS) | Baseline lineare; plateau a ~0.62 da step 250. |
| **R2** | Residual MLP (Skip + 1536->3072->SiLU->2048) + LN | 14.17M | 1500 | **0.6275** (FAIL) | **5.4121** (PASS) | **1145 / 2043** (PASS) | +4.5x params produce solo +0.006 cos; capacita' non e' il collo di bottiglia. |
| **R3 Smoke** | CausalConv1d(k=3, pad 2,0, identity-init) + Residual MLP | 21.25M | 4 | **0.6835 -> 0.6953** (train) | — | — | Warm-start R2 senza regressione; gradiente conv attivo. |
| **R3 v7** | CausalConv1d(k=3, pad 2,0) + Residual MLP | 21.25M | 200 + EVAL | **0.6063** (FAIL) | **5.4817** (PASS) | **937 / 2044** (PASS) | Run completo 56m30s; CausalConv1d appresa; 72 wav generati OK. |

### C. Stabilizzazione Pipeline ed Esecuzione Kaggle
- **EVAL su CPU (ADR-006)**: Teacher mantenuto su host CPU durante l'eval per azzerare i rischi di CUDA OOM con Gemma in VRAM.
- **Riuso Istanza Gemma (ADR-007)**: Risolto il problema del memory retention di Unsloth evitando il reload della seconda copia del modello.
- **Precedenza Checkpoint Fresco (ADR-008)**: Risolto l'errore di valutazione su pesi statici del dataset di input; priorita' assoluta a `checkpoints/ht_adapter_stage2r2_final.pt`.
- **Monkey-Patch Condizionale (ADR-009)**: Iniezione a runtime di `sample_acoustic_codes` con supporto `top_k`, `top_p`, `seed` e `temperatures_per_level`.

---

## 3. Cosa NON Funziona (Asintoto Informativo)

1. **Tetto Informativo della Rappresentazione Congelata**:
   - Con il base LLM Gemma 4 E2B congelato (4-bit), la similarita' coseno di validazione resta bloccata a $\text{cos\_sim} \approx 0.60 - 0.63$, indipendentemente dalla capacita' dell'adapter (Linear 3M -> Residual MLP 14M -> CausalConv 21M).
2. **Collasso Dimensionale del Sottospazio Predetto**:
   - L'analisi spettrale SVD mostra che la predizione dell'adapter vive in un sottospazio a rango dimezzato (SVD rank effettivo **937** vs target teacher **2044**).
3. **Mancanza di Informazione Speaker/Prosodica Globale**:
   - Il teacher Qwen3-TTS estrae $h_t$ attraverso 28 layer di causal self-attention globale con condizionamento speaker; la Gemma congelata su singola finestra non dispone di sufficiente contesto per proiettare l'intero spazio dinamico.

---

## 4. Prossimi Passi Operativi (Roadmap)

### Priorita' 1: QLoRA Co-Adaptation su Gemma 4 E2B (Rompere l'Asintoto)
- **Obiettivo**: Sbloccare l'adattamento congiunto dello spazio di rappresentazione di Gemma 4 E2B verso il manifold $h_t$ di Qwen3-TTS.
- **Configurazione**:
  - QLoRA $r=16, \alpha=32$, dropout=0.05 su moduli target (`q_proj, k_proj, v_proj, o_proj`, `gate_proj, up_proj, down_proj`).
  - Learning rate differenziato: $\text{LR}_{\text{Gemma}} = 1.5 \times 10^{-5}$, $\text{LR}_{\text{Adapter}} = 1.5 \times 10^{-4}$.
  - Sub-talker Qwen3-TTS: rigorosamente congelato.
  - Target: $\text{cos\_sim} \ge 0.80$, SVD rank $\ge 1500$.
  - Durata stimata: ~2-3 ore su Kaggle T4.

### Priorita' 2: Sintesi Streaming Full-Duplex
- **Configurazione Downstream**:
  - Sampling stocastico `tdecay` ($0.8 \cdot 0.75^{g-1}$) sui 15 codebook residui.
  - Post-gain $+4\text{ dB}$ sul waveform generato da Code2Wav per compensare la leggera perdita di energia sui residui predetti.
  - Circular buffer 2-frame per zero look-ahead streaming inference.

### Priorita' 3: Contesto Temporale e Turn-Taking
- Se la co-adaptation non raggiunge il target, espandere l'hop di contesto temporale per consentire la cattura della prosodia a lungo termine.
- Integrare la turn head ausiliaria leggera (stati: WAIT, SPEAK, YIELD, INTERRUPT) per il controllo nativo del barge-in.
