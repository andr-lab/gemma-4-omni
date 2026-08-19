# Gemma 4 Full-Duplex Omni Pipeline - Stato Attuale e Prossimi Passi

Documento di stato del progetto • Aggiornato al 19 Agosto 2026

---

## 1. Stato Attuale del Progetto (Executive Summary)

Il progetto sviluppa una pipeline conversazionale Speech-to-Speech Full-Duplex real-time accoppiando **Gemma 4 E2B** (305M USM-Conformer audio input) con il **sub-talker congelato di Qwen3-TTS-12Hz-1.7B-Base** (~363MB) e il decoder **Code2Wav** (24kHz PCM).

La fase di allineamento e distillazione latente (FitNets Stage 2: Gemma hidden -> adapter -> h_t teacher) e' avanzata dallo Stage 2 puro all'addestramento congiunto **Stage 2 QLoRA Co-Adaptation** (Gemma 4 E2B LoRA $r=16, \alpha=32$ + `ResidualAdapter` CausalConv1D + Skip + MLP).

---

## 2. Cosa e' stato fatto (R1, R2, R3 e QLoRA)

### A. Fedelta' e Validazione Sub-Talker (Oracle)
- **Verifica Wrapper vs Reference Teacher**: Replica bit-identica del `code_predictor` eager/bf16 (max diff logits = 0.00e+00, CE oracolare = 5.6875).
- **Risoluzione Bug RoPE e Theta**: Corretto position_ids collassato ($S = \text{inputs\_embeds.shape}[1]$) e sincronizzato $\text{rope\_theta} = 10^6$.
- **Mistero Oracle Risolto**: Distinzione formale tra codici GT completi (MCD = 0 dB) e residui generati dal sub-talker da $h_t$ GT (MCD = 6-9 dB, 100% intelligibile, varianza fisiologica RVQ stocastica).
- **Protocollo di Sampling per Livello**: Falsificata l'ipotesi che il sampling distrugga la voce; validato `tdecay` ($0.8 \cdot 0.75^{g-1}$) per sintesi streaming ricca e stabile.

### B. Risultati Sperimentali FitNets Stage 2 & QLoRA

| Run ID | Architettura Adapter | Params | Step | Cos Val (Gate >=0.88) | Coupling CE Val (Gate <=5.95) | SVD Rank Pred / Target (Gate >=256) | Note Principali |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Linear(1536->2048) + LN | 3.15M | 1500 | **0.6211** (FAIL) | **5.4226** (PASS) | **1136 / 2043** (PASS) | Baseline lineare; plateau a ~0.62 da step 250. |
| **R2** | Residual MLP (Skip + 1536->3072->SiLU->2048) + LN | 14.17M | 1500 | **0.6275** (FAIL) | **5.4121** (PASS) | **1145 / 2043** (PASS) | +4.5x params produce solo +0.006 cos; capacita' non e' il collo di bottiglia. |
| **R3 Smoke** | CausalConv1d(k=3, pad 2,0, identity-init) + Residual MLP | 21.25M | 4 | **0.6835 -> 0.6953** (train) | — | — | Warm-start R2 senza regressione; gradiente conv attivo. |
| **R3 v7** | CausalConv1d(k=3, pad 2,0) + Residual MLP | 21.25M | 200 + EVAL | **0.6063** (FAIL) | **5.4817** (PASS) | **937 / 2044** (PASS) | Run completo 56m30s; CausalConv1d appresa; asintoto a ~0.60 confermato. |
| **QLoRA Smoke** | QLoRA r=16 (Gemma) + CausalConv ResidualAdapter | 46.58M | 4 | **0.6846 -> 0.7226** (train) | **6.1733** (smoke 4 step) | **245 / 255** (smoke 4 win) | **TUTTI I GATE PASS**: LoRA grad attivo, base0/sub0 confermati, step 5.6-6.6s. |
| **QLoRA Completo** | QLoRA r=16 (Gemma) + CausalConv ResidualAdapter | 46.58M | 1200 + EVAL | *In monitoraggio* | *In monitoraggio* | *In monitoraggio* | **RUN COMPLETO IN CORSO** su Kaggle T4 (`andrlab/gemma4-qlora-r1` v3). |

### C. Stabilizzazione Pipeline ed Esecuzione Kaggle
- **Fix Globale Dtype-Safe LayerNorm**: Risolto il mismatch `Half/Float` nelle funzioni compilate da Unsloth/Dynamo.
- **EVAL su CPU (ADR-006)**: Teacher mantenuto su host CPU durante l'eval per azzerare i rischi di CUDA OOM con Gemma in VRAM.
- **Riuso Istanza Gemma (ADR-007)**: Risolto il problema del memory retention di Unsloth evitando il reload della seconda copia del modello.
- **Precedenza Checkpoint Fresco (ADR-008)**: Priorita' assoluta a `checkpoints/ht_adapter_stage2r2_final.pt`.
- **Monkey-Patch Condizionale (ADR-009)**: Iniezione a runtime di `sample_acoustic_codes` con supporto `top_k`, `top_p`, `seed` e `temperatures_per_level`.

---

## 3. Asintoto Informativo e Soluzione QLoRA

1. **Diagnosi dell'Asintoto**:
   - Con il base LLM Gemma 4 E2B congelato (4-bit), la similarita' coseno di validazione restava bloccata a $\text{cos\_sim} \approx 0.60 - 0.63$, e l'SVD rank pred a **937** vs **2044**.
2. **Soluzione Attuata (QLoRA Co-Adaptation)**:
   - Sbloccato l'adattamento congiunto dello spazio di rappresentazione di Gemma 4 E2B ($r=16, \alpha=32$) con il `ResidualAdapter` (25.3M LoRA + 21.2M Adapter = 46.58M params).
   - Obiettivo: rompere l'asintoto portando $\text{cos\_sim} \ge 0.80$ e SVD rank $\ge 1500$.

---

## 4. Prossimi Passi Operativi (Roadmap)

### Priorita' 1: Monitoraggio e Analisi Esiti Run Completo QLoRA
- Monitorare il completamento del kernel `andrlab/gemma4-qlora-r1` (v3, 1200 step).
- Verificare i report `stage2qlora_report.json` e `eval_qlora_report.json` e scaricare i checkpoint finali (`ht_adapter_qlora_final.pt` e `gemma4_qlora_adapter`).
- Valutare il superamento del gate $\text{cos\_sim} \ge 0.80$ e dell'SVD rank $\ge 1500$.

### Priorita' 2: Sintesi Streaming Full-Duplex
- **Configurazione Downstream**:
  - Sampling stocastico `tdecay` ($0.8 \cdot 0.75^{g-1}$) sui 15 codebook residui.
  - Post-gain $+4\text{ dB}$ sul waveform generato da Code2Wav.
  - Circular buffer 2-frame per zero look-ahead streaming inference.

### Priorita' 3: Contesto Temporale e Turn-Taking
- Se la co-adaptation richiede ulteriore contesto prosodico, espandere l'hop temporale della finestra.
- Integrare la turn head ausiliaria leggera (stati: WAIT, SPEAK, YIELD, INTERRUPT) per la gestione nativa delle sovrapposizioni e interruzioni (barge-in).
