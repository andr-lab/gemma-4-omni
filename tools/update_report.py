# -*- coding: utf-8 -*-
with open('gemma4_training_report.md', 'r', encoding='utf-8') as f:
    text = f.read()

target = '## 11. Tabella Riassuntiva dei Run (R1, R2, R3 Smoke, R3 Completo v7)'
idx = text.find(target)
if idx != -1:
    new_sec = """### 10.7 Run Stage 2 QLoRA Co-Adaptation (kernel andrlab/gemma4-qlora-r1) - 2026-08-19

**Motivazione architetturale**: I tre run precedenti (R1, R2, R3) hanno dimostrato in modo inequivocabile l'esistenza di un *tetto informativo* (cos_sim ~ 0.60-0.63, SVD rank pred 937 vs target 2044) dovuto alla rappresentazione congelata di Gemma 4 E2B. Per rompere l'asintoto, e' stata introdotta la co-adaptation congiunta di Gemma 4 E2B via QLoRA assieme al `ResidualAdapter`.

**Configurazione Architetturale e Iperparametri**:
- **Gemma 4 E2B 4-bit**: QLoRA r=16, alpha=32, dropout=0.0, target modules `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (25.3M parametri trainable) con LR_Gemma = 1.5e-5.
- **ResidualAdapter**: `CausalConv1d` (k=3, identity-init, pad 2,0) + Skip + MLP (1536 -> 3072 -> SiLU -> 2048) con warm-start dai pesi di R2/R3 (21.2M parametri trainable) con LR_Adapter = 1.5e-4.
- **Sub-Talker Qwen3-TTS**: Congelato in `bfloat16` (181.4M parametri) per calcolo coupling loss (lambda = 0.2).
- **Teacher Qwen3-TTS**: Spostato su host CPU subito dopo l'estrazione target h_t (zero occupazione GPU).
- **Ottimizzatore**: AdamW differenziale con cosine schedule e warmup (100 step nel run completo).
- **Totale parametri trainable**: **46,582,784** (25.3M LoRA + 21.2M Adapter).

**Esito SMOKE Test (v2, 4 batch, 16 finestre train + 4 val)**:
- **TUTTI I GATE PASS**:
  - `status: "PASS"` in `stage2qlora_report.json`
  - `cos_sim` iniziale: **0.6846** (> 0.5)
  - Trend loss: **2.9014 -> 2.8206 -> 2.7529 -> 2.7221** (costantemente decrescente)
  - Trend `cos_sim`: **0.6846 -> 0.7031 -> 0.7162 -> 0.7226**
  - Gradienti verificati: `lora_grad_active: True`, `base_grad_zero: True`, `sub_grad_zero: True` su tutti gli step.
  - Velocita' a regime: **5.65s - 6.63s per batch effettivo 16** (8 bs x 2 grad accum).
  - Fix critico `_dtype_safe_torch_layer_norm`: risolto il mismatch `Half/Float` nelle funzioni compilate da Unsloth/Dynamo.
  - Checkpoint e LoRA salvati correttamente e ricaricati in EVAL senza memory retention o OOM.

**Run Completo (v3, 1200 step, 1000 finestre train + 100 val)**:
- Pushed su Kaggle (`andrlab/gemma4-qlora-r1`), monitoraggio attivo in background (timeout 200m).
- Tempo stimato: ~2.0-2.5 ore.

---

## 11. Tabella Riassuntiva dei Run (R1, R2, R3, QLoRA)

| Run ID / Setup | Architettura Adapter | Params Trainable | Step & Dataset | Tempo & Hardware | Cos Val (Gate >=0.88) | Coupling CE Val (Gate <=5.95) | SVD Rank Pred / Target (Gate >=256) | Kernel Kaggle | Verdetto / Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1 (Linear)** | Linear(1536->2048) + LN | 3.15M | 1500 step (1000 finestre train) | ~100m, T4 (16GB) | **0.6211** (FAIL) | **5.4226** (PASS) | **1136 / 2043** (PASS) | `gemma4-fitnets-eval` (v24) | Baseline funzionale; plateau cos a 0.62; limite capacita adapter lineare. |
| **R2 (Residual MLP)** | Skip + MLP(1536->3072->SiLU->2048) + LN | 14.17M (+4.5x params) | 1500 step (1000 finestre train) | 126m 36s, T4 | **0.6275** (FAIL, +0.006) | **5.4121** (PASS) | **1145 / 2043** (PASS) | `gemma4-fitnets-eval-r2` | +4.5x params produce solo +0.006 cos; capacita non e il collo di bottiglia; audio senza metallico. |
| **R3 Smoke** | CausalConv1d(k=3, pad 2,0) + Residual MLP | 21.25M (identity-init) | 4 step (16 finestre train) | 14m 37s, T4 | **0.6835 -> 0.6953** (train) | - (smoke) | - | `gemma4-fitnets-syntest-r3` (v2) | Warm-start R2 senza regressione; CausalConv1d apprende (gradiente attivo); pipeline stabilizzata. |
| **R3 Completo v7** | CausalConv1d(k=3, pad 2,0) + Residual MLP | 21.25M | 200 step (warm-start R2) + EVAL2 (100 val) | 56m 30s, T4 | **0.6063** (FAIL, asintoto) | **5.4817** (PASS) | **937 / 2044** (PASS) | `gemma4-fitnets-syntest-r3` (v7) | CausalConv1d appresa; asintoto informativo a ~0.60-0.63 confermato; rank 937; synth 72 wav OK; next: QLoRA co-adaptation. |
| **QLoRA Smoke** | QLoRA r=16 (Gemma) + CausalConv ResidualAdapter | 46.58M (25.3M LoRA + 21.2M Adapt) | 4 step (16 finestre train + 4 val) | 17m 46s, T4 | **0.6846 -> 0.7226** (train) | **6.1733** (smoke 4 step) | **245 / 255** (smoke 4 win) | `gemma4-qlora-r1` (v2) | **TUTTI I GATE PASS**: LoRA grad attivo, base0/sub0 confermati, no regressione, step a regime 5.6-6.6s. |
| **QLoRA Completo** | QLoRA r=16 (Gemma) + CausalConv ResidualAdapter | 46.58M | 1200 step (1000 train + 100 val) | In corso (~2.5h, T4) | *In monitoraggio* | *In monitoraggio* | *In monitoraggio* | `gemma4-qlora-r1` (v3) | Run in esecuzione per rompere l'asintoto informativo (target cos >= 0.80, rank >= 1500). |
"""
    updated = text[:idx] + new_sec
    with open('gemma4_training_report.md', 'w', encoding='utf-8') as f:
        f.write(updated)
    print('gemma4_training_report.md updated successfully!')
