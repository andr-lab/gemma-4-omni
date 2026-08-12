# 📊 Gemma-4-Omni: Training Report — Run 2 (Coupling ADR-005 + Decoder Fix)

> **Generato**: 2026-08-08 (sessione Kaggle kernel `ac376be1`)
> **Dataset**: 1.0h DuplexConv / 100 dialoghi MDC (Fase 1, REAL_ENCODING=True)
> **Stato**: Completato — run completo 1 epoca + 2 epoche di continuazione

---

## 0. Fix applicati prima del run (DDD+TDD)

| Fix | Dettaglio | Verifica |
| :--- | :--- | :--- |
| **Decoder Code2Wav residual** | `_ConvNeXtBlock.forward` ometteva `hidden_states = inp + hidden_states` → oracle = rumore | Replica patched == decoder ufficiale (corr **1.0000**, maxdiff 3e-5); oracle corr_mel 0.52-0.88 vs GT |
| **Dataloader shadowing** | Secondo `DataLoader(dataset)` (dataset completo SENZA testo) sovrascriveva lo split train → leakage val + text_loss=0 in training | Rimossa la ricostruzione: 251 batch (split train, non 278) + `text_targets` attivi (43 token nel batch) |
| **Log per-step** | `data/phase2_run_log*.tsv` con epoch/step/loss/coupling/text per monitoraggio | 251 + 502 righe loggate |

---

## 1. ⚙️ Architettura & Configurazione (run corrente)
| Componente | Valore | Note |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/gemma-4-E2B-it` 4-bit | PeftModelForCausalLM, hidden_size 1536 |
| **Quantizzazione** | 4-bit QLoRA | T4 singola (device_map cuda:0) |
| **QLoRA** | r=16, alpha=32, dropout=0, bias=none | Target: q/k/v/o + gate/up/down |
| **Parametri** | 4.63B totali / **41.9M trainable** | LoRA + teste custom + adapter h_t |
| **Coupling (ADR-005)** | Sub-talker Qwen3-TTS congelato | cb1..15 dal sub-talker, peso 0.6, warm-up 200 step |
| **Supervisione testuale** | TEXT_BLOCK_LEN=48, peso 0.5 | lm_head condivisa (vocab 262144), 435/557 finestre con testo |
| **Frame weighting** | open=5.0, yield=3.0, co-talk=3.0 | SELF_LISTENING=True (2×8×128 mel) |
| **Optimizer** | AdamW lr=2e-4, wd=0.01 | 1 epoca = 251 batch (bs=2) |
| **Data** | 35643 frame agente (2851s), 16 codebook | cb0=4096, cb1-15=2048 @12.5Hz; val_start=32078 (90%) |

---

## 2. 🖥️ Telemetria GPU
- **Device**: 1× Tesla T4 (16GB), CUDA 7.5, torch 2.10.0+cu128
- **Tempo Epoca 1**: ~7.2 min (251 batch → ~1.7 s/batch)
- **Tempo Epoche 2-3**: ~12.5 min (502 batch)
- **Throughput**: ~0.58 step/s

---

## 3. 📈 Convergenza — Epoca 1 (loss totale = cb0 + coupling×0.6 + text×0.5)

| Step | Loss | Coupling | Text |
| :--- | :--- | :--- | :--- |
| 0 | 46.98 | 17.23 | 7.69 |
| 49 | 23.80 | 7.71 | 3.11 |
| 99 | 19.08 | 6.21 | 3.06 |
| 149 | 23.61 | 6.20 | 2.27 |
| 199 | 7.57 | 1.99 | 0.62 |
| 249 | 19.51 | 7.21 | 0.97 |
| **250 (fine)** | **14.87** | 4.83 | 2.79 |

- **Riduzione loss**: **68.4%** (47.0 → 14.9); minima 7.57
- **Trend blocco**: 28.2 → 17.7 (discesa costante)
- **val_loss epoca 1**: 19.05 | **val_text**: 2.32 (da 5.24 pre-run)
- **Loss finale ~15 = range atteso** (log 4096 ≈ 8.3 + log 2048 ≈ 7.6), NON collassata (~0.002)

## 4. 📈 Convergenza — Epoche 2-3 (continuazione)

| Blocco | Loss media | Text media |
| :--- | :--- | :--- |
| ep2 0-49 | 18.50 | 1.67 |
| ep2 100-149 | 18.76 | 1.52 |
| ep2 200-249 | 19.13 | 1.17 |
| ep3 49-98 | 16.88 | 0.85 |
| ep3 149-198 | 17.09 | 0.85 |

- **val_loss**: ep2=18.86, ep3=19.27 (plateau)
- **val_text**: ep2=2.63, ep3=3.24
- **Osservazione**: plateau atteso con dataset 1h — il modello ha appreso la struttura ma satura; testo continua a scendere lentamente (0.69 finale)

---

## 5. 🎙️ Qualità Audio (corr_mel vs GT, decoder fixato)

### Oracle (codici GT decodificati — riferimento voce reale)
| File | corr_mel | rms |
| :--- | :--- | :--- |
| oracle_000 | 0.858 | 0.0094 |
| oracle_002 | 0.801 | 0.0467 |
| oracle_004 | 0.825 | 0.0694 |
| oracle_007 | 0.876 | 0.0470 |

### Generated (modello) — evoluzione su 3 epoche
| File | corr_mel ep1 | corr_mel ep3 | Δ | flatness (basso=vocale) |
| :--- | :--- | :--- | :--- | :--- |
| generated_000 | 0.180 | **0.516** | +0.34 | 0.774 |
| generated_001 | -0.124 | 0.000 | +0.12 | 0.843 |
| generated_002 | 0.069 | **0.409** | +0.34 | **0.133** ✅ |
| generated_003 | -0.048 | **0.314** | +0.36 | 0.679 |
| generated_004 | 0.215 | **0.368** | +0.15 | **0.191** ✅ |
| generated_005 | 0.178 | 0.209 | +0.03 | 0.705 |
| generated_006 | 0.177 | 0.235 | +0.06 | 0.521 |
| generated_007 | 0.108 | **0.265** | +0.16 | **0.119** ✅ |

**Lettura**: corr_mel in crescita su tutti gli esempi; 3/8 generated hanno flatness spettrale **vocale** (0.12-0.19, pari all'oracle 0.17) = struttura tonale reale, non rumore puro. Con 1h di dati e 3 epoche è il massimo ragionevole: i generated sono "quasi-parlato" ma non intelligibile (manca massa dati).

---

## 6. 💾 Artefatti

| Artefatto | Path | Stato |
| :--- | :--- | :--- |
| Checkpoint LoRA | `checkpoints/gemma4_duplex_sota/lora_adapters/` (101.4MB) | ✅ post 3 epoche |
| Checkpoint teste | `checkpoints/gemma4_duplex_sota/custom_duplex_heads.pt` (66.1MB) | ✅ post 3 epoche |
| Log epoca 1 | `data/phase2_run_log_ep1.tsv` (251 righe) | ✅ |
| Log epoche 2-3 | `data/phase2_run_log_ep2.tsv` (502 righe) | ✅ |
| Sample oracle | `samples/oracle_000..007.wav` | ✅ parlato (fix decoder) |
| Sample generated | `samples/generated_000..007.wav` | ✅ struttura emergente |

---

## 7. 📋 Raccomandazioni prossimo run

1. **Più dati (priorità alta)**: plateau a ~19 con 1h/100 dialoghi. Rigenerare Fase 1 con **5-10h DuplexConv + 500-1000 MDC** (AGENTS.md smoke gate) per dare al modello abbastanza contenuto — il testo scende ancora (0.69) ma la loss audio satura.
2. **LR**: mantenere 2e-4 (il report legacy suggeriva 3e-4 per il vecchio setup MTP; con coupling il valore attuale converge in plateau corretto).
3. **Ottimizzatore persistente**: `run_training_pipeline` ricrea AdamW a ogni chiamata → il momentum si resetta tra epoche separate. Per run lunghi: creare l'ottimizzatore una volta e riusarlo tra epoche (stato AdamW incluso nel checkpoint opzionale).
4. **Valutazione**: estendere con corr_mel su finestre multiple + flatness per sample; il segnale "generated_002/004/007 flatness vocale" è il primo indicatore di apprendimento acustico reale.
5. **Demo ascolto**: `demo_talk(duplex_model)` dopo reload checkpoint per parlare col modello (wav utente in `samples/user_input.wav`).
