# 📊 Gemma-4-Omni: Diagnostic Training & Performance Report

> **Generato automaticamente**: Post-Training Run Analysis  
> **Stato Pipeline**: Completato con successo su `cuda`

---

## 1. ⚙️ Architettura & Configurazione Iperparametri
| Iperparametro / Componente | Valore Configurato | Note Architetturali |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/gemma-4-E2B-it` | Backbone Gemma 4 (Nativamente Multimodale Encoder-Free) |
| **Quantizzazione LLM** | 4-bit QAT (W4A16 GGUF) | Vocabolario Primario Congelato (No `resize_token_embeddings`) |
| **QLoRA Setup** | `r=16`, `lora_alpha=32`, `dropout=0.05` | Target: `q, k, v, o_proj` + `gate, up, down_proj` |
| **Parallel Audio Heads** | 15 Linear Layers (`float32`) | Predicts 15 RVQ Codebooks (Qwen3-TTS 12.5 Hz / 80ms) |
| **Upsampler Layer** | `nn.Conv1d(3072, 3072, kernel=3)` | Allineamento Frequenza Temporale (LLaMA-Omni style) |
| **Codebook Delay Pattern** | Attivo (Shift $k$ posizioni) | Stabilizzazione Gerarchica Acustica (Mini-Omni style) |
| **Optimizer & LR** | `AdamW 8-bit`, `lr=2e-4`, `wd=0.01` | Linear Warmup & Decay |
| **Silence Loss Weight** | `SILENCE_LOSS_WEIGHT = 0.2` | Cross-Entropy Pesata per Class Imbalance |
| **Data Subsets** | DuplexConv: `50.0h` | MDC ITA: `2500 samples` |

---

## 2. 🖥️ Telemetria Hardware & Performance GPU
- **Device di Esecuzione**: `cuda`
- **Peak VRAM Allocated**: `0.01 GB` (Utilizzo Quota GPU: `0.1%` di 16GB)
- **Tempo Totale Addestramento**: `1017.65 secondi` (`16.96 minuti`)
- **Throughput Medio**: `0.85 steps/secondo`

---

## 3. 📈 Analisi della Convergenza & Curva di Loss
- **Loss Iniziale (Step 1)**: `7.7837`
- **Loss Finale**: `6.8739`
- **Loss Minima Raggiunta**: `6.2018`
- **Loss Massima**: `7.7837`
- **Loss Media complessiva**: `6.8421`
- **Percentuale di Riduzione Loss**: `11.69%`
- **Perplessità Acustica (Acoustic PPL)**: `966.76`

### Estratto della Loss Step-by-Step:
| Step | Loss Computata | Perplessità Equivalente |
| :--- | :--- | :--- |
| 1 | 7.7837 | 2401.05 |
| 2 | 7.2943 | 1471.84 |
| 3 | 7.2160 | 1361.06 |
| 4 | 7.1191 | 1235.34 |
| 5 | 7.0629 | 1167.77 |
| 6 | 6.9506 | 1043.74 |
| 7 | 6.9980 | 1094.47 |
| 8 | 6.6132 | 744.83 |
| 9 | 6.7928 | 891.42 |
| 10 | 6.7654 | 867.28 |
| ... | ... | ... |
| 867 | 6.8739 | 966.76 |

---

## 🎙️ 4. Scomposizione della Loss per Codebook & Turn-Taking
- **Loss Semantica (Codebook 0 - Contenuto)**: `6.0491`
- **Loss Acustica (Codebook 1-14 - Prosodia/Timbro)**: `7.1489`
- **Predizione Silenzio (`<|model_silence|>)**: `98.5%` dei frame totali
- **Predizione Parlato Attivo (Agente)**: `1.5%` dei frame totali

---

## 💡 5. Raccomandazioni ed Azioni correttive per il Prossimo Run
- **[ALLERTA LOSS ELEVATA]**: La loss finale (`6.8739`) supera la soglia di stabilità (2.0). **Azione**: Aumentare `learning_rate` da `2e-4` a `3e-4` oppure incrementare `lora_alpha` da `32` a `64` per dare maggiore capacità espressiva agli adattatori.
- **[SQUILIBRIO SILENZIO]**: Il modello genera oltre l'85% di token di silenzio. **Azione**: Ridurre `SILENCE_LOSS_WEIGHT` da `0.2` a `0.1` per penalizzare la scorciatoia del silenzio.
- **VRAM Allocation**: Prossimi run possono aumentare `batch_size` a 4 o 8 se su A10G/A100.
