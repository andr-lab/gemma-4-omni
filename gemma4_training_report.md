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

---

# 🔬 FitNets Stage 2 — Verdetto R1 e Run R2 (append, 18/08)

## 1. Run completo v24 (adapter lineare R1, 1500 step, train 1000 finestre idx 0-999)
- loss 6.38 → 3.29 | latent(MSE) 4.19 → 2.11 | coupling CE 10.94 → 5.89 | cos_sim 0.005 → 0.62
- Plateau cos ~0.586-0.618 da step 250 (spike 0.778 a step 1400 = rumore batch)
- Freeze verificato su tutti i 1500 step (base0 True, sub0 True); ~3.93s/step
- Checkpoint: `ht_adapter_stage2_final.pt` + periodici ogni 250 → dataset `andrlab/gemma4-ht-adapter-r1`

## 2. EVAL R1 (val split pulito, idx 1000-1099, mai viste; batch 4; teacher mai in GPU)
| Metrica | Valore val | Gate | Esito |
| :--- | :--- | :--- | :--- |
| cos_sim | **0.6211** | ≥ 0.88 | ❌ |
| coupling CE | **5.4226** | ≤ 5.95 | ✅ |
| rank SVD effettivo | **1136** (cum 0.959) | ≥ 256 | ✅ |
| rank target | 2043 (cum 1.000) | — | — |
| mse | 3.4303 | — | — |
- Acc greedy cb1-15: **pred 0.032** vs **oracolo 0.056** (tetto della catena) → il predetto raggiunge ~57% del tetto

## 3. Verdetto (revisione agy, contesto fresco)
- **R1 = baseline funzionale temporanea** (CE e rank sotto gate; utile per testare lo streaming full-duplex)
- **R1 NON per produzione audio**: angolo medio 51.6° tra h_t predetto e target → timbro degradato
- Plateau train=val (0.6200 vs 0.6211) → **limite di capacita' dell'adapter lineare 3.15M**, non mancanza di step
- **R2 necessario** per cos ≥ 0.85: MLP residuale + LR schedule (loss gia' nella forma target: cos 1.0 + 0.5 MSE + 0.2 CE)

## 4. R2 (implementato da agy, validato con round-trip)
- **ResidualAdapter**: LayerNorm( MLP(h) + skip(h) ), Linear 1536→3072, SiLU, 3072→2048, skip 1536→2048, LayerNorm(2048) → **14.17M params** fp32
- Factory `build_ht_adapter(hid, variant)` (variant=1 R1 invariato, variant=2 R2) a livello modulo
- LR schedule: **cosine warmup** (LambdaLR): 3e-4, warmup 100 step lineare, decay coseno → 1.5e-5 a step 1500
- Checkpoint separati `ht_adapter_stage2r2_*.pt`, report `stage2r2_report.json`
- **Smoke R2 PASS** (4 step, 20m13s): freeze OK (base0/sub0 True), warmup LR attivo (3e-6→1.2e-5), ~4.3s/step → run completo 1500 step ~2h T4
