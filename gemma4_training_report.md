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

## 5. Run completo R2 (1500 step, T4, 126m36s) — train
- cos (blocchi 100): 0.440 → **0.645** finale (media ultimi 100); min 0.003, max 0.869 (spike)
- loss media ultimi 100 = 3.031 (ultimo 3.237 vs R1 3.29); grad_norm fine run 0.2-0.5; ~4.37s/step
- R1 train: plateau 0.62 → R2 train **+0.025 (+4%)** nonostante 4.5× params → capacità NON il collo di bottiglia
- Checkpoint: `ht_adapter_stage2r2_final.pt` (54MB fp32) → dataset `andrlab/gemma4-ht-adapter-r2`

## 6. EVAL R2 (val idx 1000-1099, 100 finestre, batch 4, teacher su CPU; report `eval_r2_report.json`)
| Metrica | R1 | R2 | Gate | Esito |
| :--- | :--- | :--- | :--- | :--- |
| cos_sim | 0.6211 | **0.6275** | ≥ 0.88 | ❌ |
| mse | 3.4303 | **3.3908** | — | — |
| coupling CE | 5.4226 | **5.4121** | ≤ 5.95 | ✅ |
| rank SVD pred | 1136 | **1145** (cum 0.926) | ≥ 256 | ✅ |
| acc_pred (cb1-15 greedy) | 0.0319 | **0.0351** | — | — |
| acc_oracle | 0.0555 | **0.0555** | — | — |
- **R2 ≈ R1 su val**: +0.006 cos, +0.003 acc. Il MLP residuale 14.17M NON sposta il tetto informativo
- Wav sintesi 3 campioni (pred vs oracle): `eval_r2_pred_{a,b,c}.wav` — analisi oggettiva: energia concentrata in banda vocale (0.3-3kHz: b 94.6%, c 72.7%), **nessuna energia anomala >6kHz (0-0.3%)** → nessun segnale di "metallico"; giudizio finale all'ascolto

## 7. Verdetto finale R1+R2 (revisione agy)
- **cos 0.63 su val è il tetto informativo del canale Gemma(hidden+mel) → h_t(teacher)**, non un limite di capacità:
  +4.5× params → +0.006 cos. Iterare sull'adapter è il modo sbagliato di spendere GPU
- Cause candidate del tetto: il Gemma E2B non vede lo speaker embedding / reference audio che il teacher Qwen3-TTS usa; prompt degenerati; poche finestre (1000 train)
- **Raccomandazione agy**: giudicare sul downstream reale — l'ascolto. Se intelligibile e senza metallico → **congelare R2 e passare all'integrazione dello streaming full-duplex** (Gemma E2B + adapter + sub-talker + code2wav)
- Baseline per lo streaming: adapter R2 (cos 0.6275 val, CE 5.41, rank 1145) — migliore di R1, da congelare

## 8. Syntest v3: sampling differenziato + isolamento RVQ (kernel `gemma4-fitnets-syntest` v4, 22m47s)

### Obiettivo (test decisivo supposizioni utente)
1. Il sampling T=0.9 su **tutti** i 15 codebook residui è distruttivo → test di **sampling differenziato per livello** (greedy, t01, t03, t09, tdecay `0.8·0.75^(g-1)`)
2. **Isolamento RVQ** per individuare da quale codebook parte la corruzione

### Output validi (48 wav + report)
- 3 `gt_ref` + 30 varianti sampling (oracle/pred × 3 finestre × 5 varianti) + 15 `rvq_gtz` (GT 0..N + zeri, N=0,1,3,7,15)
- ⚠️ Blocco 4 (rvq GT+greedy) fallito per slicing errato (`_gr[:, N+1:]` su 15 col vs `_mix` 16 col) — **fixato nel codice**, riaudio opzionale se serve

### MCD gold-standard (metodo calibrato: gt vs gt=0, parlato diverso=16.66, rumore=16.93 dB ✓)
| Gruppo | MCD (dB, più basso = più simile al GT) |
| :--- | :--- |
| **Oracle** greedy/t01/t03/t09/tdecay | 6.2-9.6 / 6.4-9.4 / 6.7-9.5 / 7.1-9.1 / 6.8-8.9 |
| **Pred** greedy/t01/t03/t09/tdecay | 11.6-13.9 / 11.7-14.4 / 11.6-16.1 / 10.1-13.6 / 10.3-14.1 |
| **RVQ** N0 / N1 / N3 / N7 / N15 | 13.5-16.7 / 5.1-7.0 / 3.4-4.1 / 1.9-2.2 / 0.0 |

### Verdetto audit audio (agy, 48 wav ascoltate + spettrogrammi)
- **RVQ N0** (solo cb0): treno di click a 12.5Hz, zero formanti — non-parlato
- **RVQ N1** (cb0+cb1): **parlato 100% intelligibile** (telefonico) → MCD 5-7 dB
- **RVQ N3/N7**: naturale / quasi indistinguibile dal GT
- **Oracle** (h_t GT): intelligibile con **qualsiasi** sampling, anche greedy puro argmax
- **Pred** (h_t adapter): distrutto su **tutte** le varianti (click secchi con greedy/t01, gorgoglio soffiato con t09/tdecay)

### 🔴 Verdetto netto (agy + MCD concordi)
> **La causa del suono distrutto è la predizione sbagliata dei residui da parte del sub-talker a partire da h_t predetto dall'adapter — NON il sampling.**
> Prova inconfutabile: h_t oracolare → sub-talker OK con qualsiasi campionatore; h_t adapter (cos 0.627, coupling CE 6.07) → nessun sampling salva l'audio.

### Implicazioni
1. **La supposizione "T=0.9 distruttivo" è falsificata dai dati**: anche greedy (T=0) resta corrotto su pred. Il sampling differenziato è una leva inutile per il problema principale.
2. **Il contenuto linguistico vive nei residui 1-7** (cb0-only = click train): il sub-talker è architetturalmente essenziale; il collo di bottiglia è la qualità di h_t.
3. **Raccomandazione sampling per lo streaming full-duplex**: `tdecay` `0.8·0.75^(g-1)` (ricchezza cb1-cb3, greedy sui livelli profondi) — configurazione ottimale, non dannosa.
4. **Prossima leva vera**: migliorare h_t (conditioning speaker esplicito P3, dataset più ricco, prompt migliori) — iterare sull'adapter è già dimostrato inutile (R2 ≈ R1).
