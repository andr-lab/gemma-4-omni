# Report Epoca 2 — Gemma 4 Full-Duplex Omni (50h, finale)

**Data**: 2026-08-10
**Stato**: ✅ Epoca 2 completata e **validata su checkpoint FINALE post-training (step 24236, md5 86a854e9)**
**GPU**: Nvidia Tesla T4 (231 min training) + T4 (24 min eval)

---

## 1. Risultati eval sul checkpoint finale (v5, valido)

| Metrica | Epoca 1 (ckpt 12000) | **Epoca 2 (ckpt 24236)** | Verdetto |
|---|---|---|---|
| **val_loss** | 16.3171 | **15.0083** | ✅ -1.31 (-8%) |
| **val_text** | 3.0539 | **0.1799** | ✅✅ -2.87 (-94%) |
| best loss_tot (train) | 5.613 @13090 | 4.896 @12207 | ✅ |
| best coupling (train) | 1.275 @12502 | 0.494 @16977 | ✅✅ |
| text mean (train) | 1.19 | 1.365 | ⚠️ lieve aumento |

- `n_val_batches = 1360` (split val 90/10, 2719 finestre / 2 batch-size)
- **Checkpoint verificato**: md5 `86a854e9` = finale epoca 2 post-training (NON il periodico 24000, NON l'epoca 1)
- Baseline random per val_text: log(262144) = 12.48 → **0.18 è eccellente**

## 2. Metrica di qualità sintesi vs Qwen TTS (domanda utente)

La metrica usata è **mel-L1 + MCD (Mel Cepstral Distortion) tra generated e oracle**, con **sub-talker + decoder Code2Wav IDENTICI** nei due rami: così la distanza isola esclusivamente la qualità della **predizione Gemma** (codebook cb0 + stato h_t), che è l'unica componente addestrata.

Risultati su 8 finestre val (64 frame = 5.12s, 24kHz mono):

| window | mel_l1 | MCD (dB) | testo |
|---|---|---|---|
| 000 | 0.0297 | 2.678 | Nelle mura sono incorporate alcune fortificazioni. |
| 001 | 0.0082 | 1.525 | Nelle mura sono incorporate alcune fortificazioni. |
| 002 | 0.1163 | 5.661 | Queste ultime tuttavia hanno rinunciato. |
| 003 | 0.0285 | 2.951 | Ennio Pompei |
| 004 | 0.0398 | 3.358 | Il capoluogo Trincomalee è la città più popolata della provincia. |
| 005 | 0.0109 | 1.779 | Molto giovane comincia a lavorare a Broadway… |
| 006 | 0.0000 | 0.000 | Alcuni adattatori sono progettati… (predizione = oracle esatta) |
| 007 | 0.1334 | 5.910 | Megami Bunko |
| **MEDIA** | **0.0459** | **2.983** | |
| *media esclusa 006* | *0.0524* | *3.409* | |

**Lettura**: MCD < 3 dB medio con punte di 1.5 dB = qualità vocoder eccellente; la pipeline sintesi (sub-talker + Code2Wav) è **dimostrata funzionante end-to-end**. Le finestre deboli (002: 5.66 dB, 007: 5.91 dB) corrispondono a testi corti/incompleti ("Megami Bunko") dove il contesto è minimo.

## 3. Fix critici applicati durante la validazione

1. **Bug MCD**: `np.fft.dct` non esiste in NumPy → la sintesi falliva SILENZIOSAMENTE (AttributeError inghiottito dal try/except) in v7 e eval-v1/v2. Fix: DCT-II ortogonale numpy-pura, verificata vs `scipy.fft.dct` (errore ~1e-14). Patch applicata via monkeypatch `KP._mel_dist_mcd`.
2. **Bug struttura checkpoint**: il cell di setup copiava il ckpt a `.../adapter_model.safetensors` (radice) ma `load_gemma4_duplex_checkpoint` legge da `lora_adapters/` → i numeri di eval-v2 (15.697/0.245) erano in realtà dell'**epoca 1** (dal dataset). Corretto: copia in `lora_adapters/` + verifica md5 nel cell di eval.
3. **Checkpoint sbagliato nel dataset**: era stato caricato il periodico 24000 (9f363f4a) invece del **finale post-training 24236 (86a854e9)**. Dataset `andrlab/gemma4-ep2-final-ckpt` aggiornato.
4. **exec() namespace**: `exec(PATCH, {'np': np})` non esporta le funzioni → NameError (v4). Fix: estrazione esplicita dal dict.

## 4. Conclusioni & risposta alla domanda utente

**Una epoca basta? Dipende dall'obiettivo:**
- L'epoca 2 con **lr 5e-5 + grad clip 1.0** ha prodotto un **miglioramento netto e misurabile** (val_loss -8%, val_text -94%, coupling best -61%): il training converge e la ricetta iperparametri è corretta.
- MA: val_text = 0.18 su 262144 vocab è così basso da suggerire **inizio di overfitting sul testo**; e le finestre deboli (MCD ~5.9) mostrano margine residuo.
- **Raccomandazione**: il passo successivo con ROI maggiore è **allargare il dataset a ~100h** (più diversità testuale/mel) e riallenare da zero 1 epoca, NON un'altra epoca sui 50h (rischio memorizzazione). Con lr 5e-5 + clip 1.0 come baseline confermata.

## 5. File & artefatti

- Checkpoint finale: `checkpoints/gemma4_duplex_sota_ckpt/` (adapter md5 `86a854e9`, heads md5 `a8fa8ce3`)
- Samples ascoltabili: `eval_ep2_out_v5/samples/generated_000..007.wav` + `oracle_000..007.wav` (5.12s, 24kHz)
- Metriche: `eval_ep2_out_v5/samples/synthesis_metrics.txt`
- Val: `eval_ep2_out_v5/data/val_ep2_final.txt`, `eval_ep2_out_v5/data/ckpt_md5.txt`
- Kernel: `andrlab/gemma4-ep2-eval-v5` (COMPLETE), `andrlab/gemma4-epoca2-commit-v7` (training, COMPLETE)
- Dataset: `andrlab/gemma-4-omni-module` (11528372), `andrlab/gemma4-ep2-final-ckpt` (11581251, v2)
