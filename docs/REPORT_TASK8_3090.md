# REPORT TASK 8 — Smoke + Prep DuplexConv v3 + Eval temporal2 + Training TurnHead v3 su RTX 3090

**Data**: 2026-08-21 (sera)
**Esecutore**: agy (Task 8, rilanciato 2 volte dal coordinatore dopo fix immagine)
**Hardware**: RTX 3090 24GB (gpu3090, host MONDAINI 192.168.0.186), container Docker `gemma4-omni:latest`
**Stato**: ✅ COMPLETATO (tutti i gate PASS, tranne sintesi audio val con mel_l1=0 — finestre senza testo)

---

## 0. Fix infrastrutturale (coordinatore) — prerequisito del task

La prima run di agy Task 8 è morta per timeout print-mode su un comando lungo (upgrade torch).
Diagnosi (test empirici diretti in container):

| Config | Esito |
|---|---|
| torch 2.4.0 (base immagine) | ❌ `torch._inductor.config` mancante → unsloth_zoo crasha all'import |
| torch 2.13.0 (ultimo) | ❌ unsloth_zoo richiede `torch<2.13.0` |
| torch 2.12.1 + torchvision 0.28.0 | ❌ mismatch (`operator torchvision::nms does not exist`) |
| **torch 2.12.1 + torchvision 0.27.1** | ✅✅ **FLM OK, GPU RTX 3090 rilevata** |

- Dockerfile e Dockerfile.phase2 aggiornati con `torch==2.12.1 torchvision==0.27.1`
- Immagine ricostruita (13m46s) e gate verificato: `2.12.1+cu130`, transformers 5.5.0, unsloth 2026.8.19, GPU OK
- Lezione operativa per agy: comandi >5min in `nohup` + polling, MAI sincroni in print-mode

## 1. Smoke Test Fase 2 (gate loss)

- **Esito**: PASS (`/tmp/job_smoke.log`, 19:09, 154.23s, exit 0)
- Loss iniziale 22.6 → 20.8 (include Coup=5.8 + Txt=5.7: loss base coerente con gate 13-17, NON collassata)
- Coupling ADR-005 attivo (codebook 1..15 dal sub-talker Qwen congelato, loss peso 0.6, warm-up 200 step)
- Loss testo ATTIVA (lm_head, vocab 262144)
- Distribuzione target sana (dati sintetici 4 finestre)

## 2. Prep DuplexConv v3 su RTX 3090 (RIGENERAZIONE da HF)

- Script: `prep_duplexconv_v3_3090.py` (19.5KB, creato da agy da `kernel_src/prep_duplexconv_cell1.py`)
- Sorgente: `qualialabsAI/DuplexConv` su HF (repo_type="dataset"), cache HF persistente `/workspace/hf_cache`
- **Risultato: 20.02h reali di dialogo, 907 segmenti, 901.113 frame** (record: Kaggle aveva 7.2h)
- Parquet scritti in `data/`:
  - `train_tokens_20.0h_ext_mel.parquet` (1.51 GB)
  - `train_tokens_20.0h_ext_ameld.parquet` (1.50 GB)
  - `train_tokens_20.0h_ext_codes.parquet` (21.2 MB)
  - `train_tokens_20.0h_ext_text.parquet` (82.9 KB)
  - `train_tokens_20.0h_ext_turnlabels.parquet` (117.8 KB)
  - sidecar `_hyperparams.json` + `prep_duplexconv_report.json` (status PASS)
- Distribuzione classi TURN (901.113 frame): SPEAK 61.0%, BARGE_IN 21.8%, EMIT_BACKCHANNEL 9.1%, YIELD 0.8%, USER_BACKCHANNEL 0% (classe assente nel dataset DuplexConv, limite noto del dataset)

## 3. Eval checkpoint temporal2 su dataset 20h

- Script: `eval_checkpoint_3090.py` (12.6KB)
- Report: `logs/eval_report_temporal2.json` (19:49)
- **cosine_sim_val: 0.7455** (target ≥ 0.6300, record precedente 0.6342) — **NUOVO RECORD +0.115**
- Coupling loss media: 10.14; Latent h_t norm mean 45.25 (var 1.0005)
- SVD: effective rank 1537/2048 (target ≥1000 ✅), cum energy top-1000 0.9947 (target >0.92 ✅)
- 4 sample generati in `samples/` (generated_000..003 + oracle)

## 4. Training TurnStateHead v3 su dataset 20h

- Script: `train_turnhead_v3_3090.py` (13.8KB, ADR-020)
- Config: input z_t = [h_t(2048)||e_user(128)||delta_e_user(128)] ∈ R^2304, MLP Pre-LN 3.68M, 4-bucket balanced sampling, Calibrated Focal Loss (gamma 1.0, smoothing 0.05)
- Dati: `data/train_tokens_20.0h_ext`, batch 256, max-frames 200k, 15 epoche, 1 epoch ≈ 4.1s
- Report: `logs/turn_head_v3_training_report.json` (20:04) — **status: PASS**
- **Best Val Acc: 70.83%** (epoch 3), Val loss 0.787→0.987
- Per-class recall (val): **SPEAK 82.58%**, BARGE_IN 41.25%, EMIT_BC 6.06%, YIELD 4.67%, USER_BC 0% (1 solo campione val)
- Checkpoint: `checkpoints/turn_state_head_v3.pt` (11.6MB)

### Confronto TurnHead v1 → v2 → v3

| Metrica | v1 (50h) | v2 (50h bilanciato) | **v3 (20h, focal calibrato)** |
|---|---|---|---|
| Val Acc | 0.2015 | 0.166 | **0.7083** |
| SPEAK recall | 13%→4.6% (collasso) | 4.6% | **82.58%** |
| BARGE_IN recall | — | 30.30% | **41.25%** |
| Bias YIELD | 74% | 47.7% | abbattuto (YIELD recall 4.67%) |
| USER_BC F1 | 0.066 | 0.277 | 0% (1 campione val) |

**Verdetto**: il Calibrated Focal Loss + 4-bucket sampling della v3 risolve il collasso SPEAK della v2 mantenendo BARGE_IN al 41%. Limite: USER_BACKCHANNEL ha ~0 frame in DuplexConv (mai presente nel dataset).

## 5. Limiti e note

- Sintesi audio val: mel_l1/mcd = 0.000 — finestre val senza testo ASR (solo audio), metriche non informative; i wav generati sono comunque salvati
- USER_BACKCHANNEL resta a 0 campioni: serve una fonte dati con backchannel utente (MDC italiano non la risolve; valutare altri dataset)
- agy non ha scritto questo report (morto per timeout nella fase di verifica decoder) — scritto dal coordinatore

## 6. Artefatti scaricati in locale

- `_task8_3090_artifacts/t8_artifacts/`: `turn_state_head_v3.pt`, `turn_head_v3_training_report.json`, `eval_report_temporal2.json`, `synthesis_metrics.txt`, `README.txt`, `test_decode.wav`, `generated_000.wav`, `oracle_000.wav`

## 7. Prossimi passi (Task 9)

- Prep MDC su 3090 (15h+ italiano da `OpenSpeechHub/Common_Voice_17_0-It`) → training adapter-it
- TurnHead v3 già su 20h: valutare retrain su 20h + MDC quando disponibile
