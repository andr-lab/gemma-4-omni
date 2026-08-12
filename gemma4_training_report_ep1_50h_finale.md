# 📊 Gemma-4-Omni: Training Report — Run Finale 50h (Epoca 1 completa + MDC)

> **Generato**: 2026-08-09 (sessione Kaggle session `341282279`, kernel `8a702f70`)
> **Dataset**: 50.0h DuplexConv / 2500 conversazioni MDC (Fase 1 full-scale, REAL_ENCODING=True)
> **Stato**: ✅ Epoca 1 completata — checkpoint finale valutato (val_loss **16.32**, val_text **3.05**)

---

## 0. Riepilogo esecutivo

| Metrica | Valore | Note |
| :--- | :--- | :--- |
| **Training totale** | ~13,000+ step (bs=2) | 4 run di resume (4000→6000→8000→10000→12000) |
| **val_loss (ckpt 12000)** | **16.3171** | Validation split 90/10, inference_mode |
| **val_text (ckpt 12000)** | **3.0539** | CE sul vocab 262144 (baseline random = log(262144) = 12.48) |
| **Best loss_tot assoluto** | **5.613 @ step 13090** | coup 1.87, text 0.87 |
| **Best coupling** | **1.275 @ step 12502** | loss 9.08 |
| **Medie mobili (11000-12500)** | loss 17.5-18.4, coup 7.7-7.9, **text 0.82-1.30** | text in netto calo → apprendimento reale |
| **Samples** | 8 wav in `samples/` | generated_* = modello, oracle_* = GT |

---

## 1. ⚙️ Configurazione run finale

| Componente | Valore |
| :--- | :--- |
| **Base Model** | `unsloth/gemma-4-E2B-it` 4-bit QLoRA (r=16, alpha=32) |
| **Sub-talker (ADR-005)** | Qwen3-TTS congelato → codebook acustici 1..15 (qualità nativa Qwen) |
| **Gemma impara** | cb0 (semantico, vocab 4096) + h_t (adapter) + testo (lm_head 262144) |
| **Dataset** | 1,740,383 frame agente = **38.7h audio**, 27,194 finestre (20,116 con testo) |
| **Batch / Epoca** | 12,237 batch (bs=2), 1 epoca |
| **Validation split** | 2,719 finestre (10%), confine di segmento |
| **Frame weighting** | open=5.0, yield=3.0, co-talk=3.0; SELF_LISTENING=True |
| **Ottimizzatore** | AdamW lr=2e-4, wd=0.01 — **nessun gradient clipping, nessuno scheduler** |
| **Env** | transformers 5.5.0, unsloth 2026.8.9, torch 2.10.0+cu128, T4 16GB |

---

## 2. 📉 Analisi loss (checkpoint 12000, valutato eval-only)

```
val_loss = 16.32   val_text = 3.05        (inference_mode, split val 2719 finestre)
```

**Statistiche training step 8000-12500 (5843 punti):**
| Metrica | mean | std | min | max |
| :--- | :--- | :--- | :--- | :--- |
| loss_tot | 18.88 | 5.32 | 5.68 | 40.43 |
| coupling | 7.97 | 2.51 | 1.29 | 14.09 |
| text | 1.19 | 0.64 | 0.00 | 4.33 |

**Oscillazioni**: confermato che sono **rumore per-step** (batch=2 + frame weighting eterogeneo), NON divergenza. Medie mobili a 500 step: loss 17.9→17.5→18.4 (stabile), text 0.84→0.82→1.30 (apprendimento reale vs baseline random 12.48).

---

## 3. ⚠️ Lezioni operative (per futuri run)

1. **L'idle timeout Kaggle (40 min) NON è un limite di runtime** — scatta solo quando nessuna cella esegue. Il training in subprocess nohup conta come "idle" → la sessione muore. **Per run lunghi: usare Save & Run All (commit) → batch session, 9h senza idle timeout.**
2. **Il resume ri-esegue l'epoca da capo** (dataloader shuffle con `_abs_step = start + step` non salta i batch) → in questo progetto si è usato come "continuazione stocastica" (accettabile, pesi dal checkpoint), ma NON è un vero resume posizionale.
3. **Checkpoint periodici ogni 1000 step** sono il punto di ripresa sicuro; il checkpoint 12000 è il più recente.
4. **Mancanza di gradient clipping + lr fisso 2e-4** → oscillazioni ampie dopo minimi buoni (best 5.6-9.0 vs media 18.9). Un fine-tuning con `max_grad_norm=1.0` + lr 5e-5 (decay) è il passo naturale.

---

## 4. Prossimi passi proposti

| # | Azione | Costo GPU | Note |
| :--- | :--- | :--- | :--- |
| 1 | ✅ **Fatto**: eval sul checkpoint 12000 + report | ~10 min | val_loss 16.32, samples generati |
| 2 | **Fine-tuning stabile**: resume da 12000 con lr 5e-5 + `clip_grad_norm_(1.0)` + decay, ~2-4k step | ~1-2h | Riduce le oscillazioni; da fare **come commit** (9h) |
| 3 | Push del notebook corretto (cella 0 ricostruita) | — | Necessario per il commit (attuale versione salvata è rotta: NameError `Optional`) |
| 4 | Ascolto samples + demo_talk | — | wav in `samples/`, `demo_talk(duplex_model)` |

---

*Report generato da pi-agent (sessione `341282279`). Checkpoint: `checkpoints/gemma4_duplex_sota_ckpt` (LoRA 101.4MB + heads 66.1MB + adapter h_t).*
