# -*- coding: utf-8 -*-
"""RESUME training Gemma 4 Full-Duplex (Fase 2) dopo il crash del kernel.

- Ambiente: subprocess fresco con transformers 5.5.0 + unsloth (il kernel principale
  ha 5.0.0 in memoria, non compatibile con unsloth appena installato).
- Riprende dal checkpoint periodico checkpoints/gemma4_duplex_sota_ckpt (step 12000 —
  l'ultimo checkpoint periodico salvato; il run precedente era arrivato a step 12517
  quando la sessione è stata ruotata da Kaggle).
- Log per-step su data/phase2_run_log_50h_mdc_ep1_resume.tsv (step ASSOLUTI).
- Checkpoint periodico ogni 1000 batch (step assoluti) + log su phase2_ckpt_log_50h.txt.
- NOTA: non usa il websocket; robusto contro disconnessioni dell'agente.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# import unsloth PRIMA di transformers (best practice unsloth)
import unsloth  # noqa: F401  (applica le patch)

# Il dataset andrlab/gemma-4-omni-module contiene qwen_subtalker.py (modulo) e
# extract_qwen_subtalker.py. Deve stare PRIMA di /kaggle/working nel sys.path,
# altrimenti la dir /kaggle/working/qwen_subtalker/ (pesi) fa da namespace package
# vuoto e l'import `from qwen_subtalker import load_qwen_subtalker` fallisce.
sys.path.insert(0, "/kaggle/input/datasets/andrlab/gemma-4-omni-module")
sys.path.insert(0, "/kaggle/working")
import kaggle_pipeline as KP  # noqa: E402

print("=== RESUME TRAINING (subprocess) ===")
print("transformers:", __import__("transformers").__version__)
print("big_disk:", KP.big_disk_base())
sys.stdout.flush()

# sanity: qwen_subtalker importable (con la funzione giusta)
import qwen_subtalker as _qst
assert hasattr(_qst, "load_qwen_subtalker"), "load_qwen_subtalker mancante!"
print("qwen_subtalker module OK:", _qst.__file__)
sys.stdout.flush()

t0 = time.time()

# 1) Carica il sistema completo (backbone Unsloth + subtalker congelato + heads custom)
#    NB: load_gemma4_duplex_system restituisce la TUPLA (duplex_model, tokenizer)
model, _tokenizer = KP.load_gemma4_duplex_system()
print(f"[Resume] Sistema caricato in {time.time()-t0:.1f}s")
sys.stdout.flush()

# 2) Configura env per run_training_pipeline: resume dal checkpoint periodico
#    (ultimo checkpoint disponibile: step 12000 — aggiornato il 2026-08-09 dopo la
#    rotazione della sessione; il run precedente era arrivato a step 12517)
os.environ["PHASE2_RESUME_CKPT"] = "checkpoints/gemma4_duplex_sota_ckpt"
os.environ["PHASE2_START_STEP"] = "12000"
os.environ["PHASE2_RUN_LOG"] = "data/phase2_run_log_50h_mdc_ep1_resume.tsv"
os.environ["PHASE2_CKPT_LOG"] = "data/phase2_ckpt_log_50h.txt"
os.environ["PHASE2_CKPT_EVERY"] = "1000"
# nessuna smoke: run completo fino a fine epoca. Il fix EPOCA COMPLETA in
# kaggle_pipeline.py fa break quando _abs_step >= _start_step + len(dataloader),
# quindi da 12000 mancano ~237 batch + validation finale + sintesi (~30 min).

# 3) Training (completamento della 1 epoca su 38.7h audio: step 12000 -> 12237)
KP.run_training_pipeline(model, num_epochs=1)

print("=== RESUME TRAINING COMPLETATO ===")
sys.stdout.flush()
