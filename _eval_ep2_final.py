# -*- coding: utf-8 -*-
"""
EVAL-ONLY sul checkpoint FINALE epoca 2 (step 24236, run v7).
Nessun training: PHASE2_SMOKE_BATCHES=0 -> il loop fa break al primo batch,
poi la pipeline esegue validation finale + checkpoint + sintesi (con
synthesis_metrics: mel-L1/MCD generated vs oracle) sui pesi epoca 2.

Uso (in Kaggle, kernel):
    python /kaggle/working/_eval_ep2_final.py
"""
import os
import sys

sys.path.insert(0, "/kaggle/input/datasets/andrlab/gemma-4-omni-module")
sys.path.insert(0, "/kaggle/working")

import unsloth  # noqa: F401  (applica le patch in un processo FRESCO)

# ---- Config eval-only epoca 2 ----
os.environ["PHASE2_RESUME_CKPT"] = "checkpoints/gemma4_duplex_sota_ckpt"
os.environ["PHASE2_START_STEP"] = "24236"   # resume dai pesi finali epoca 2
os.environ["PHASE2_SMOKE_BATCHES"] = "0"    # 0 batch di training: solo eval
os.environ["PHASE2_RUN_LOG"] = "data/phase2_run_log_ep2_final_eval.tsv"
os.environ["PHASE2_CKPT_LOG"] = "data/phase2_ckpt_log_ep2_final_eval.txt"

import kaggle_pipeline as KP

print("=== EVAL-ONLY: checkpoint finale epoca 2 (step 24236) ===")
sys.stdout.flush()

duplex_model, _ = KP.load_gemma4_duplex_system()
KP.run_training_pipeline(duplex_model, num_epochs=1)

print("=== EVAL-ONLY EPOCA 2 COMPLETATO ===")
