# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 1 — EPOCA 2 (lr 5e-5 + gradient clipping) da checkpoint epoca 1
# Importa kaggle_pipeline dal working (aggiornato: PHASE2_LR, PHASE2_MAX_GRAD_NORM,
# resume, metrica sintesi generated-vs-oracle). Subprocess FRESCO + nohup:
# la batch session esegue questa cella; il training gira come subprocess robusto.
# ============================================================================
import unsloth  # noqa: F401  (patch in processo fresco)

import kaggle_pipeline as KP

print("=== COMMIT EPOCA 2: lr=5e-5, max_grad_norm=1.0, resume da step 12000 ===")
sys.stdout.flush()

# env per il training (visti da kaggle_pipeline.run_training_pipeline)
os.environ["PHASE2_RESUME_CKPT"] = "checkpoints/gemma4_duplex_sota_ckpt"
os.environ["PHASE2_START_STEP"] = "12000"
os.environ["PHASE2_LR"] = "5e-5"
os.environ["PHASE2_MAX_GRAD_NORM"] = "1.0"
os.environ["PHASE2_RUN_LOG"] = "data/phase2_run_log_ep2.tsv"
os.environ["PHASE2_CKPT_LOG"] = "data/phase2_ckpt_log_50h_ep2.txt"

duplex_model, _ = KP.load_gemma4_duplex_system()
KP.run_training_pipeline(duplex_model, num_epochs=1)

print("=== EPOCA 2 COMPLETATA ===")
