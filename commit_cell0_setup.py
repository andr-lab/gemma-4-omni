# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 0 — IMPORTS & SETUP (commit epoca 2)
# Esegue in batch session (Save & Run All / commit): niente idle timeout, 9h.
# ============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single-GPU su Kaggle 2xT4
import sys
import json
import time
import gc
import shutil

# Il dataset input gemma4-ep2-data contiene: kaggle_pipeline.py (aggiornato con
# lr env + gradient clipping + metrica sintesi), parquet 50h, checkpoint epoca 1,
# subtalker, code2wav_weights, moduli. Copiamo tutto in /kaggle/working.
_INPUT = "/kaggle/input/andrlab/gemma4-ep2-data"
print("Input dataset files:", os.listdir(_INPUT) if os.path.isdir(_INPUT) else "MANCANTE")
sys.stdout.flush()

def _copy_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            _copy_tree(s, d)
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)

# Copia moduli + dati in working (idempotente)
for sub in ("data", "checkpoints", "code2wav_weights", "qwen_subtalker"):
    s = os.path.join(_INPUT, sub)
    if os.path.isdir(s):
        _copy_tree(s, sub)
for f in ("kaggle_pipeline.py", "code2wav_decoder.py", "extract_qwen_subtalker.py",
          "qwen_subtalker.py", "_epoca2.py"):
    s = os.path.join(_INPUT, f)
    if os.path.isfile(s):
        d = os.path.join("/kaggle/working", f)
        if not os.path.exists(d):
            shutil.copy2(s, d)

sys.path.insert(0, "/kaggle/working")
# verifica parquet
import glob
par = glob.glob("data/train_tokens_50.0h_2500s_mel.parquet")
print("parquet 50h presente:", bool(par), par)
sys.stdout.flush()
