# -*- coding: utf-8 -*-
"""
PUSH DEL NOTEBOOK DI COMMIT per l'epoca 2 (batch session, 9h, niente idle timeout).
Eseguito NEL KERNEL Kaggle (dove la CLI kaggle ha le credenziali della sessione):
    python /kaggle/working/_push_commit.py

Il notebook gemma4-epoca2-commit:
  cella 0: copia dati dal dataset input gemma4-ep2-data -> /kaggle/working
  cella 1: pip install transformers==5.5.0 + unsloth
  cella 2: epoca 2 (lr 5e-5, max_grad_norm 1.0, resume da step 12000)
"""
import os
import shutil
import subprocess

SRC = "/kaggle/working/commit_nb"
os.makedirs(SRC, exist_ok=True)

# 1) kernel-metadata.json
meta = {
    "id": "andrlab/gemma4-epoca2-commit",
    "title": "gemma4-epoca2-commit",
    "code_file": "gemma4-epoca2-commit.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "competition_sources": [],
    "dataset_sources": ["andrlab/gemma4-ep2-data"],
    "kernel_sources": [],
}
with open(os.path.join(SRC, "kernel-metadata.json"), "w") as f:
    import json
    json.dump(meta, f, indent=2)
print("kernel-metadata.json scritto")

# 2) il notebook .ipynb (da file locale sincronizzato)
nb_src = "/kaggle/working/commit_nb/gemma4-epoca2-commit.ipynb"
if not os.path.exists(nb_src):
    print("NB: notebook non presente — va sincronizzato dal locale (PUT)")
else:
    print("notebook presente:", os.path.getsize(nb_src), "bytes")

# 3) push (avvia il run batch)
r = subprocess.run(
    ["kaggle", "kernels", "push", "-p", SRC],
    capture_output=True, text=True, timeout=600
)
print("rc:", r.returncode)
print(r.stdout[-1200:] if r.stdout else r.stderr[-800:])
