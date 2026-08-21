# -*- coding: utf-8 -*-
import os, sys, json, ast, argparse

def build_notebook(smoke=True, out_dir="kernel_adapter_it_r1", slug="gemma4-adapter-it-r1"):
    os.makedirs(out_dir, exist_ok=True)
    
    cell0_source = """# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 0 - IMPORTS & SETUP (Gemma 4 Frozen + Temporal Residual Adapter su Voce IT)
# ============================================================================
import os, sys, traceback
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import json, time, gc, shutil, zipfile, hashlib

    _CANDIDATES = [
        "/kaggle/input/andrlab/gemma-4-omni-module",
        "/kaggle/input/datasets/andrlab/gemma-4-omni-module",
        "/kaggle/input/andrlab/gemma4-ht-adapter-r2",
        "/kaggle/input/datasets/andrlab/gemma4-ht-adapter-r2",
        "/kaggle/input/andrlab/gemma-4-omni-module-mdc",
        "/kaggle/input/datasets/andrlab/gemma-4-omni-module-mdc",
    ]
    _INPUT = None
    for c in _CANDIDATES:
        if os.path.isdir(c):
            if "gemma-4-omni-module" in c and _INPUT is None:
                _INPUT = c
    print("Input dataset files:", os.listdir(_INPUT) if _INPUT else "MANCANTE")
    sys.stdout.flush()

    def _extract_zip(zpath, dst):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dst)

    def _copy_tree(src, dst):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.islink(s):
                continue
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                _copy_tree(s, d)
            else:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if not os.path.exists(d):
                    shutil.copy2(s, d)

    if _INPUT:
        for f in ("kaggle_pipeline.py", "code2wav_decoder.py", "extract_qwen_subtalker.py",
                  "qwen_subtalker.py"):
            s = os.path.join(_INPUT, f)
            if os.path.isfile(s):
                d = os.path.join("/kaggle/working", f)
                if not os.path.exists(d):
                    shutil.copy2(s, d)
        for sub in ("data", "checkpoints", "code2wav_weights", "qwen_subtalker"):
            s = os.path.join(_INPUT, sub)
            z = os.path.join(_INPUT, sub + ".zip")
            if os.path.isdir(s):
                _copy_tree(s, sub)
            elif os.path.isfile(z):
                os.makedirs(sub, exist_ok=True)
                _extract_zip(z, sub)
                print(f"zip estratto: {sub}.zip -> {sub}/")

    # Copia warm-start adapter R2 se disponibile
    for cand in ("/kaggle/input/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
                 "/kaggle/input/datasets/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt"):
        if os.path.isfile(cand):
            os.makedirs("checkpoints", exist_ok=True)
            shutil.copy2(cand, "checkpoints/ht_adapter_stage2r2_final.pt")
            print(f"ht_adapter R2 copiato da {cand} a checkpoints/ht_adapter_stage2r2_final.pt")
            break

    sys.path.insert(0, "/kaggle/working")
    print("=== SETUP ADAPTER IT COMPLETATO ===")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("SETUP FALLITO:\\n" + tb)
    sys.stdout.flush()
    raise
"""

    cell1_source = """# CELLA INSTALL - dipendenze Fase 2 su GPU
import subprocess, sys, os

r = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
     'transformers==5.5.0', 'unsloth', 'safetensors', 'soundfile', 'pyarrow', 'datasets', 'librosa'],
    capture_output=True, text=True, timeout=1800)
print("stack install rc:", r.returncode, (r.stdout + r.stderr)[-200:])
sys.stdout.flush()
"""

    if smoke:
        cell2_header = """# ---- RUN SMOKE ADAPTER IT (PHASE2_SMOKE_BATCHES=4) ----
import os
os.environ.setdefault("STAGE2_ADAPTER_IT", "1")
os.environ.setdefault("STAGE2_TEMPORAL2", "1")
os.environ.setdefault("STAGE2_TEMPORAL", "1")
os.environ.setdefault("SMOKE", "1")
os.environ.setdefault("PHASE2_SMOKE_BATCHES", "4")
os.environ.setdefault("FITEX_N", "16")
os.environ.setdefault("TEMPORAL_DIM", "1024")
os.environ.setdefault("TEMPORAL_LAYERS", "4")
os.environ.setdefault("TEMPORAL_HEADS", "8")
os.environ.setdefault("TEMPORAL_FF_DIM", "2048")
os.environ.setdefault("TEMPORAL_LR_ADAPTER", "1.5e-4")
os.environ.setdefault("TEMPORAL_LR_MODULE", "1.5e-4")
os.environ.setdefault("TEMPORAL_WARMUP", "1")
os.environ.setdefault("STAGE2_BATCH_SIZE", "8")
os.environ.setdefault("STAGE2_GRAD_ACCUM", "2")
os.environ.setdefault("STAGE2_LAMBDA_COUPLING", "0.2")
os.environ.setdefault("STAGE2_CKPT_EVERY", "2")
os.environ.setdefault("EVAL_ADAPTER_IT", "1")
os.environ.setdefault("EVAL_TEMPORAL2", "1")
os.environ.setdefault("EVAL_N_VAL", "4")
os.environ.setdefault("EVAL_BATCH", "4")
os.environ.setdefault("ROPE_FIX", "1")
"""
    else:
        cell2_header = """# ---- RUN COMPLETO ADAPTER IT (PHASE2_SMOKE_BATCHES=1200, ~1.5h su T4) ----
import os
os.environ.setdefault("STAGE2_ADAPTER_IT", "1")
os.environ.setdefault("STAGE2_TEMPORAL2", "1")
os.environ.setdefault("STAGE2_TEMPORAL", "1")
os.environ.setdefault("SMOKE", "0")
os.environ.setdefault("PHASE2_SMOKE_BATCHES", "1200")
os.environ.setdefault("FITEX_N", "1000")
os.environ.setdefault("TEMPORAL_DIM", "1024")
os.environ.setdefault("TEMPORAL_LAYERS", "4")
os.environ.setdefault("TEMPORAL_HEADS", "8")
os.environ.setdefault("TEMPORAL_FF_DIM", "2048")
os.environ.setdefault("TEMPORAL_LR_ADAPTER", "1.5e-4")
os.environ.setdefault("TEMPORAL_LR_MODULE", "1.5e-4")
os.environ.setdefault("TEMPORAL_WARMUP", "100")
os.environ.setdefault("STAGE2_BATCH_SIZE", "8")
os.environ.setdefault("STAGE2_GRAD_ACCUM", "2")
os.environ.setdefault("STAGE2_LAMBDA_COUPLING", "0.2")
os.environ.setdefault("STAGE2_CKPT_EVERY", "250")
os.environ.setdefault("EVAL_ADAPTER_IT", "1")
os.environ.setdefault("EVAL_TEMPORAL2", "1")
os.environ.setdefault("EVAL_N_VAL", "100")
os.environ.setdefault("EVAL_BATCH", "4")
os.environ.setdefault("ROPE_FIX", "1")
"""

    with open("kernel_src/adapter_it_cell2.py", "r", encoding="utf-8") as f:
        cell2_body = f.read()

    cell2_full = cell2_header + "\n" + cell2_body

    # Validazione AST
    for idx, code in enumerate([cell0_source, cell1_source, cell2_full]):
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"ERRORE DI SINTASSI IN CELLA {idx}: {e}")
            raise

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c0_setup",
                "metadata": {},
                "outputs": [],
                "source": cell0_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c1_install",
                "metadata": {},
                "outputs": [],
                "source": cell1_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "c2_adapter_it",
                "metadata": {},
                "outputs": [],
                "source": cell2_full.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = os.path.join(out_dir, f"{slug}.ipynb")
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)

    # Verifica byte ASCII
    with open(nb_path, "rb") as f:
        data = f.read()
    non_ascii = [b for b in data if b > 127]
    assert len(non_ascii) == 0, f"Trovati {len(non_ascii)} byte non-ASCII nel notebook!"

    # Genera kernel-metadata.json
    meta = {
        "id": f"andrlab/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "andrlab/gemma-4-omni-module",
            "andrlab/gemma4-ht-adapter-r2",
        ],
    }

    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)

    print(f"Notebook {'SMOKE' if smoke else 'FULL'} generato con successo in: {nb_path} ({len(data)} bytes, 0 non-ASCII)")
    print(f"Metadata generato in: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default="kernel_adapter_it_r1")
    parser.add_argument("--slug", type=str, default="gemma4-adapter-it-r1")
    args = parser.parse_args()
    
    is_smoke = not args.full
    build_notebook(smoke=is_smoke, out_dir=args.out_dir, slug=args.slug)
