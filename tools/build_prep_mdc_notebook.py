# -*- coding: utf-8 -*-
import os, sys, json, ast, argparse

def build_notebook(out_dir="kernel_prep_mdc_r1", slug="gemma4-prep-mdc-r1"):
    os.makedirs(out_dir, exist_ok=True)
    
    cell0_source = """# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 0 - IMPORTS & SETUP DATA PREPARATION (Mozilla Common Voice MDC Italiano)
# ============================================================================
import os, sys, traceback
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import json, time, gc, shutil, zipfile, hashlib

    _CANDIDATES = [
        "/kaggle/input/andrlab/gemma-4-omni-module",
        "/kaggle/input/datasets/andrlab/gemma-4-omni-module",
    ]
    _INPUT = None
    for c in _CANDIDATES:
        if os.path.isdir(c):
            _INPUT = c
            break
    print("Input dataset files:", os.listdir(_INPUT) if _INPUT else "MANCANTE")
    sys.stdout.flush()

    if _INPUT:
        for f in ("kaggle_pipeline.py", "code2wav_decoder.py", "extract_qwen_subtalker.py", "qwen_subtalker.py"):
            s = os.path.join(_INPUT, f)
            if os.path.isfile(s):
                d = os.path.join("/kaggle/working", f)
                if not os.path.exists(d):
                    shutil.copy2(s, d)
        for sub in ("data", "qwen_subtalker"):
            s = os.path.join(_INPUT, sub)
            if os.path.isdir(s):
                for item in os.listdir(s):
                    sp = os.path.join(s, item)
                    dp = os.path.join(sub, item)
                    os.makedirs(os.path.dirname(dp), exist_ok=True)
                    if not os.path.exists(dp) and os.path.isfile(sp):
                        shutil.copy2(sp, dp)

    sys.path.insert(0, "/kaggle/working")
    print("=== SETUP PREPARATION MDC COMPLETATO ===")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("SETUP FALLITO:\\n" + tb)
    sys.stdout.flush()
    raise
"""

    cell1_install = """# CELLA INSTALL - Dipendenze per tokenizzazione Qwen3-TTS & datasets HF/MDC
import subprocess, sys, os

r = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet',
     'qwen-tts==0.1.1', 'datasets', 'soundfile', 'pyarrow', 'tqdm', 'librosa', 'huggingface_hub', 'datacollective'],
    capture_output=True, text=True, timeout=1800)
print("install rc:", r.returncode, (r.stdout + r.stderr)[-200:])
sys.stdout.flush()
"""

    with open("kernel_src/prep_mdc_cell1.py", "r", encoding="utf-8") as f:
        prep_code = f.read()

    # Validazione AST
    for idx, code in enumerate([cell0_source, cell1_install, prep_code]):
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
                "metadata": {},
                "outputs": [],
                "source": cell0_source.splitlines(keepends=True)
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell1_install.splitlines(keepends=True)
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": prep_code.splitlines(keepends=True)
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = os.path.join(out_dir, f"{slug}.ipynb")
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=2)

    # Validazione byte ASCII
    with open(nb_path, "rb") as f:
        data = f.read()
    non_ascii = [b for b in data if b > 127]
    assert len(non_ascii) == 0, f"Trovati {len(non_ascii)} byte non-ASCII nel notebook!"

    meta = {
        "id": f"andrlab/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "andrlab/gemma-4-omni-module"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Notebook generato: {nb_path} ({len(data)} bytes, 0 non-ASCII)")
    print(f"[OK] Metadata generato: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="kernel_prep_mdc_r1")
    parser.add_argument("--slug", default="gemma4-prep-mdc-r1")
    args = parser.parse_args()
    build_notebook(out_dir=args.out_dir, slug=args.slug)
