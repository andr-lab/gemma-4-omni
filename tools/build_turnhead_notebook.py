# -*- coding: utf-8 -*-
import os, sys, json, ast, argparse

def build_notebook(smoke=True, out_dir="kernel_turnhead_r1", slug="gemma4-turnhead-r1"):
    os.makedirs(out_dir, exist_ok=True)
    
    cell0_source = """# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 0 - IMPORTS & SETUP (Gemma 4 Frozen + Adapter Frozen + TurnStateHead)
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
        "/kaggle/input/andrlab/gemma4-ep2-final-ckpt",
        "/kaggle/input/datasets/andrlab/gemma4-ep2-final-ckpt",
    ]
    _INPUT = None
    _CKPT_INPUT = None
    for c in _CANDIDATES:
        if os.path.isdir(c):
            if "gemma4-ep2-final-ckpt" in c:
                _CKPT_INPUT = c
            elif _INPUT is None:
                _INPUT = c
    print("Input dataset files:", os.listdir(_INPUT) if _INPUT else "MANCANTE")
    print("_INPUT =", _INPUT, "| _CKPT_INPUT =", _CKPT_INPUT)
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
                  "qwen_subtalker.py", "_epoca2.py"):
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

    # Copia warm-start adapter R2 se disponibile nei dataset input
    for cand in ("/kaggle/input/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
                 "/kaggle/input/datasets/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt"):
        if os.path.isfile(cand):
            os.makedirs("checkpoints", exist_ok=True)
            shutil.copy2(cand, "checkpoints/ht_adapter_stage2r2_final.pt")
            print(f"ht_adapter R2 copiato da {cand} a checkpoints/ht_adapter_stage2r2_final.pt")
            break

    sys.path.insert(0, "/kaggle/working")
    import glob
    par = glob.glob("data/train_tokens_50.0h_2500s_mel.parquet")
    print("parquet 50h presente:", bool(par))
    sys.stdout.flush()

    print("=== SETUP COMPLETATO ===")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("SETUP FALLITO:\\n" + tb)
    sys.stdout.flush()
    with open("setup_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
"""

    cell1_source = """# CELLA INSTALL - dipendenze Fase 2 su GPU
import subprocess, sys, os, re as _re

def gpu_info():
    try:
        r = subprocess.run([sys.executable, '-c',
            "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"],
            capture_output=True, text=True, timeout=120)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        cap = None
        if len(lines) > 1:
            m = _re.search(r'(\\d+)', lines[1])
            cap = int(m.group(1)) if m else None
        return (lines[0] if lines else "?"), cap
    except Exception as e:
        return f"ERR {str(e)[:50]}", None

name, cap = gpu_info()
print("GPU rilevata:", name, "| compute capability:", cap)
sys.stdout.flush()

if cap is not None and cap < 7:
    print(">>> GPU Pascal rilevata: installo torch 2.6.0+cu118 <<<")
    sys.stdout.flush()
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet',
         '--index-url', 'https://download.pytorch.org/whl/cu118',
         'torch==2.6.0', 'torchvision==0.21.0', 'torchaudio==2.6.0'],
        capture_output=True, text=True, timeout=1800)
    print("torch cu118 rc:", r.returncode, (r.stdout + r.stderr)[-150:])
    sys.stdout.flush()
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
         'transformers==5.5.0', 'unsloth', 'unsloth_zoo', 'peft', 'safetensors', 'soundfile', 'pyarrow',
         'trl', 'xformers==0.0.29.post3', 'bitsandbytes', 'triton', 'tyro', 'structlog',
         'nest-asyncio', 'hf_transfer', 'diffusers', 'typer', 'sentencepiece', 'datasets'],
        capture_output=True, text=True, timeout=1800)
    print("stack cu118 rc:", r.returncode, (r.stdout + r.stderr)[-200:])
else:
    print(">>> GPU Volta/Turing+ rilevata: installo stack aggiornato <<<")
    sys.stdout.flush()
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
         'transformers==5.5.0', 'unsloth', 'safetensors', 'soundfile', 'pyarrow'],
        capture_output=True, text=True, timeout=1800)
    print("stack cu128 rc:", r.returncode, (r.stdout + r.stderr)[-250:])
    sys.stdout.flush()

import importlib.metadata as md
for pkg in ['transformers', 'unsloth', 'peft', 'torch']:
    try: print(pkg + ": " + md.version(pkg))
    except Exception: print(pkg + ": MANCANTE")
r2 = subprocess.run([sys.executable, '-c',
    "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"],
    capture_output=True, text=True, timeout=120)
print("check cuda:", (r2.stdout + r2.stderr).strip()[-200:])
"""

    if smoke:
        cell2_header = """# ---- RUN SMOKE TURNHEAD (PHASE2_SMOKE_BATCHES=4, N_TURN_CLASSES=5) ----
import os
os.environ.setdefault("STAGE3_TURNHEAD", "1")
os.environ.setdefault("SMOKE", "1")
os.environ.setdefault("PHASE2_SMOKE_BATCHES", "4")
os.environ.setdefault("FITEX_N", "16")
os.environ.setdefault("WINDOW_T", "64")
os.environ.setdefault("N_TURN_CLASSES", "5")
os.environ.setdefault("TURNHEAD_LR", "3e-4")
os.environ.setdefault("TURNHEAD_WEIGHT_DECAY", "1e-2")
os.environ.setdefault("STAGE3_BATCH_SIZE", "8")
os.environ.setdefault("STAGE3_GRAD_ACCUM", "2")
os.environ.setdefault("STAGE3_CKPT_EVERY", "2")
os.environ.setdefault("EVAL_TURNHEAD", "1")
os.environ.setdefault("EVAL_N_VAL", "8")
os.environ.setdefault("ROPE_FIX", "1")
"""
    else:
        cell2_header = """# ---- RUN COMPLETO TURNHEAD (PHASE2_SMOKE_BATCHES=1200, N_TURN_CLASSES=5, ~1.2h su T4) ----
import os
os.environ.setdefault("STAGE3_TURNHEAD", "1")
os.environ.setdefault("SMOKE", "0")
os.environ.setdefault("PHASE2_SMOKE_BATCHES", "1200")
os.environ.setdefault("FITEX_N", "1000")
os.environ.setdefault("WINDOW_T", "64")
os.environ.setdefault("N_TURN_CLASSES", "5")
os.environ.setdefault("TURNHEAD_LR", "3e-4")
os.environ.setdefault("TURNHEAD_WEIGHT_DECAY", "1e-2")
os.environ.setdefault("STAGE3_BATCH_SIZE", "8")
os.environ.setdefault("STAGE3_GRAD_ACCUM", "2")
os.environ.setdefault("STAGE3_CKPT_EVERY", "250")
os.environ.setdefault("EVAL_TURNHEAD", "1")
os.environ.setdefault("EVAL_N_VAL", "100")
os.environ.setdefault("ROPE_FIX", "1")
"""

    with open("kernel_src/turnhead_cell2.py", "r", encoding="utf-8") as f:
        cell2_body = f.read()

    cell2_full = cell2_header + "\n" + cell2_body

    # Validazione AST di tutte le celle
    for idx, code in enumerate([cell0_source, cell1_source, cell2_full]):
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"ERRORE DI SINTASSI IN CELLA {idx}: {e}")
            raise

    # Costruzione struttura notebook jupyter v4
    cells = [
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
            "id": "c2_turnhead",
            "metadata": {},
            "outputs": [],
            "source": cell2_full.splitlines(keepends=True),
        },
    ]

    nb = {
        "cells": cells,
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

    # Validazione round-trip
    with open(nb_path, "r", encoding="utf-8") as f:
        loaded_nb = json.load(f)
    assert len(loaded_nb["cells"]) == 3
    for i, c in enumerate(loaded_nb["cells"]):
        assert isinstance(c["source"], list), f"Cell {i} source non e una lista!"
        joined = "".join(c["source"])
        ast.parse(joined)

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
    parser.add_argument("--out-dir", type=str, default="kernel_turnhead_r1")
    parser.add_argument("--slug", type=str, default="gemma4-turnhead-r1")
    args = parser.parse_args()
    
    is_smoke = not args.full
    build_notebook(smoke=is_smoke, out_dir=args.out_dir, slug=args.slug)
