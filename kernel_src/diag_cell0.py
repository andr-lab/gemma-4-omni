# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 0 - IMPORTS & SETUP (diagnosi audio metallico, v6)
# ============================================================================
import os, sys, traceback
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import json, time, gc, shutil, zipfile, hashlib

    _CANDIDATES = [
        "/kaggle/input/andrlab/gemma-4-omni-module",
        "/kaggle/input/datasets/andrlab/gemma-4-omni-module",
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

    sys.path.insert(0, "/kaggle/working")
    import glob
    par = glob.glob("data/train_tokens_50.0h_2500s_mel.parquet")
    print("parquet 50h presente:", bool(par))
    sys.stdout.flush()

    # ---- Checkpoint FINALE epoca 2 in lora_adapters/ ----
    if _CKPT_INPUT:
        dst = "checkpoints/gemma4_duplex_sota_ckpt"
        os.makedirs(os.path.join(dst, "lora_adapters"), exist_ok=True)
        def _find_ckpt_root(root):
            for cand in (os.path.join(root, "checkpoints", "gemma4_duplex_sota_ckpt"),
                         os.path.join(root, "gemma4_duplex_sota_ckpt")):
                if os.path.isdir(cand):
                    return cand
            for dirpath, dirnames, filenames in os.walk(root):
                if os.path.basename(dirpath) == "gemma4_duplex_sota_ckpt" and "lora_adapters" in dirnames:
                    return dirpath
            return None
        base = _find_ckpt_root(_CKPT_INPUT)
        if base is None:
            for sub in os.listdir(_CKPT_INPUT):
                subp = os.path.join(_CKPT_INPUT, sub)
                if os.path.isdir(subp):
                    base = _find_ckpt_root(subp)
                    if base is not None:
                        break
        if base is None:
            print("!!! base ckpt NON trovata nel dataset ep2 !!!")
        else:
            for f in ("adapter_model.safetensors", "adapter_config.json", "README.md"):
                s = os.path.join(base, "lora_adapters", f)
                if os.path.isfile(s):
                    shutil.copy2(s, os.path.join(dst, "lora_adapters", f))
            s = os.path.join(base, "custom_duplex_heads.pt")
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(dst, "custom_duplex_heads.pt"))
            h = hashlib.md5(open(os.path.join(dst, "lora_adapters", "adapter_model.safetensors"), "rb").read()).hexdigest()[:8]
            print(f"ckpt ep2 md5(adapter): {h} (atteso 86a854e9)")
    sys.stdout.flush()
    print("=== SETUP COMPLETATO ===")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("SETUP FALLITO:\n" + tb)
    sys.stdout.flush()
    with open("setup_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
