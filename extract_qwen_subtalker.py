# -*- coding: utf-8 -*-
"""
extract_qwen_subtalker.py — ESTRATTORE del sub-talker acustico dal teacher
Qwen/Qwen3-TTS-12Hz-1.7B-Base (Fase 1).

Cosa estrae (variante **Base**, vincolante: i pesi del sub-talker differiscono
tra varianti — hash verificato `talker.code_predictor.lm_head.0.weight`):

  * talker.model.codec_embedding.weight (3072 x 2048)   -> cb0_embedding.weight
  * talker.code_predictor.model.codec_embedding.{0..14} (2048 x 2048)
                                                        -> group_embeddings.{i}
  * talker.code_predictor.small_to_mtp_projection.{w,b} (2048->1024, bias=True)
  * talker.code_predictor.model.layers.{0..4}.*         -> layers.{i}.*
    (5 x [q,k,v,o_proj + q_norm/k_norm RMSNorm per-head + gate/up/down +
     input_layernorm + post_attention_layernorm])
  * talker.code_predictor.model.norm.weight             -> norm.weight
  * talker.code_predictor.lm_head.{0..14}.weight (2048 x 1024, bias=False)
                                                        -> lm_head.{i}.weight

NON estrae: speaker_encoder (serve solo alla clonazione vocale del modello
completo, NON al coupling), talker completo (text_embedding/layers/text_
projection/codec_head) — l'input del sub-talker è [h_t, codici], la voce arriva
da h_t di Gemma (adapter trainabile): nessun sample vocale a inferenza.

Metodo di download: Range HTTP parziale (header safetensors -> offset esatti dei
tensori -> merge in intervalli contigui -> GET parallele). Il repo è GATED:
serve un token HF in env `HF_TOKEN` (il token NON viene mai scritto su disco).
Estrae ~363MB invece di scaricare l'intero checkpoint da 3.8GB. Fallback: se
le richieste parziali falliscono, download completo via huggingface_hub.

Output (nella directory di destinazione, su /kaggle/working per la Fase 2):
  * qwen_subtalker.safetensors  (pesi slim del sub-talker)
  * config.json                 (config del sub-talker)
Entrambi letti da `load_qwen_subtalker` di qwen_subtalker.py (strict=True).
"""
import os
import sys
import json
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_FILE = "model.safetensors"
HEADER_CHUNK = 2_000_000  # header + un po' di margine

# Nomi HF -> nomi slim del modulo qwen_subtalker.py
PREFIX_CP = "talker.code_predictor."
PREFIX_CB0 = "talker.model.codec_embedding."
PREFIX_GROUP = "talker.code_predictor.model.codec_embedding."
PREFIX_NORM = "talker.code_predictor.model.norm."
PREFIX_LAYERS = "talker.code_predictor.model.layers."
PREFIX_LM = "talker.code_predictor.lm_head."

CONFIG_KEYS = [
    "talker_hidden_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "rms_norm_eps", "vocab_size", "num_code_groups",
    "max_position_embeddings", "rope_theta", "cb0_vocab_size",
]

SLIM_CONFIG = {
    "talker_hidden_size": 2048,
    "hidden_size": 1024,
    "num_hidden_layers": 5,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "rms_norm_eps": 1e-6,
    "vocab_size": 2048,
    "num_code_groups": 16,
    "max_position_embeddings": 65536,
    "rope_theta": 10000.0,
    "cb0_vocab_size": 3072,
}


def _url(repo: str, fname: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{fname}"


def _headers(token: str) -> dict:
    h = {"User-Agent": "Mozilla/5.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _download_range(url: str, start: int, end: int, token: str, timeout=300) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={**_headers(token), "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_full(url: str, token: str, dest: str) -> None:
    import urllib.request
    print(f"[Extract] Download completo fallback: {url} -> {dest} (potrebbe richiedere molto tempo)...")
    sys.stdout.flush()
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"[Extract] Download completo completato: {dest}")
    sys.stdout.flush()


def extract_subtalker(out_dir: str, big_disk_dir: str, repo: str = REPO,
                      max_parallel: int = 6) -> dict:
    """Estrae i pesi del sub-talker dal checkpoint Base (download parziale)."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(big_disk_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    url = _url(repo, MODEL_FILE)

    print(f"[Extract] Repo {repo} (gated: {'token presente' if token else 'NESSUN TOKEN'})")
    sys.stdout.flush()
    if not token:
        print("[Extract] ⚠️ Repo gated: senza HF_TOKEN il download fallirà. "
              "Imposta il secret Kaggle HF_TOKEN.")
        sys.stdout.flush()

    # 1. Header safetensors
    header_bytes = _download_range(url, 0, HEADER_CHUNK, token)
    header_len = int.from_bytes(header_bytes[0:8], "little")
    header = json.loads(header_bytes[8:8 + header_len].decode())
    # I data_offsets di safetensors sono RELATIVI al blocco dati (che inizia a
    # 8 + header_len nel file): per i range HTTP servono offset assoluti di file.
    data_base = 8 + header_len

    # 2. Selezione tensori del sub-talker
    selected = {}
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        if k.startswith(PREFIX_CB0) or k.startswith(PREFIX_CP):
            selected[k] = meta
    print(f"[Extract] Tensori selezionati: {len(selected)} (su {len(header) - 1} totali)")
    sys.stdout.flush()

    # 3. Merge intervalli contigui (gap <= 1MB) per ridurre il numero di richieste
    spans = sorted(
        (data_base + meta["data_offsets"][0], data_base + meta["data_offsets"][1])
        for meta in selected.values()
    )
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= 1_000_000:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    total_bytes = sum(e - s for s, e in merged)
    print(f"[Extract] Intervalli: {len(merged)} (totale {total_bytes / 1e6:.0f} MB)")
    sys.stdout.flush()

    # 4. Download parallelo
    t0 = time.time()
    chunks = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(_download_range, url, s, e - 1, token): (s, e) for s, e in merged}
        for i, fut in enumerate(futs):
            s, e = futs[fut]
            chunks[(s, e)] = fut.result()
            if (i + 1) % 5 == 0 or i + 1 == len(futs):
                print(f"[Extract] Scaricati {i + 1}/{len(merged)} intervalli "
                      f"({(time.time() - t0):.1f}s)...")
                sys.stdout.flush()

    # 5. Ricostruzione tensori (mapping nomi HF -> slim)
    out_state = {}
    for k, meta in selected.items():
        s, e = meta["data_offsets"]  # offset RELATIVI al blocco dati
        buf = b""
        for (ms, me) in merged:  # ms/me assoluti di file; base = data_base
            if ms <= data_base + s and data_base + e <= me:
                buf = chunks[(ms, me)][(data_base + s) - ms: (data_base + e) - ms]
                break
        arr = np.frombuffer(buf, dtype=np.uint8)
        # dtype: nel header safetensors è dichiarato sotto `dtype` (F32/F16/BF16...);
        # per robustezza lo deduciamo anche dalla lunghezza del payload.
        numel = int(np.prod(meta["shape"]))
        nbytes = e - s
        dtype_name = meta.get("dtype")
        if dtype_name is None or dtype_name not in {"F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"}:
            dtype_name = {8: "F64", 4: "F32", 2: "BF16", 1: "U8"}.get(nbytes // max(1, numel), "F32")
        np_dtype = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
                    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
                    "U8": np.uint8, "BOOL": np.bool_}.get(dtype_name)
        if dtype_name == "BF16":
            # numpy NON ha bfloat16: lettura raw uint16 + view torch (nessuna dipendenza extra)
            tensor = torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).copy())
            tensor = tensor.view(torch.bfloat16).reshape(meta["shape"]).clone()
        else:
            tensor = torch.from_numpy(np.frombuffer(buf, dtype=np_dtype).reshape(meta["shape"]).copy())
        out_state[_map_key(k)] = tensor

    # 6. Verifica conteggio: 1 (cb0) + 88 (code_predictor) = 89 tensori attesi
    print(f"[Extract] Tensori slim: {len(out_state)}")
    for expected in (89,):
        if len(out_state) != expected:
            print(f"[Extract] ⚠️ attesi {expected} tensori, trovati {len(out_state)} "
                  "(verifica il mapping).")

    # 7. Salvataggio
    st_path = os.path.join(out_dir, "qwen_subtalker.safetensors")
    cfg_path = os.path.join(out_dir, "config.json")
    import safetensors.torch as st
    st.save_file(out_state, st_path)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(SLIM_CONFIG, f, indent=2)
    size_mb = os.path.getsize(st_path) / 1e6
    print(f"[Extract] Salvato: {st_path} ({size_mb:.1f} MB) + {cfg_path}")
    sys.stdout.flush()
    return out_state


def _map_key(k: str) -> str:
    if k.startswith(PREFIX_CB0):
        return "cb0_embedding.weight"
    if k.startswith(PREFIX_GROUP):
        return "group_embeddings." + k[len(PREFIX_GROUP):]  # codec_embedding.{i}.weight -> group_embeddings.{i}.weight
    if k.startswith(PREFIX_LAYERS):
        return "layers." + k[len(PREFIX_LAYERS):]
    if k.startswith(PREFIX_NORM):
        return "norm." + k[len(PREFIX_NORM):]
    if k.startswith(PREFIX_LM):
        return "lm_head." + k[len(PREFIX_LM):]
    if k.startswith(PREFIX_CP):
        return k[len(PREFIX_CP):]  # small_to_mtp_projection.*
    raise KeyError(f"chiave HF non mappata: {k}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Estrae il sub-talker Qwen3-TTS Base")
    parser.add_argument("--out-dir", default="qwen_subtalker")
    parser.add_argument("--big-disk", default=None, help="dir disco grande per i file temporanei")
    args = parser.parse_args()

    big = args.big_disk
    if big is None:
        big = os.environ.get("KAGGLE_BIG_DISK", "/root/kaggle_big")

    t0 = time.time()
    try:
        extract_subtalker(args.out_dir, big)
    except Exception as e:
        print(f"[Extract] ERRORE download parziale: {e}")
        sys.stdout.flush()
        # Fallback: download completo + lettura da disco (più lento ma robusto)
        token = os.environ.get("HF_TOKEN", "")
        full_path = os.path.join(big, "qwen3_tts_base_model.safetensors")
        if not os.path.exists(full_path):
            _download_full(_url(REPO, MODEL_FILE), token, full_path)
        from safetensors.torch import load_file
        full = load_file(full_path)
        out_state = {}
        for k in full:
            if k.startswith(PREFIX_CB0) or k.startswith(PREFIX_CP):
                out_state[_map_key(k)] = full[k]
        os.makedirs(args.out_dir, exist_ok=True)
        import safetensors.torch as st
        st.save_file(out_state, os.path.join(args.out_dir, "qwen_subtalker.safetensors"))
        with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(SLIM_CONFIG, f, indent=2)
        print(f"[Extract] Salvato via fallback: {args.out_dir} ({len(out_state)} tensori)")
        sys.stdout.flush()

    # Verifica finale: load + smoke test con load_qwen_subtalker
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from qwen_subtalker import load_qwen_subtalker
        module, cfg = load_qwen_subtalker(args.out_dir)
        torch.manual_seed(0)
        with torch.no_grad():
            B, T = 2, 6
            h_t = torch.randn(
                B, T, cfg["talker_hidden_size"],
                dtype=next(module.parameters()).dtype,  # bf16: allineato al modello
            )
            codes = torch.randint(0, cfg["vocab_size"], (B, T, cfg["num_code_groups"]))
            logits = module.forward_sub_talker_finetune(h_t, codes)
        assert tuple(logits.shape) == (B, T, cfg["num_code_groups"] - 1, cfg["vocab_size"]), logits.shape
        loss = module.coupling_loss(logits, codes)
        assert torch.isfinite(loss), loss
        print(f"[Extract] ✅ SMOKE TEST OK: logits {tuple(logits.shape)}, coupling_loss {float(loss):.4f} "
              f"({(time.time() - t0):.1f}s totali)")
        sys.stdout.flush()
        return 0
    except Exception as e:
        print(f"[Extract] ❌ Verifica fallita: {e}")
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
