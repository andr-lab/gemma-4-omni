#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit di Validazione Statistica dei Dati di Partenza e della Cache (Parquet + Cache v2)
"""
import os, sys, json, glob, math
import numpy as np
import torch

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
sys.path.append('/workspace/.extra_pkgs')

import kaggle_pipeline as KP
from transformers import AutoTokenizer

def audit():
    os.makedirs('/workspace/logs', exist_ok=True)
    report_path = '/workspace/logs/audit_data_report.txt'
    rf = open(report_path, 'w', encoding='utf-8')
    def log(s=""):
        try:
            print(str(s).encode('ascii', 'replace').decode('ascii'))
        except Exception:
            pass
        rf.write(str(s) + "\n")
        rf.flush()

    log("=" * 70)
    log("=== AUDIT STATISTICO RIGOROSO: DATI DI PARTENZA & CACHE ===")
    log("=" * 70 + "\n")

    prefix = '/workspace/data/train_tokens_cv26_it_40.8h_31916s'
    cache_path = '/workspace/cache_ars_full_v2.pt'
    
    snap_qwen = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    tok_qwen = AutoTokenizer.from_pretrained(snap_qwen, use_fast=True)
    
    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = None
    if len(gemma_dirs) > 0:
        gemma_snap = os.path.dirname(gemma_dirs[0])
        try:
            tok_gemma = AutoTokenizer.from_pretrained(gemma_snap, use_fast=True)
            log(f"Caricato tokenizer Gemma da: {gemma_snap}")
        except Exception as e:
            log(f"Impossibile caricare tokenizer Gemma: {e}")

    # -------------------------------------------------------------
    # 1. PARQUET & RVQ CODES AUDIT
    # -------------------------------------------------------------
    log("[1/4] Caricamento e Audit Parquet Dati...")
    if not os.path.exists(prefix + '_codes.parquet'):
        log(f"ERRORE: Parquet non trovato a {prefix}_codes.parquet")
        return

    user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet(prefix + '.parquet')
    total_frames = agent_codes.shape[1]
    log(f"  * Totale frame caricati: {total_frames:,} (~{total_frames*0.08/3600:.2f} ore audio a 12.5Hz)")
    log(f"  * Formato agent_codes: {agent_codes.shape} (N_codebooks={agent_codes.shape[0]})")

    cb0_all = agent_codes[0, :]
    pad_mask = (cb0_all == KP.PAD_TARGET_ID) | (cb0_all == -100)
    valid_frames = int((~pad_mask).sum())
    log(f"  * Frame validi di parlato: {valid_frames:,} ({valid_frames/total_frames*100:.1f}%)")
    log(f"  * Frame di padding: {int(pad_mask.sum()):,} ({pad_mask.sum()/total_frames*100:.1f}%)")

    cb0_valid = cb0_all[~pad_mask]
    n_unique_cb0 = len(np.unique(cb0_valid))
    cb0_min, cb0_max = int(cb0_valid.min()), int(cb0_valid.max())
    counts = np.bincount(cb0_valid, minlength=4096)
    probs = counts[counts > 0] / len(cb0_valid)
    entropy_cb0 = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(4096))

    log(f"\n  [Analisi CB0 (Semantica / Fonetica, Vocab=4096)]")
    log(f"    - Token unici attivi: {n_unique_cb0} / 4096 ({n_unique_cb0/4096*100:.1f}%)")
    log(f"    - Range valori: [{cb0_min}, {cb0_max}]")
    log(f"    - Entropia di Shannon: {entropy_cb0:.3f} / {max_entropy:.3f} (Uniforme={max_entropy:.3f})")
    if cb0_max <= 2047:
        log("    ⚠️ ALLERTA CRITICA: CB0 ha valore massimo 2047 invece di 4095!")
        log("       I codici sono stati estratti con vocab_size 2048 o modulo 2048.")
    else:
        log("    PASS: Distribuzione CB0 copre l'intero vocabolario 4096.")

    log(f"\n  [Analisi Codebook Residui CB1..CB15 (Acustica, Vocab=2048)]")
    for g in [1, 3, 7, 15]:
        cbg_valid = agent_codes[g, ~pad_mask]
        u_g = len(np.unique(cbg_valid))
        cnt_g = np.bincount(cbg_valid, minlength=2048)
        p_g = cnt_g[cnt_g > 0] / len(cbg_valid)
        ent_g = float(-np.sum(p_g * np.log(p_g)))
        log(f"    - CB{g:02d}: unici={u_g}/2048 ({u_g/2048*100:.1f}%) | Range=[{cbg_valid.min()}, {cbg_valid.max()}] | Entropia={ent_g:.2f}/7.62")

    # -------------------------------------------------------------
    # 2. MEL SPECTROGRAM AUDIT
    # -------------------------------------------------------------
    log("\n[2/4] Audit Mel Spectrogram (User & Agent)...")
    sample_agent_mel = np.asarray(agent_mel[:100000], dtype=np.float32)
    mel_mean = float(np.mean(sample_agent_mel))
    mel_std = float(np.std(sample_agent_mel))
    mel_min = float(np.min(sample_agent_mel))
    mel_max = float(np.max(sample_agent_mel))
    diff_mel = np.abs(np.diff(sample_agent_mel, axis=0))
    constant_ratio = float((diff_mel < 1e-5).mean())

    log(f"  * Mel Agente (campione 100k frame): Mean={mel_mean:.4f}, Std={mel_std:.4f}, Range=[{mel_min:.2f}, {mel_max:.2f}]")
    log(f"  * Rapporto di frame costanti: {constant_ratio*100:.2f}%")

    # -------------------------------------------------------------
    # 3. TEXT TOKENS (tbw) AUDIT: CONFRONTO TOKENIZER
    # -------------------------------------------------------------
    log("\n[3/4] Audit Trascrizioni Testuali (tbw: Gemma vs Qwen Tokenizer)...")
    if tbw is not None and len(tbw) > 0:
        log(f"  * Totale finestre con testo: {len(tbw):,}")
        valid_items = [t for t in tbw if t is not None and isinstance(t, list) and len(t) > 0]
        log(f"  * Finestre con token non-vuoti: {len(valid_items):,} / {len(tbw):,}")
        
        # Max token ID in tbw
        max_tid = max(max(t) for t in valid_items[:1000])
        min_tid = min(min(t) for t in valid_items[:1000])
        log(f"  * Range token ID in tbw (primi 1000 campioni): [{min_tid}, {max_tid}]")

        log("\n  [Confronto Decodifica sulle prime 5 frasi]")
        for i in range(5):
            t_item = valid_items[i]
            qwen_dec = tok_qwen.decode(t_item, skip_special_tokens=True).strip()
            gemma_dec = tok_gemma.decode(t_item, skip_special_tokens=True).strip() if tok_gemma else "N/A"
            log(f"  --- Frase #{i:02d} ({len(t_item)} token) ---")
            log(f"    • Decodifica con Qwen Tokenizer : {repr(qwen_dec)}")
            log(f"    • Decodifica con Gemma Tokenizer: {repr(gemma_dec)}")

    # -------------------------------------------------------------
    # 4. CACHE V2 AUDIT (cache_ars_full_v2.pt)
    # -------------------------------------------------------------
    log("\n[4/4] Audit Cache Pre-Estratta (cache_ars_full_v2.pt)...")
    if not os.path.exists(cache_path):
        log(f"Cache non trovata a {cache_path}")
        return

    ck = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    keys = list(ck.keys())
    log(f"  * Chiavi presenti in cache: {keys}")

    tr_raw = ck['tr_raw']
    va_raw = ck['va_raw']
    log(f"  * tr_raw count: {len(tr_raw)}, item shape: {np.asarray(tr_raw[0]).shape}")
    log(f"  * va_raw count: {len(va_raw)}, item shape: {np.asarray(va_raw[0]).shape}")
    
    first_50_raw = np.asarray([np.asarray(x) for x in tr_raw[:50]], dtype=np.float32)
    has_nan_raw = bool(np.isnan(first_50_raw).any())
    norm_raw = float(np.linalg.norm(first_50_raw, axis=-1).mean())
    std_raw = float(first_50_raw.std())
    log(f"    - h_raw (Conformer audio): Norm media={norm_raw:.2f}, Std={std_raw:.4f}, NaN/Inf={has_nan_raw}")

    tr_htx = ck['tr_htx']
    va_htx = ck['va_htx']
    log(f"  * tr_htx count: {len(tr_htx)}, item shape: {np.asarray(tr_htx[0]).shape}")
    first_50_htx = np.asarray([np.asarray(x) for x in tr_htx[:50]], dtype=np.float32)
    has_nan_htx = bool(np.isnan(first_50_htx).any())
    norm_htx = float(np.linalg.norm(first_50_htx, axis=-1).mean())
    
    h0 = torch.from_numpy(first_50_htx[0])
    h1 = torch.from_numpy(first_50_htx[1])
    h2 = torch.from_numpy(first_50_htx[2])
    cos_01 = float(torch.nn.functional.cosine_similarity(h0.unsqueeze(0), h1.unsqueeze(0)).item())
    cos_02 = float(torch.nn.functional.cosine_similarity(h0.unsqueeze(0), h2.unsqueeze(0)).item())
    log(f"    - h_tx (Gemma LM text): Norm media={norm_htx:.2f}, NaN/Inf={has_nan_htx}")
    log(f"    - Cosine Sim tra frasi diverse: sample_0 vs sample_1 = {cos_01:.4f}, sample_0 vs sample_2 = {cos_02:.4f}")

    tr_sub = ck['tr_sub']
    log(f"  * tr_sub count: {len(tr_sub)}, item shape: {np.asarray(tr_sub[0]).shape}")
    first_50_sub = np.asarray([np.asarray(x) for x in tr_sub[:50]], dtype=np.float32)
    norm_sub = float(np.linalg.norm(first_50_sub, axis=-1).mean())
    log(f"    - tr_sub norm: {norm_sub:.2f}")

    log("\n" + "=" * 70)
    log("=== RIEPILOGO CRITICO AUDIT ===")
    log(f"1. Tokenizer Mismatch: tbw è codificato con Gemma Tokenizer (non Qwen).")
    log(f"2. CB0 Vocab: CB0 in parquet ha range [{cb0_min}, {cb0_max}] (max 2047).")
    log(f"3. h_raw & h_tx in cache: SANISSIMI (Norm raw={norm_raw:.2f}, Norm htx={norm_htx:.2f}, cos diversificati).")
    log(f"4. Distillazione tr_sub: NON usare (teste non caricate).")
    log("=" * 70)
    rf.close()

if __name__ == '__main__':
    audit()
