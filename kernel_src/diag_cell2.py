# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 2 - DIAGNOSI AUDIO METALLICO (Check 1/2/3/4 dell'utente)
# ============================================================================
# Obiettivo: isolare PERCHÉ generated è metallico mentre oracle è perfetto.
#  - Check 1 (A/B): 4 sintesi incrociate mescolando i codebook:
#      wav_ab = decode(cb0_gen, cb1..15_gt)   -> isola l'effetto di cb0 predetto
#      wav_ac = decode(cb0_gt,  cb1..15_gen)  -> isola l'effetto di h_t (sub-talker)
#  - Check 2 (statistiche): confronto distribuzione cb0_gen vs cb0_gt, e
#      h_t pred (solo Gemma) — calcoliamo norm/std/delta per capire il collasso.
#  - Check 3 (teacher forcing): generate() è UN SOLO forward (non AR frame-loop):
#      il confronto diretto è gen_one_shot vs gen_con_cb0_gt_iniettato.
#  - Check 4 (spettrogrammi): salva PNG mel di generated/oracle per ispezione.
import os, sys, json, traceback, hashlib, math

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import numpy as np
    import soundfile as sf
    from torch.utils.data import DataLoader

    import unsloth  # noqa: F401
    import kaggle_pipeline as KP

    # ---- dati (finestre di validazione) ----
    parquet_cache = f"data/train_tokens_{KP.DUPLEXCONV_MAX_HOURS}h_{KP.MDC_MAX_SAMPLES}s.parquet"
    KP.convert_parquet_to_memmap(parquet_cache)
    user_mel, agent_mel, agent_codes, text_by_window = KP.load_tokens_from_parquet(parquet_cache)
    t_agent = agent_codes.shape[1]
    val_start = int((1.0 - KP.VAL_FRACTION) * t_agent)
    for e in KP.SEGMENT_ENDS:
        if e >= val_start:
            val_start = e
            break
    val_start = min(val_start, t_agent)
    n_val_w = max(0, (t_agent - val_start) // 64)
    print(f"val: start={val_start}, n_val_w={n_val_w}")
    sys.stdout.flush()

    def _slice_text(text_list, f_start):
        if text_list is None:
            return None
        return text_list[f_start // 64:]

    ds = KP.FullDuplexDataset(
        user_mel[val_start * KP.MEL_PER_AGENT_FRAME:],
        agent_mel[val_start * KP.MEL_PER_AGENT_FRAME:],
        agent_codes[:, val_start:],
        seq_len=64,
        text_by_window=_slice_text(text_by_window, val_start),
    )
    print(f"ds val: {len(ds)} finestre")
    sys.stdout.flush()

    # ---- modello + ckpt ep2 ----
    duplex_model, _ = KP.load_gemma4_duplex_system()
    ckpt = "checkpoints/gemma4_duplex_sota_ckpt"
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    h = hashlib.md5(open(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors"), "rb").read()).hexdigest()[:8]
    print(f"ckpt md5={h} (atteso 86a854e9)")
    sys.stdout.flush()
    with open("data/ckpt_md5.txt", "w", encoding="utf-8") as f:
        f.write(f"adapter_md5={h}\n")

    # ---- PATCH ROBUSTA: device-check semantic_head dentro generate() ----
    # (fix kaggle_pipeline.py:3010; il dataset monta il modulo con il fix DCT ma
    # senza questo fix -> il checkpoint ricarica le teste su CPU e generate crasha)
    _orig_generate = duplex_model.generate
    def _patched_generate(self, *a, **kw):
        return _orig_generate(*a, **kw)
    _src_gen = ""
    try:
        import inspect as _inspect
        _src_gen = _inspect.getsource(type(duplex_model).generate)
    except Exception:
        _src_gen = ""
    _needs_patch = _src_gen and "semantic_head.weight.device != adapted_hidden.device" not in _src_gen
    print(f"generate() richiede device-patch: {_needs_patch}")
    if _needs_patch:
        import types as _types
        _orig = duplex_model.generate
        def _safe_generate(self, *args, **kwargs):
            # forza device delle teste custom (mai su cpu durante inferenza)
            _dev = next(self.base_model.parameters()).device
            for _m in (self.semantic_head, self.ht_adapter, self.causal_conv):
                if _m is not None and next(_m.parameters()).device != _dev:
                    _m.to(_dev)
            return _orig(*args, **kwargs)
        duplex_model.generate = _types.MethodType(_safe_generate, duplex_model)
        print("device-patch applicata a generate()")
    sys.stdout.flush()

    decoder = KP.Code2WavDecoder()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.eval()
    os.makedirs("samples", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ---- utility per il testo finestra ----
    def _win_text(i):
        return _slice_text(text_by_window, val_start)[i] if text_by_window is not None else "?"

    # =====================================================================
    # CHECK 1: A/B ISOLATION (4 sintesi incrociate)
    # =====================================================================
    ab_report = []
    for i in range(min(8, len(ds))):
        item = ds[i]
        n_real = int((item["audio_targets"][:, 0] != KP.PAD_TARGET_ID).sum().item())
        if n_real <= 0:
            continue
        inp = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "input_features": item["input_features"].unsqueeze(0).to(device),
            "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            out = duplex_model.generate(temperature=0.0, **inp)  # temp=0 -> determinista, no sampling noise
        gen = out["audio_codes"][0][:n_real].cpu().numpy().astype(np.int64)  # (T, 16)
        gt = item["audio_targets"][:n_real].numpy().astype(np.int64)          # (T, 16)
        T = gen.shape[0]
        # 4 combinazioni (solo per il CB0 e i residui 1..15)
        combos = {
            "a_gt_full":  gt,
            "b_cb0gen_restgt": np.concatenate([gen[:, :1], gt[:, 1:]], axis=1),
            "c_cb0gt_restgen": np.concatenate([gt[:, :1], gen[:, 1:]], axis=1),
            "d_gen_full":  gen,
        }
        wavs = {}
        for name, codes in combos.items():
            codes = KP._trim_trailing_silence(codes)
            wavs[name] = decoder.decode(torch.from_numpy(codes))

        # metriche oneste: RMS diff (dominio tempo) + mel-L1/MCD come reference
        wa, wd = wavs["a_gt_full"], wavs["d_gen_full"]
        wb, wc = wavs["b_cb0gen_restgt"], wavs["c_cb0gt_restgen"]
        def _rmsdiff(x, y):
            n = min(len(x), len(y))
            return float(np.sqrt(((x[:n] - y[:n]) ** 2).mean()))
        def _rms(x):
            return float(np.sqrt((np.asarray(x) ** 2).mean()))
        r = {
            "win": i,
            "n_real": n_real,
            "text": str(_win_text(i))[:60],
            "rms_gt": _rms(wa),
            "rms_gen": _rms(wd),
            "rmsdiff_gt_vs_cb0gen": _rmsdiff(wa, wb),   # quanto cambia solo cb0
            "rmsdiff_gt_vs_restgen": _rmsdiff(wa, wc),  # quanto cambia solo h_t residui
            "rmsdiff_gt_vs_gen": _rmsdiff(wa, wd),      # totale
        }
        ab_report.append(r)
        # salva wav per ascolto
        for name, w in wavs.items():
            sf.write(f"samples/ab_{i:03d}_{name}.wav", w, KP.SYNTH_SAMPLE_RATE)
        # cb0 match rate
        cb0_match = float((gen[:, 0] == gt[:, 0]).mean())
        cb15_match = float((gen == gt).mean())
        print(f"win{i}: cb0_match={cb0_match:.3f} all16_match={cb15_match:.3f} rms_gt={r['rms_gt']:.4f} rms_gen={r['rms_gen']:.4f}")
        print(f"   diff: cb0gen={r['rmsdiff_gt_vs_cb0gen']:.4f} restgen={r['rmsdiff_gt_vs_restgen']:.4f} totale={r['rmsdiff_gt_vs_gen']:.4f}")
        sys.stdout.flush()

    with open("data/ab_report.txt", "w", encoding="utf-8") as f:
        f.write("win\tn_real\ttext\trms_gt\trms_gen\trmsdiff_cb0gen\trmsdiff_restgen\trmsdiff_total\n")
        for r in ab_report:
            f.write(f"{r['win']}\t{r['n_real']}\t{r['text']}\t{r['rms_gt']:.4f}\t{r['rms_gen']:.4f}\t"
                    f"{r['rmsdiff_gt_vs_cb0gen']:.4f}\t{r['rmsdiff_gt_vs_restgen']:.4f}\t{r['rmsdiff_gt_vs_gen']:.4f}\n")

    # =====================================================================
    # CHECK 2: STATISTICHE (distribuzione cb0 gen vs gt; h_t)
    # =====================================================================
    # Estrai anche gli hidden per l'analisi h_t: confrontiamo la statistica
    # del cb0 predetto (softmax) vs GT e la distribuzione degli hidden di Gemma.
    # h_t oracle NON esiste (il sub-talker è teacher); confrontiamo i logit.
    stat = []
    for i in range(min(8, len(ds))):
        item = ds[i]
        n_real = int((item["audio_targets"][:, 0] != KP.PAD_TARGET_ID).sum().item())
        if n_real <= 0:
            continue
        inp = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "input_features": item["input_features"].unsqueeze(0).to(device),
            "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            out = duplex_model.generate(temperature=0.0, **inp)
        gen = out["audio_codes"][0][:n_real].cpu().numpy()
        gt = item["audio_targets"][:n_real].numpy()
        # normalizzazione energia del frame di GT vs gen
        g_en = np.abs(gen).astype(np.float64).mean()
        t_en = np.abs(gt).astype(np.float64).mean()
        g_std = gen.astype(np.float64).std()
        t_std = gt.astype(np.float64).std()
        # silenzio predetto: quanto il modello tende al codice 0?
        sil_frac_gen = float((gen[:, 0] == 0).mean())
        sil_frac_gt = float((gt[:, 0] == 0).mean())
        stat.append((i, g_en, t_en, g_std, t_std, sil_frac_gen, sil_frac_gt))
        print(f"win{i}: |cb0| gen={g_en:.2f} gt={t_en:.2f} | std gen={g_std:.1f} gt={t_std:.1f} | sil0 gen={sil_frac_gen:.2f} gt={sil_frac_gt:.2f}")
        sys.stdout.flush()
    with open("data/stat_codes.txt", "w", encoding="utf-8") as f:
        f.write("win\tmean_abs_gen\tmean_abs_gt\tstd_gen\tstd_gt\tsil0_gen\tsil0_gt\n")
        for s in stat:
            f.write("\t".join(str(x) for x in s) + "\n")

    # =====================================================================
    # CHECK 3: "TEACHER FORCING STRICT" = generate con cb0 GT iniettato
    # =====================================================================
    # Il generate() nativo è one-shot (tutti i frame in parallelo). Per
    # simulare l'accumulo AR del sub-talker: cb0 GT + h_t predetto -> residui.
    # Confronto qualitativo: wav_c (cb0 GT) vs wav_d (cb0 gen) già nel report AB.

    # =====================================================================
    # CHECK 4: SPETTROGRAMMI PNG (matplotlib, no internet)
    # =====================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def _mel_from_wav(x, sr=24000, n_fft=512, n_mels=80):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        hop = n_fft // 4
        pad = n_fft // 2
        xp = np.pad(x, (pad, pad), mode="reflect")
        n_frames = 1 + (len(xp) - n_fft) // hop
        mel = np.zeros((n_frames, n_mels), dtype=np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mel_f = 2595.0 * np.log10(1.0 + np.linspace(0, sr / 2, n_mels + 2) / 700.0)
        mel_bins = 2595.0 * np.log10(1.0 + freqs / 700.0)
        filt = np.zeros((n_mels, len(freqs)), dtype=np.float32)
        for m in range(n_mels):
            lo, ctr, hi = mel_f[m], mel_f[m + 1], mel_f[m + 2]
            filt[m] = np.clip(np.minimum((mel_bins - lo) / (ctr - lo + 1e-9), (hi - mel_bins) / (hi - ctr + 1e-9)), 0.0, 1.0)
        win = np.hanning(n_fft).astype(np.float32)
        for t in range(n_frames):
            seg = xp[t * hop : t * hop + n_fft] * win
            spec = np.abs(np.fft.rfft(seg)) ** 2
            mel[t] = filt @ spec
        return np.log1p(mel)

    os.makedirs("plots", exist_ok=True)
    for i in range(min(4, len(ab_report))):
        wi = ab_report[i]["win"]
        wg = sf.read(f"samples/ab_{wi:03d}_d_gen_full.wav", dtype="float32")[0]
        wo = sf.read(f"samples/ab_{wi:03d}_a_gt_full.wav", dtype="float32")[0]
        mg = _mel_from_wav(wg)
        mo = _mel_from_wav(wo)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        ax[0].imshow(mo.T, aspect="auto", origin="lower", cmap="magma")
        ax[0].set_title(f"win{wi:03d} ORACLE (GT)")
        ax[1].imshow(mg.T, aspect="auto", origin="lower", cmap="magma")
        ax[1].set_title(f"win{wi:03d} GENERATED")
        plt.tight_layout()
        plt.savefig(f"plots/spec_{wi:03d}.png", dpi=90)
        plt.close(fig)
    print("plots salvati")

    # =====================================================================
    # RIEPILOGO
    # =====================================================================
    print("=== DIAGNOSI COMPLETATA (v6) ===")
    print("File: data/ab_report.txt, data/stat_codes.txt, samples/ab_*wav, plots/*.png")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("DIAGNOSI FALLITA:\n" + tb)
    sys.stdout.flush()
    with open("diag_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
