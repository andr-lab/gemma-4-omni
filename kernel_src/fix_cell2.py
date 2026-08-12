# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 2 - FIX RAPIDO v11: perché i residui 1..15 sono scadenti?
# ============================================================================
# v11: fix bug v10 - (1) device-check esplicito ht_adapter/subtalker (il
# checkpoint li ricarica su CPU, generate() li sposta ma il percorso manuale
# no -> RuntimeError mat2 on cpu); (2) allineamento n_agent = n_real come nel
# forward di training (n_agent = audio_targets.shape[1]); (3) cb0 2D (1, T).
#
#  - Check A (CE per-frame): teacher-forcing CE del sub-talker, SPLIT per
#      frame parlato/silenzio -> discrimina ipotesi 1 (h_t fuori manifold,
#      CE ALTA anche in TF) vs exposure bias (CE BASSA in TF, sampling scadente).
#  - Check B (match rate per codebook): gen vs GT a temp=0, per colonna.
#      Il metallico vive nei codebook alti (cb9..15 = dettaglio HF)?
#  - Check C (fix LayerNorm su h_t): h_t normalizzato (layer_norm) prima del
#      sub-talker -> test a costo zero dell'ipotesi "scale mismatch h_t".
#  - Check D (metriche oneste): RMS ratio + correlazione per ogni configurazione.
#  - Output: report txt + wav (oracle / baseline / lrn) + plot mel.
import os, sys, json, traceback, hashlib, math

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import torch.nn.functional as F
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
    print(f"ckpt md5(adapter)={h} (atteso 86a854e9)")
    device = next(duplex_model.base_model.parameters()).device

    # ---- PATCH ROBUSTA: device-check teste custom dentro generate() ----
    # (fix kaggle_pipeline.py:3010; il dataset monta il modulo senza questo fix
    # -> il checkpoint ricarica le teste su CPU e generate() crasha, bug v12)
    import inspect as _inspect
    import types as _types
    _src_gen = ""
    try:
        _src_gen = _inspect.getsource(type(duplex_model).generate)
    except Exception:
        _src_gen = ""
    _needs_patch = _src_gen and "semantic_head.weight.device != adapted_hidden.device" not in _src_gen
    print(f"generate() richiede device-patch: {_needs_patch}")
    if _needs_patch:
        _orig = duplex_model.generate
        def _safe_generate(self, *args, **kwargs):
            _dev = next(self.base_model.parameters()).device
            for _m in (self.semantic_head, self.ht_adapter, self.causal_conv):
                if _m is not None and next(_m.parameters()).device != _dev:
                    _m.to(_dev)
            return _orig(*args, **kwargs)
        duplex_model.generate = _types.MethodType(_safe_generate, duplex_model)
        print("device-patch applicata a generate()")
    sys.stdout.flush()

    duplex_model.eval()

    decoder = KP.Code2WavDecoder()
    os.makedirs("samples", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    PAD = KP.PAD_TARGET_ID
    SAMPLE_RATE = KP.SYNTH_SAMPLE_RATE
    FRAME_SAMPLES = int(round(SAMPLE_RATE / KP.FRAME_RATE_HZ))  # 1920 @24kHz per 80ms frame

    def _rms(x):
        x = np.asarray(x, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    def _frame_active_mask(wav, n_real):
        """frame parlato = energia RMS del wav GT nel blocco > soglia relativa."""
        n = min(len(wav), n_real * FRAME_SAMPLES)
        w = np.asarray(wav[:n], dtype=np.float32)
        nf = n // FRAME_SAMPLES
        if nf == 0:
            return np.zeros(n_real, dtype=bool), np.zeros(n_real, dtype=np.float32)
        rms_per = np.array([_rms(w[t * FRAME_SAMPLES:(t + 1) * FRAME_SAMPLES]) for t in range(nf)])
        thr = max(0.02, 0.25 * (rms_per.max() if rms_per.max() > 0 else 1.0))
        mask = np.zeros(n_real, dtype=bool)
        mask[:nf] = rms_per > thr
        return mask, rms_per

    # =====================================================================
    # Percorso h_t condiviso (stessa slice di forward()/generate())
    # =====================================================================
    fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
    audio_feat_start = 2 + KP.TEXT_BLOCK_LEN
    sub = duplex_model.subtalker
    V = sub.cfg["vocab_size"]

    def _extract_h_t(hidden_states, n_agent):
        """hidden_states (B, L, H) -> h_t (B, n_agent, talker_hidden), con
        device-check delle teste custom (come generate(): il checkpoint le
        ricarica su CPU)."""
        agent_hidden = hidden_states[
            :, audio_feat_start + fpf - 1 : audio_feat_start + fpf * n_agent : fpf, :
        ]
        adapted_hidden = duplex_model._apply_causal_conv(agent_hidden)
        if next(duplex_model.ht_adapter.parameters()).device != adapted_hidden.device:
            duplex_model.ht_adapter.to(adapted_hidden.device)
        if next(sub.parameters()).device != adapted_hidden.device:
            sub.to(adapted_hidden.device)
        ad_dtype = next(duplex_model.ht_adapter.parameters()).dtype
        h_t = duplex_model.ht_adapter(adapted_hidden.to(ad_dtype))
        return h_t.to(next(sub.parameters()).dtype)

    # =====================================================================
    # CHECK A: CE per-frame (teacher forcing) parlato vs silenzio
    # =====================================================================
    a_report = []
    for i in range(min(8, len(ds))):
        item = ds[i]
        n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
        if n_real <= 0:
            continue
        inp = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "input_features": item["input_features"].unsqueeze(0).to(device),
            "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            outputs = duplex_model.base_model(
                **inp, output_hidden_states=True, return_dict=True, logits_to_keep=1
            )
            hidden_states = outputs.hidden_states[-1]
            # n_agent = n_real (allineamento al training: audio_targets.shape[1])
            h_t = _extract_h_t(hidden_states, n_real)
            gt = item["audio_targets"][:n_real].to(device)  # (T, 16)
            sub_logits = sub.forward_sub_talker_finetune(h_t, gt.unsqueeze(0))
            per = F.cross_entropy(
                sub_logits.reshape(-1, V),
                gt[:, 1:].reshape(-1),
                reduction="none",
                ignore_index=PAD,
            ).reshape(n_real, 15)
            # h_t stats
            ht_stats = {
                "mean": float(h_t[0].mean()),
                "std": float(h_t[0].std()),
                "norm": float(h_t[0].norm() / math.sqrt(h_t.shape[-1])),
                "dmean": float((h_t[0, 1:] - h_t[0, :-1]).abs().mean()),
            }
        # frame parlato da energia del wav GT
        wav_gt = decoder.decode(torch.from_numpy(KP._trim_trailing_silence(gt.cpu().numpy())))
        active, rms_per = _frame_active_mask(wav_gt, n_real)
        ce_all = per.mean().item()
        ce_act = per[active].mean().item() if active.any() else float("nan")
        ce_sil = per[~active].mean().item() if (~active).any() else float("nan")
        ce_by_cb = {
            "cb1..3": float(per[:, :3].mean()),
            "cb4..8": float(per[:, 4:9].mean()),
            "cb9..15": float(per[:, 9:].mean()),
        }
        a_report.append({
            "win": i, "n_real": n_real, "n_act": int(active.sum()),
            "ce_all": ce_all, "ce_act": ce_act, "ce_sil": ce_sil,
            "ce_cb13": ce_by_cb["cb1..3"], "ce_cb48": ce_by_cb["cb4..8"], "ce_cb915": ce_by_cb["cb9..15"],
            "ht_mean": ht_stats["mean"], "ht_std": ht_stats["std"],
            "ht_norm": ht_stats["norm"], "ht_dmean": ht_stats["dmean"],
        })
        print(f"[A] win{i}: n_act={active.sum()}/{n_real} CE all={ce_all:.3f} act={ce_act:.3f} sil={ce_sil:.3f} "
              f"| cb1-3={ce_by_cb['cb1..3']:.3f} cb4-8={ce_by_cb['cb4..8']:.3f} cb9-15={ce_by_cb['cb9..15']:.3f}")
        sys.stdout.flush()

    # =====================================================================
    # CHECK B/C/D: match per codebook + fix LayerNorm + metriche oneste
    # =====================================================================
    b_report = []
    for i in range(min(4, len(ds))):
        item = ds[i]
        n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
        if n_real <= 0:
            continue
        inp = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "input_features": item["input_features"].unsqueeze(0).to(device),
            "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
        }
        gt = item["audio_targets"][:n_real].numpy().astype(np.int64)
        with torch.inference_mode():
            out = duplex_model.generate(temperature=0.0, **inp)
            gen = out["audio_codes"][0][:n_real].cpu().numpy().astype(np.int64)  # (T, 16)
            cb0_sampled = gen[:, 0]  # (T,)

            # h_t per il fix LayerNorm (replica percorso, allineato a n_real)
            outputs = duplex_model.base_model(
                **inp, output_hidden_states=True, return_dict=True, logits_to_keep=1
            )
            hidden_states = outputs.hidden_states[-1]
            h_t = _extract_h_t(hidden_states, n_real)
            # Fix LayerNorm: h_t normalizzato, stessi cb0
            h_t_lrn = F.layer_norm(h_t, (h_t.shape[-1],))
            cb0_t = torch.from_numpy(cb0_sampled).to(device).long().unsqueeze(0)  # (1, T)
            acust_lrn = sub.sample_acoustic_codes(h_t_lrn, cb0_t, temperature=0.0)
            gen_lrn = np.concatenate([cb0_sampled[:, None], acust_lrn[0].cpu().numpy()], axis=1)  # (T,16)

            combos = {
                "oracle": gt,
                "baseline": gen,
                "lrn": gen_lrn,
            }
            wavs = {}
            for name, codes in combos.items():
                codes = KP._trim_trailing_silence(codes)
                wavs[name] = decoder.decode(torch.from_numpy(codes))
            # metriche oneste
            def _corr(x, y):
                n = min(len(x), len(y))
                a, b = np.asarray(x[:n], np.float32), np.asarray(y[:n], np.float32)
                if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    return float("nan")
                return float(np.corrcoef(a, b)[0, 1])
            row = {"win": i}
            for name, w in wavs.items():
                row[f"rms_{name}"] = _rms(w)
                row[f"corr_{name}"] = _corr(w, wavs["oracle"])
                row[f"rmsratio_{name}"] = _rms(w) / max(_rms(wavs["oracle"]), 1e-9)
            # match rate per codebook
            mt = []
            for c in range(1, 16):
                mt.append(float((gen[:, c] == gt[:, c]).mean()))
            row["match_cb1..5"] = float(np.mean(mt[:5]))
            row["match_cb6..10"] = float(np.mean(mt[5:10]))
            row["match_cb11..15"] = float(np.mean(mt[10:]))
            b_report.append(row)
            print(f"[B] win{i}: rmsratio baseline={row['rmsratio_baseline']:.2f} lrn={row['rmsratio_lrn']:.2f} "
                  f"| corr baseline={row['corr_baseline']:.2f} lrn={row['corr_lrn']:.2f} "
                  f"| match cb1-5={row['match_cb1..5']:.3f} cb6-10={row['match_cb6..10']:.3f} cb11-15={row['match_cb11..15']:.3f}")
            # salva wav
            for name, w in wavs.items():
                sf.write(f"samples/fix_{i:03d}_{name}.wav", w, SAMPLE_RATE)
            # plot mel oracle vs baseline vs lrn
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(3, 1, figsize=(14, 9))
                for ax, (name, w) in zip(axes, [("oracle", wavs["oracle"]), ("baseline", wavs["baseline"]), ("lrn", wavs["lrn"])]):
                    if len(w) == 0:
                        continue
                    import librosa
                    m = librosa.feature.melspectrogram(y=np.asarray(w, np.float32), sr=SAMPLE_RATE, n_mels=80, fmax=12000)
                    ax.imshow(librosa.power_to_db(m, ref=np.max), aspect="auto", origin="lower", cmap="magma")
                    ax.set_title(f"win{i} {name}")
                plt.tight_layout()
                plt.savefig(f"plots/fix_{i:03d}_mel.png", dpi=80)
                plt.close(fig)
            except Exception as e:
                print(f"  plot win{i}: {str(e)[:80]}")
        sys.stdout.flush()

    # ---- salva report ----
    with open("data/fix_report.txt", "w", encoding="utf-8") as f:
        f.write("# CHECK A - CE teacher-forcing per-frame (parlato vs silenzio)\n")
        f.write("win\tn_real\tn_act\tce_all\tce_act\tce_sil\tce_cb1-3\tce_cb4-8\tce_cb9-15\tht_mean\tht_std\tht_norm\tht_dmean\n")
        for r in a_report:
            f.write(f"{r['win']}\t{r['n_real']}\t{r['n_act']}\t{r['ce_all']:.4f}\t{r['ce_act']:.4f}\t{r['ce_sil']:.4f}\t"
                    f"{r['ce_cb13']:.4f}\t{r['ce_cb48']:.4f}\t{r['ce_cb915']:.4f}\t"
                    f"{r['ht_mean']:.4f}\t{r['ht_std']:.4f}\t{r['ht_norm']:.4f}\t{r['ht_dmean']:.4f}\n")
        f.write("\n# CHECK B/C/D - fix LayerNorm + match + metriche oneste\n")
        keys = ["win", "rms_oracle", "rms_baseline", "rms_lrn", "rmsratio_baseline", "rmsratio_lrn",
                "corr_baseline", "corr_lrn", "match_cb1..5", "match_cb6..10", "match_cb11..15"]
        f.write("\t".join(keys) + "\n")
        for r in b_report:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print("=== FIX REPORT SALVATO: data/fix_report.txt ===")
    print("File: data/fix_report.txt, samples/fix_*.wav, plots/fix_*_mel.png")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("CELL2 FALLITA:\n" + tb)
    sys.stdout.flush()
    with open("fix_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
