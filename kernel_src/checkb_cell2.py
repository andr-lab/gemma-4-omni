# -*- coding: utf-8 -*-
# ============================================================================
# CHECK-B (post-v14): discriminare H1 (covariate shift train/val) vs
# H2/H3 (il modello non ha MAI imparato il parlato -> distillazione/capacita').
# Misura la CE parlato per-frame in TEACHER FORCING su N batch TRAIN e
# N batch VAL, con la STESSA definizione di frame parlato del training:
#   real  = (tgt != PAD).all(dim=-1)   [PAD = -100]
#   sil   = (tgt == 0).all(dim=-1)     [silenzio = 16 codebook a 0]
#   speech = real & (~sil)
# Verdetto automatico:
#   CE_speech_train << CE_speech_val  (~2 vs ~7)  -> H1 covariate shift
#   CE_speech_train ~ CE_speech_val ~ 7           -> H2/H3 mai imparato
# ============================================================================
import os, sys, json, traceback, hashlib, math

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import torch.nn.functional as F
    import numpy as np

    import unsloth  # noqa: F401
    import kaggle_pipeline as KP

    N_TRAIN = int(os.environ.get("CHECKB_N_TRAIN", "8"))
    N_VAL = int(os.environ.get("CHECKB_N_VAL", "8"))

    # ---- dati (stessa costruzione di fix_cell2: val = ultimo VAL_FRACTION) ----
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
    print(f"split: train=[0,{val_start}) | val=[{val_start},{t_agent}) n_val_w={n_val_w}")
    sys.stdout.flush()

    def _slice_text(text_list, f_start):
        if text_list is None:
            return None
        return text_list[f_start // 64:]

    MPF = KP.MEL_PER_AGENT_FRAME

    def _make_ds(lo, hi):
        return KP.FullDuplexDataset(
            user_mel[lo * MPF:hi * MPF],
            agent_mel[lo * MPF:hi * MPF],
            agent_codes[:, lo:hi],
            seq_len=64,
            text_by_window=_slice_text(text_by_window, lo),
        )

    ds_train = _make_ds(0, val_start)
    ds_val = _make_ds(val_start, t_agent)
    print(f"ds_train: {len(ds_train)} finestre | ds_val: {len(ds_val)} finestre")
    sys.stdout.flush()

    # ---- modello + ckpt ep2 (86a854e9) ----
    duplex_model, _ = KP.load_gemma4_duplex_system()
    ckpt = "checkpoints/gemma4_duplex_sota_ckpt"
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    h = hashlib.md5(open(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors"), "rb").read()).hexdigest()[:8]
    print(f"ckpt md5(adapter)={h} (atteso 86a854e9)")
    device = next(duplex_model.base_model.parameters()).device

    # ---- device-patch (stessa di fix_cell2) ----
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

    PAD = KP.PAD_TARGET_ID
    fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
    audio_feat_start = 2 + KP.TEXT_BLOCK_LEN
    sub = duplex_model.subtalker
    V = sub.cfg["vocab_size"]

    def _extract_h_t(hidden_states, n_agent):
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
    # Misura CE per-frame (teacher forcing) su uno split
    # =====================================================================
    def eval_split(ds, tag, n_win):
        agg = {"n_win": 0, "n_real": 0, "n_speech": 0, "n_sil": 0,
               "ce_all_sum": 0.0, "ce_speech_sum": 0.0, "ce_sil_sum": 0.0,
               "cb_speech": {f"cb{g}": 0.0 for g in ("1..3", "4..8", "9..15")},
               "n_cb_speech": 0}
        rows = []
        n_scanned = 0
        for i in range(len(ds)):
            if n_scanned >= n_win:
                break
            item = ds[i]
            n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
            if n_real <= 0:
                continue
            n_scanned += 1
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
                h_t = _extract_h_t(hidden_states, n_real)
                gt = item["audio_targets"][:n_real].to(device)  # (T, 16)
                sub_logits = sub.forward_sub_talker_finetune(h_t, gt.unsqueeze(0))
                per = F.cross_entropy(
                    sub_logits.reshape(-1, V),
                    gt[:, 1:].reshape(-1),
                    reduction="none",
                    ignore_index=PAD,
                ).reshape(n_real, 15)
            # frame parlato: stessa definizione del monkeypatch v14
            real = (gt != PAD).all(dim=-1)          # (T,)
            sil = (gt == 0).all(dim=-1)             # silenzio: 16 codebook a 0
            speech = real & (~sil)
            ce_all = per.mean().item()
            ce_sp = per[speech].mean().item() if speech.any() else float("nan")
            ce_si = per[~speech & real].mean().item() if (~speech & real).any() else float("nan")
            n_sp = int(speech.sum().item())
            rows.append({"win": i, "n_real": n_real, "n_speech": n_sp,
                         "ce_all": ce_all, "ce_speech": ce_sp, "ce_sil": ce_si,
                         "cb_speech": {
                             "cb1..3": float(per[speech][:, :3].mean()) if n_sp else float("nan"),
                             "cb4..8": float(per[speech][:, 4:9].mean()) if n_sp else float("nan"),
                             "cb9..15": float(per[speech][:, 9:].mean()) if n_sp else float("nan"),
                         }})
            agg["n_win"] += 1
            agg["n_real"] += n_real
            agg["n_speech"] += n_sp
            agg["n_sil"] += int((~speech & real).sum().item())
            if not math.isnan(ce_all):
                agg["ce_all_sum"] += ce_all
            if not math.isnan(ce_sp):
                agg["ce_speech_sum"] += ce_sp
                agg["n_cb_speech"] += 1
            if not math.isnan(ce_si):
                agg["ce_sil_sum"] += ce_si
            for g in ("cb1..3", "cb4..8", "cb9..15"):
                v = rows[-1]["cb_speech"][g]
                if not math.isnan(v):
                    agg["cb_speech"][g] += v
            print(f"[{tag}] win{i}: n_real={n_real} n_speech={n_sp} | "
                  f"CE all={ce_all:.3f} speech={ce_sp:.3f} sil={ce_si:.3f} | "
                  f"cb1-3={rows[-1]['cb_speech']['cb1..3']:.3f} cb4-8={rows[-1]['cb_speech']['cb4..8']:.3f} "
                  f"cb9-15={rows[-1]['cb_speech']['cb9..15']:.3f}")
            sys.stdout.flush()
        n_ws = max(agg["n_win"], 1)
        n_cb = max(agg["n_cb_speech"], 1)
        res = {
            "split": tag,
            "n_win": agg["n_win"],
            "n_real": agg["n_real"],
            "n_speech": agg["n_speech"],
            "speech_frac": agg["n_speech"] / max(agg["n_real"], 1),
            "ce_all_mean": agg["ce_all_sum"] / n_ws,
            "ce_speech_mean": agg["ce_speech_sum"] / n_cb,
            "ce_sil_mean": agg["ce_sil_sum"] / max(agg["n_win"], 1),
            "ce_speech_cb1_3": agg["cb_speech"]["cb1..3"] / n_cb,
            "ce_speech_cb4_8": agg["cb_speech"]["cb4..8"] / n_cb,
            "ce_speech_cb9_15": agg["cb_speech"]["cb9..15"] / n_cb,
        }
        return res, rows

    r_train, rows_train = eval_split(ds_train, "TRAIN", N_TRAIN)
    print("=== VAL SPLIT ===", flush=True)
    r_val, rows_val = eval_split(ds_val, "VAL", N_VAL)

    # =====================================================================
    # Verdetto
    # =====================================================================
    ce_tr = r_train["ce_speech_mean"]
    ce_va = r_val["ce_speech_mean"]
    frac_tr = r_train["speech_frac"]
    frac_va = r_val["speech_frac"]
    if not math.isnan(ce_tr) and not math.isnan(ce_va):
        if ce_tr < 4.0 and ce_va >= 6.0:
            verdict = ("H1 COVARIATE SHIFT: il modello predice il parlato in TRAIN "
                       "(CE %.2f) ma NON in VAL (CE %.2f). Il gap e' train/val, "
                       "non gradiente annegato. Fix: val split stratificato/random "
                       "+ ri-train; verificare anche shift di frazione parlato "
                       "(train %.1f%% vs val %.1f%%)." % (ce_tr, ce_va, 100*frac_tr, 100*frac_va))
        elif ce_tr >= 6.0 and ce_va >= 6.0:
            verdict = ("H2/H3 MAI IMPARATO: CE parlato alta ANCHE in train (%.2f) e val (%.2f). "
                       "Il modello non ha mai appreso a predire il parlato -> serve "
                       "distillazione h_t dal teacher Qwen (FitNets) o piu' capacita' "
                       "adapter (MLP/LoRA)." % (ce_tr, ce_va))
        elif ce_tr < 4.0 and ce_va < 4.0:
            verdict = ("MODEL OK IN TF: CE parlato bassa sia in train (%.2f) che val (%.2f). "
                       "Il problema e' a valle del teacher forcing (sampling/exposure "
                       "bias o decoder) -> Check C2 (autoregressivo con codici GT "
                       "sintetici) e/o sampling." % (ce_tr, ce_va))
        else:
            verdict = ("CASO INTERMEDIO: train=%.2f val=%.2f. Servono piu' dati (aumentare N)."
                       % (ce_tr, ce_va))
    else:
        verdict = f"DATI INSUFFICIENTI: ce_tr={ce_tr} ce_va={ce_va} (nan per mancanza di frame parlati?)"

    print("\n" + "=" * 70)
    print("VERDETTO CHECK-B:", verdict)
    print("=" * 70)
    print(json.dumps({"train": r_train, "val": r_val}, indent=2))
    sys.stdout.flush()

    with open("data/checkb_report.txt", "w", encoding="utf-8") as f:
        f.write("CHECK-B: CE parlato teacher-forcing, train vs val\n")
        f.write("ckpt md5(adapter) = %s\n\n" % h)
        f.write("VERDETTO: %s\n\n" % verdict)
        f.write("TRAIN\n")
        f.write(json.dumps(r_train, indent=2) + "\n\n")
        f.write("VAL\n")
        f.write(json.dumps(r_val, indent=2) + "\n\n")
        f.write("RIGHE TRAIN (win\tn_real\tn_speech\tce_all\tce_speech\tce_sil\n")
        for r in rows_train:
            f.write(f"{r['win']}\t{r['n_real']}\t{r['n_speech']}\t{r['ce_all']:.4f}\t"
                    f"{r['ce_speech']:.4f}\t{r['ce_sil']:.4f}\n")
        f.write("\nRIGHE VAL\n")
        for r in rows_val:
            f.write(f"{r['win']}\t{r['n_real']}\t{r['n_speech']}\t{r['ce_all']:.4f}\t"
                    f"{r['ce_speech']:.4f}\t{r['ce_sil']:.4f}\n")
    print("=== CHECK-B REPORT SALVATO: data/checkb_report.txt ===")
    print("=== CHECK-B COMPLETATO ===")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("CHECK-B FALLITA:\n" + tb)
    sys.stdout.flush()
    with open("checkb_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
