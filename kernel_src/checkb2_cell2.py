# -*- coding: utf-8 -*-
# ============================================================================
# CHECK-B2 (post CHECK-B): maschera ENERGIA (come Check A v13) + CAPACITY PROBE
# Il CHECK-B ha rivelato che (tgt==0).all(-1) NON separa parlato/silenzio:
# il silenzio Qwen3-TTS non e' codificato come 16 codebook a 0. -> la maschera
# codici da' speech_frac=1.0 ovunque, inutile. Qui riusiamo la maschera RMS
# energia del wav GT (soglia relativa, come fix_cell2 v13) su TRAIN e VAL.
#
# PARTE 1 (CHECK-B2): CE parlato/silenzio per energia, train vs val.
# PARTE 2 (CAPACITY PROBE): fit SOLO ht_adapter su UNA finestra densa di
#   parlato (train, ~200 step Adam lr 1e-3). Se la CE parlato scende
#   nettamente (-> ~1-2) l'adapter PUO' rappresentare la mappatura e il
#   problema era l'addestramento (H1/gradiente). Se resta alta (~5-7)
#   l'adapter NON puo' proiettare h_t nel manifold (H2/H3) -> distillazione.
# ============================================================================
import os, sys, json, traceback, hashlib, math

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import torch.nn.functional as F
    import numpy as np
    import soundfile as sf

    import unsloth  # noqa: F401
    import kaggle_pipeline as KP

    N_TRAIN = int(os.environ.get("CHECKB2_N_TRAIN", "8"))
    N_VAL = int(os.environ.get("CHECKB2_N_VAL", "8"))
    PROBE_STEPS = int(os.environ.get("CHECKB2_PROBE_STEPS", "200"))
    PROBE_LR = float(os.environ.get("CHECKB2_PROBE_LR", "1e-3"))

    # ---- dati ----
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

    # ---- modello + ckpt ep2 ----
    duplex_model, _ = KP.load_gemma4_duplex_system()
    ckpt = "checkpoints/gemma4_duplex_sota_ckpt"
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    h = hashlib.md5(open(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors"), "rb").read()).hexdigest()[:8]
    print(f"ckpt md5(adapter)={h} (atteso 86a854e9)")
    device = next(duplex_model.base_model.parameters()).device

    # ---- device-patch ----
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
    SAMPLE_RATE = KP.SYNTH_SAMPLE_RATE
    FRAME_SAMPLES = int(round(SAMPLE_RATE / KP.FRAME_RATE_HZ))
    fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
    audio_feat_start = 2 + KP.TEXT_BLOCK_LEN
    sub = duplex_model.subtalker
    V = sub.cfg["vocab_size"]

    def _rms(x):
        x = np.asarray(x, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    def _frame_active_mask(wav, n_real):
        """frame parlato = energia RMS del wav GT nel blocco > soglia relativa
        (IDENTICA a fix_cell2 v13: thr = max(0.02, 0.25*max))."""
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

    decoder = KP.Code2WavDecoder()

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
    # PARTE 1: CE per-frame con maschera ENERGIA, train vs val
    # =====================================================================
    def eval_split_energy(ds, tag, n_win):
        agg = {"n_win": 0, "n_real": 0, "n_act": 0, "n_sil": 0,
               "ce_all_sum": 0.0, "ce_act_sum": 0.0, "ce_sil_sum": 0.0,
               "n_act_win": 0, "n_sil_win": 0,
               "cb_act": {f"cb{g}": 0.0 for g in ("1..3", "4..8", "9..15")},
               "n_cb_act": 0}
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
                gt = item["audio_targets"][:n_real].to(device)
                sub_logits = sub.forward_sub_talker_finetune(h_t, gt.unsqueeze(0))
                per = F.cross_entropy(
                    sub_logits.reshape(-1, V),
                    gt[:, 1:].reshape(-1),
                    reduction="none",
                    ignore_index=PAD,
                ).reshape(n_real, 15)
            wav_gt = decoder.decode(torch.from_numpy(KP._trim_trailing_silence(gt.cpu().numpy())))
            active, rms_per = _frame_active_mask(wav_gt, n_real)
            ce_all = per.mean().item()
            ce_act = per[active].mean().item() if active.any() else float("nan")
            ce_sil = per[~active].mean().item() if (~active).any() else float("nan")
            n_act = int(active.sum())
            rows.append({"win": i, "n_real": n_real, "n_act": n_act,
                         "ce_all": ce_all, "ce_act": ce_act, "ce_sil": ce_sil})
            agg["n_win"] += 1
            agg["n_real"] += n_real
            agg["n_act"] += n_act
            agg["n_sil"] += n_real - n_act
            if not math.isnan(ce_all):
                agg["ce_all_sum"] += ce_all
            if not math.isnan(ce_act):
                agg["ce_act_sum"] += ce_act
                agg["n_act_win"] += 1
            if not math.isnan(ce_sil):
                agg["ce_sil_sum"] += ce_sil
                agg["n_sil_win"] += 1
            for g, sl in (("cb1..3", slice(0, 3)), ("cb4..8", slice(4, 9)), ("cb9..15", slice(9, 15))):
                v = per[active][:, sl].mean() if n_act else float("nan")
                if not math.isnan(v):
                    agg["cb_act"][g] += float(v)
            if n_act:
                agg["n_cb_act"] += 1
            print(f"[{tag}] win{i}: n_act={n_act}/{n_real} | CE all={ce_all:.3f} act={ce_act:.3f} sil={ce_sil:.3f}")
            sys.stdout.flush()
        n_ws = max(agg["n_win"], 1)
        res = {
            "split": tag,
            "n_win": agg["n_win"],
            "n_real": agg["n_real"],
            "n_act": agg["n_act"],
            "speech_frac_energy": agg["n_act"] / max(agg["n_real"], 1),
            "ce_all_mean": agg["ce_all_sum"] / n_ws,
            "ce_act_mean": agg["ce_act_sum"] / max(agg["n_act_win"], 1),
            "ce_sil_mean": agg["ce_sil_sum"] / max(agg["n_sil_win"], 1),
            "ce_act_cb1_3": agg["cb_act"]["cb1..3"] / max(agg["n_cb_act"], 1),
            "ce_act_cb4_8": agg["cb_act"]["cb4..8"] / max(agg["n_cb_act"], 1),
            "ce_act_cb9_15": agg["cb_act"]["cb9..15"] / max(agg["n_cb_act"], 1),
        }
        return res, rows

    print("=== TRAIN SPLIT (maschera energia) ===", flush=True)
    r_train, rows_train = eval_split_energy(ds_train, "TRAIN", N_TRAIN)
    print("=== VAL SPLIT (maschera energia) ===", flush=True)
    r_val, rows_val = eval_split_energy(ds_val, "VAL", N_VAL)

    # =====================================================================
    # PARTE 2: CAPACITY PROBE - fit SOLO ht_adapter su una finestra densa
    # di parlato (train). Scopo: l'adapter PUO' rappresentare la mappatura?
    # =====================================================================
    print("\n=== CAPACITY PROBE (fit ht_adapter, %d step lr=%g) ===" % (PROBE_STEPS, PROBE_LR), flush=True)
    probe_out = {}
    # scegli la finestra train con frazione parlato piu' alta tra le prime N_TRAIN
    best = max(rows_train, key=lambda r: r["n_act"] / max(r["n_real"], 1))
    idx = best["win"]
    item = ds_train[idx]
    n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
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
        # h_t con adapter CONGELATO (pesi ep2) -> baseline
        h_t_base = _extract_h_t(hidden_states, n_real)
        gt = item["audio_targets"][:n_real].to(device)
        sub_logits_base = sub.forward_sub_talker_finetune(h_t_base, gt.unsqueeze(0))
        per_base = F.cross_entropy(
            sub_logits_base.reshape(-1, V), gt[:, 1:].reshape(-1),
            reduction="none", ignore_index=PAD,
        ).reshape(n_real, 15)
    wav_gt = decoder.decode(torch.from_numpy(KP._trim_trailing_silence(gt.cpu().numpy())))
    active, rms_per = _frame_active_mask(wav_gt, n_real)
    ce_act_base = float(per_base[active].mean()) if active.any() else float("nan")
    ce_all_base = float(per_base.mean())
    print(f"probe: finestra train win{idx} n_act={int(active.sum())}/{n_real} "
          f"| CE act BASE={ce_act_base:.3f} all BASE={ce_all_base:.3f}", flush=True)

    # adattatore addestrabile: SOLO ht_adapter; tutto il resto congelato
    adapter = duplex_model.ht_adapter
    for p in adapter.parameters():
        p.requires_grad = True
    # assicura che base_model/subtalker siano frozen (eval, no grad)
    opt = torch.optim.Adam(adapter.parameters(), lr=PROBE_LR)
    # ricalcola adapted_hidden UNA volta (base congelato); h_t = adapter(adapted_hidden)
    agent_hidden = hidden_states[
        :, audio_feat_start + fpf - 1 : audio_feat_start + fpf * n_real : fpf, :
    ]
    adapted_hidden = duplex_model._apply_causal_conv(agent_hidden)
    adapted_hidden = adapted_hidden.to(next(adapter.parameters()).dtype)
    adapted_hidden = adapted_hidden.detach()
    gt_d = gt  # (n_real, 16)
    targets = gt_d[:, 1:].reshape(-1).to(device)  # (n_real*15,)
    ce_trace = []
    for step in range(PROBE_STEPS):
        opt.zero_grad()
        h_t = adapter(adapted_hidden)                      # (1, n_real, 2048)
        h_t = h_t.to(next(sub.parameters()).dtype)
        sub_logits = sub.forward_sub_talker_finetune(h_t, gt_d.unsqueeze(0))
        loss = F.cross_entropy(
            sub_logits.reshape(-1, V), targets, ignore_index=PAD
        )
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == PROBE_STEPS - 1:
            with torch.inference_mode():
                h_t_e = adapter(adapted_hidden).to(next(sub.parameters()).dtype)
                sl = sub.forward_sub_talker_finetune(h_t_e, gt_d.unsqueeze(0))
                per_e = F.cross_entropy(
                    sl.reshape(-1, V), targets, reduction="none", ignore_index=PAD
                ).reshape(n_real, 15)
                ce_act_e = float(per_e[active].mean()) if active.any() else float("nan")
                ce_all_e = float(per_e.mean())
            ce_trace.append((step, float(loss.item()), ce_act_e, ce_all_e))
            print(f"  step {step}: loss={loss.item():.3f} CE act={ce_act_e:.3f} CE all={ce_all_e:.3f}", flush=True)
    # ripristina pesi ep2 per non inquinare altri check
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    print("pesi adapter ripristinati dal ckpt ep2", flush=True)
    ce_act_final = ce_trace[-1][2] if ce_trace else float("nan")
    probe_out = {
        "win": idx,
        "n_act": int(active.sum()),
        "n_real": n_real,
        "ce_act_base": ce_act_base,
        "ce_act_final": ce_act_final,
        "ce_all_base": ce_all_base,
        "ce_all_final": ce_trace[-1][3] if ce_trace else float("nan"),
        "steps": PROBE_STEPS,
        "trace": [(s, round(l, 4), round(a, 4), round(al, 4)) for (s, l, a, al) in ce_trace],
    }
    if not math.isnan(ce_act_base) and not math.isnan(ce_act_final):
        if ce_act_final < ce_act_base - 1.5:
            probe_verdict = ("PROBE OK: CE act %.2f -> %.2f (-%.2f). L'adapter PUO' "
                             "rappresentare la mappatura: il problema era "
                             "l'addestramento (H1/gradiente/distribuzione)."
                             % (ce_act_base, ce_act_final, ce_act_base - ce_act_final))
        else:
            probe_verdict = ("PROBE KO: CE act %.2f -> %.2f (delta %.2f < 1.5). "
                             "L'adapter NON riesce a ridurre la CE nemmeno "
                             "overfittando UNA finestra di parlato -> H2/H3 "
                             "(informazione/capacita'): serve distillazione h_t "
                             "o piu' capacita' (MLP/LoRA sub-talker)."
                             % (ce_act_base, ce_act_final, ce_act_final - ce_act_base))
    else:
        probe_verdict = "PROBE INCONCLUSIVO (nan)"
    print("PROBE VERDICT:", probe_verdict, flush=True)

    # =====================================================================
    # Verdetto combinato
    # =====================================================================
    ce_tr = r_train["ce_act_mean"]
    ce_va = r_val["ce_act_mean"]
    frac_tr = r_train["speech_frac_energy"]
    frac_va = r_val["speech_frac_energy"]
    if not math.isnan(ce_tr) and not math.isnan(ce_va):
        if ce_tr < 4.0 and ce_va >= 6.0:
            verdict = ("H1 COVARIATE SHIFT: CE act train=%.2f val=%.2f (frac parlato "
                       "train=%.1f%% val=%.1f%%). Fix: val split stratificato."
                       % (ce_tr, ce_va, 100*frac_tr, 100*frac_va))
        elif ce_tr >= 6.0 and ce_va >= 6.0:
            verdict = ("H2/H3 MAI IMPARATO: CE act train=%.2f val=%.2f. + %s"
                       % (ce_tr, ce_va, probe_verdict))
        elif ce_tr < 4.0 and ce_va < 4.0:
            verdict = ("MODEL OK IN TF: CE act train=%.2f val=%.2f. Problema a valle "
                       "(sampling/exposure). + %s" % (ce_tr, ce_va, probe_verdict))
        else:
            verdict = ("CASO MISTO: train=%.2f val=%.2f. + %s"
                       % (ce_tr, ce_va, probe_verdict))
    else:
        verdict = "DATI INSUFFICIENTI (nan). + " + probe_verdict

    print("\n" + "=" * 70)
    print("VERDETTO CHECK-B2:", verdict)
    print("=" * 70)
    print(json.dumps({"train": r_train, "val": r_val, "probe": probe_out}, indent=2))
    sys.stdout.flush()

    with open("data/checkb2_report.txt", "w", encoding="utf-8") as f:
        f.write("CHECK-B2: CE per energia train vs val + capacity probe\n")
        f.write("ckpt md5(adapter) = %s\n\n" % h)
        f.write("VERDETTO: %s\n\n" % verdict)
        f.write("TRAIN\n" + json.dumps(r_train, indent=2) + "\n\n")
        f.write("VAL\n" + json.dumps(r_val, indent=2) + "\n\n")
        f.write("PROBE\n" + json.dumps(probe_out, indent=2) + "\n\n")
        f.write("RIGHE TRAIN (win\tn_real\tn_act\tce_all\tce_act\tce_sil)\n")
        for r in rows_train:
            f.write(f"{r['win']}\t{r['n_real']}\t{r['n_act']}\t{r['ce_all']:.4f}\t{r['ce_act']:.4f}\t{r['ce_sil']:.4f}\n")
        f.write("\nRIGHE VAL\n")
        for r in rows_val:
            f.write(f"{r['win']}\t{r['n_real']}\t{r['n_act']}\t{r['ce_all']:.4f}\t{r['ce_act']:.4f}\t{r['ce_sil']:.4f}\n")
    print("=== CHECK-B2 REPORT SALVATO: data/checkb2_report.txt ===")
    print("=== CHECK-B2 COMPLETATO ===")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("CHECK-B2 FALLITA:\n" + tb)
    sys.stdout.flush()
    with open("checkb2_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
