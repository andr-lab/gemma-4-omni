# -*- coding: utf-8 -*-
# ============================================================================
# CHECK-C: ADAPTER-ONLY TRAINING (generalizzazione) — post-CHECK-B3 (PROBE OK)
# B3 ha dimostrato: l'adapter attuale PUO' overfittare UNA finestra a CE act
# ~0.01 in 300 step (Adam lr 1e-3, h_t FISSO dal base congelato, CE pura).
# CHECK-C risponde: quella capacita' GENERALIZZA su piu' finestre train E su
# finestre VAL, quando il training e':
#   * adapter-only (LoRA Gemma CONGELATA -> h_t fisso, come nel probe)
#   * lr alto (1e-3, come nel probe)
#   * CE coupling pura (senza competizione con text_loss)
# Se SI' -> la causa di H1 era l'ottimizzazione congiunta (lr basso + LoRA
# mobile + competizione text loss): fix = separare l'ottimizzatore dell'adapter.
# Se NO (train scende, val no) -> overfit sulle finestre campionate.
# Se NIENTE scende -> contraddice B3 (da investigare).
# ============================================================================
import os, sys, json, traceback, hashlib, math

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    import unsloth  # noqa: F401
    import kaggle_pipeline as KP

    STEPS = int(os.environ.get("CHECKC_STEPS", "400"))
    LR = float(os.environ.get("CHECKC_LR", "1e-3"))
    N_TRAIN = int(os.environ.get("CHECKC_N_TRAIN", "12"))   # finestre train con parlato
    N_VAL = int(os.environ.get("CHECKC_N_VAL", "12"))       # finestre val con parlato
    MIN_ACT_TRAIN = int(os.environ.get("CHECKC_MIN_ACT_TRAIN", "4"))
    MIN_ACT_VAL = int(os.environ.get("CHECKC_MIN_ACT_VAL", "2"))
    EVAL_EVERY = int(os.environ.get("CHECKC_EVAL_EVERY", "25"))

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
    print(f"split: train=[0,{val_start}) | val=[{val_start},{t_agent})")
    sys.stdout.flush()

    def _slice_text(text_list, f_start):
        if text_list is None:
            return None
        return text_list[f_start // 64:]

    MPF = KP.MEL_PER_AGENT_FRAME
    ds_train = KP.FullDuplexDataset(
        user_mel[:val_start * MPF], agent_mel[:val_start * MPF],
        agent_codes[:, :val_start], seq_len=64,
        text_by_window=_slice_text(text_by_window, 0),
    )
    ds_val = KP.FullDuplexDataset(
        user_mel[val_start * MPF:], agent_mel[val_start * MPF:],
        agent_codes[:, val_start:], seq_len=64,
        text_by_window=_slice_text(text_by_window, val_start),
    )
    print(f"ds_train: {len(ds_train)} finestre | ds_val: {len(ds_val)} finestre")
    sys.stdout.flush()

    # ---- modello + ckpt ep2 ----
    duplex_model, _ = KP.load_gemma4_duplex_system()
    ckpt = "checkpoints/gemma4_duplex_sota_ckpt"
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    h = hashlib.md5(open(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors"), "rb").read()).hexdigest()[:8]
    print(f"ckpt md5(adapter)={h} (atteso 86a854e9)")
    device = next(duplex_model.base_model.parameters()).device

    # ---- device-patch (come fix_cell2 / checkb3) ----
    import inspect as _inspect
    import types as _types
    _src_gen = ""
    try:
        _src_gen = _inspect.getsource(type(duplex_model).generate)
    except Exception:
        _src_gen = ""
    _needs_patch = _src_gen and "semantic_head.weight.device != adapted_hidden.device" not in _src_gen
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
    decoder = KP.Code2WavDecoder()

    def _rms(x):
        x = np.asarray(x, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    def _frame_active_mask(wav, n_real):
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
    # 1) estrai finestre train/val (adapted_hidden FROZEN + gt + maschera energia)
    #    -> h_t fisso, esattamente come nel probe B3 che converge
    # =====================================================================
    def _extract_window(ds, idx):
        item = ds[idx]
        n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
        if n_real <= 0:
            return None
        with torch.inference_mode():
            inp = {
                "input_ids": item["input_ids"].unsqueeze(0).to(device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
                "input_features": item["input_features"].unsqueeze(0).to(device),
                "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
            }
            outputs = duplex_model.base_model(
                **inp, output_hidden_states=True, return_dict=True, logits_to_keep=1
            )
            hidden_states = outputs.hidden_states[-1]
            agent_hidden = hidden_states[
                :, audio_feat_start + fpf - 1 : audio_feat_start + fpf * n_real : fpf, :
            ]
            adapted_hidden = duplex_model._apply_causal_conv(agent_hidden)
            gt = item["audio_targets"][:n_real]
            wav_gt = decoder.decode(torch.from_numpy(KP._trim_trailing_silence(gt.numpy())))
            active, _ = _frame_active_mask(wav_gt, n_real)
            if int(active.sum()) == 0:
                return None
            return {
                "ah": adapted_hidden.detach().cpu().float(),
                "gt": gt.detach().cpu(),
                "active": active.copy(),
            }

    print("=== SCAN finestre TRAIN ===", flush=True)
    train_wins = []
    for i in range(min(256, len(ds_train))):
        w = _extract_window(ds_train, i)
        if w is None:
            continue
        n_act = int(w["active"].sum())
        if n_act >= MIN_ACT_TRAIN:
            train_wins.append((i, n_act, w))
            if len(train_wins) >= N_TRAIN:
                break
    print(f"  raccolte {len(train_wins)} finestre train (n_act>={MIN_ACT_TRAIN})")
    for i, n_act, _ in train_wins:
        print(f"  win{i}: n_act={n_act}")
    sys.stdout.flush()

    print("=== SCAN finestre VAL ===", flush=True)
    val_wins = []
    for i in range(min(256, len(ds_val))):
        w = _extract_window(ds_val, i)
        if w is None:
            continue
        n_act = int(w["active"].sum())
        if n_act >= MIN_ACT_VAL:
            val_wins.append((i, n_act, w))
            if len(val_wins) >= N_VAL:
                break
    print(f"  raccolte {len(val_wins)} finestre val (n_act>={MIN_ACT_VAL})")
    for i, n_act, _ in val_wins:
        print(f"  win{i}: n_act={n_act}")
    sys.stdout.flush()

    if not train_wins or not val_wins:
        print("CHECK-C IMPOSSIBILE: servono finestre con parlato in train E val")
        raise SystemExit(0)

    # =====================================================================
    # 2) adapter FRESCO (stessa architettura, pesi ep2 come init)
    #    -> evita i tensori in inference mode ereditati dal load del ckpt
    # =====================================================================
    src_adapter = duplex_model.ht_adapter
    lin, ln = src_adapter[0], src_adapter[1]
    adapter = nn.Sequential(
        nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None),
        nn.LayerNorm(ln.normalized_shape, elementwise_affine=ln.elementwise_affine),
    ).to(device)
    with torch.no_grad():
        adapter[0].weight.copy_(lin.weight.detach())
        if lin.bias is not None:
            adapter[0].bias.copy_(lin.bias.detach())
        adapter[1].weight.copy_(ln.weight.detach())
        adapter[1].bias.copy_(ln.bias.detach())
    adapter.train()
    for p in adapter.parameters():
        p.requires_grad = True
    print("adapter fresco costruito: %d params trainable" %
          sum(p.numel() for p in adapter.parameters() if p.requires_grad), flush=True)

    sub_dtype = next(sub.parameters()).dtype

    def _ce_metrics(w):
        ah = w["ah"].to(device).float()
        gt = w["gt"].to(device)
        active = torch.from_numpy(w["active"]).to(device)
        with torch.inference_mode():
            h_t = adapter(ah).to(sub_dtype)
            sl = sub.forward_sub_talker_finetune(h_t, gt.unsqueeze(0))
            per = F.cross_entropy(
                sl.reshape(-1, V), gt[:, 1:].reshape(-1),
                reduction="none", ignore_index=PAD,
            ).reshape(gt.shape[0], 15)
            n_act = int(active.sum())
            ce_act = float(per[active].mean()) if n_act else float("nan")
            ce_all = float(per.mean())
        return ce_act, ce_all

    def _eval_split(wins):
        """CE act media pesata per frame attivo + CE all media su tutte le finestre."""
        tot_act = 0
        sum_act = 0.0
        tot_all = 0
        sum_all = 0.0
        n_fin = 0
        per_win = []
        for i, n_act, w in wins:
            ce_act, ce_all = _ce_metrics(w)
            per_win.append((i, ce_act, ce_all))
            if not math.isnan(ce_act):
                sum_act += ce_act * n_act
                tot_act += n_act
            sum_all += ce_all
            tot_all += 1
            n_fin += 1
        return {
            "ce_act": (sum_act / tot_act) if tot_act else float("nan"),
            "ce_all": (sum_all / tot_all) if tot_all else float("nan"),
            "per_win": per_win,
        }

    # baseline con adapter ep2 (init)
    base_train = _eval_split(train_wins)
    base_val = _eval_split(val_wins)
    print(f"\nBASELINE (adapter ep2): TRAIN ce_act={base_train['ce_act']:.3f} ce_all={base_train['ce_all']:.3f}"
          f" | VAL ce_act={base_val['ce_act']:.3f} ce_all={base_val['ce_all']:.3f}", flush=True)

    # =====================================================================
    # 3) training adapter-only: Adam lr, CE coupling pura, h_t fisso
    #    (identico al probe B3 ma su N finestre train + eval su val)
    # =====================================================================
    print(f"\n=== TRAINING ADAPTER-ONLY: {STEPS} step, lr={LR}, {len(train_wins)} win train ===", flush=True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.0)
    trace = []
    for step in range(STEPS):
        i, n_act, w = train_wins[step % len(train_wins)]
        ah = w["ah"].to(device).float()
        gt = w["gt"].to(device)
        opt.zero_grad()
        h_t = adapter(ah).to(sub_dtype)
        sub_logits = sub.forward_sub_talker_finetune(h_t, gt.unsqueeze(0))
        loss = F.cross_entropy(sub_logits.reshape(-1, V), gt[:, 1:].reshape(-1), ignore_index=PAD)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            tr = _eval_split(train_wins)
            va = _eval_split(val_wins)
            trace.append((step, float(loss.item()), tr["ce_act"], tr["ce_all"], va["ce_act"], va["ce_all"]))
            print(f"  step {step}: loss={loss.item():.3f} | TRAIN ce_act={tr['ce_act']:.3f} ce_all={tr['ce_all']:.3f}"
                  f" | VAL ce_act={va['ce_act']:.3f} ce_all={va['ce_all']:.3f}", flush=True)

    final_train = _eval_split(train_wins)
    final_val = _eval_split(val_wins)
    d_val = base_val["ce_act"] - final_val["ce_act"]

    if not math.isnan(final_val["ce_act"]) and d_val >= 1.5:
        verdict = (f"CHECK-C OK: VAL ce_act {base_val['ce_act']:.3f} -> {final_val['ce_act']:.3f} "
                   f"(delta {d_val:.3f}). L'adapter-only con lr alto GENERALIZZA a finestre val: "
                   f"la causa di H1 era l'ottimizzazione congiunta (lr basso + LoRA mobile + "
                   f"competizione text loss). Fix: separare l'ottimizzatore dell'adapter.")
    elif not math.isnan(final_val["ce_act"]) and final_val["ce_act"] <= 3.0:
        verdict = (f"CHECK-C PARZIALE: VAL ce_act {base_val['ce_act']:.3f} -> {final_val['ce_act']:.3f} "
                   f"(delta {d_val:.3f}). Il valore assoluto è basso ma il delta è contenuto: "
                   f"segno di apprendimento, da consolidare con più dati/finestre.")
    elif not math.isnan(final_train["ce_act"]) and final_train["ce_act"] < 1.0 and not math.isnan(final_val["ce_act"]) and final_val["ce_act"] > 4.0:
        verdict = (f"CHECK-C OVERFIT: TRAIN ce_act={final_train['ce_act']:.3f} ma VAL ce_act={final_val['ce_act']:.3f}. "
                   f"L'adapter memorizza le finestre train ma non generalizza: poche finestre o "
                   f"problema più profondo (da investigare con più finestre).")
    else:
        verdict = (f"CHECK-C INCONCLUSIVO: TRAIN ce_act={final_train['ce_act']:.3f} VAL ce_act={final_val['ce_act']:.3f} "
                   f"(base val={base_val['ce_act']:.3f}). Nessun apprendimento chiaro.")

    print("\n" + "=" * 70)
    print("CHECK-C VERDICT:", verdict)
    print("=" * 70)
    print(json.dumps({
        "steps": STEPS, "lr": LR, "n_train_wins": len(train_wins), "n_val_wins": len(val_wins),
        "base_train": {"ce_act": base_train["ce_act"], "ce_all": base_train["ce_all"]},
        "base_val": {"ce_act": base_val["ce_act"], "ce_all": base_val["ce_all"]},
        "final_train": {"ce_act": final_train["ce_act"], "ce_all": final_train["ce_all"]},
        "final_val": {"ce_act": final_val["ce_act"], "ce_all": final_val["ce_all"]},
        "trace": [(s, round(l, 4), round(a, 4), round(b, 4), round(c, 4), round(d, 4))
                  for (s, l, a, b, c, d) in trace],
    }, indent=2))
    sys.stdout.flush()

    with open("data/checkc_report.txt", "w", encoding="utf-8") as f:
        f.write("CHECK-C ADAPTER-ONLY TRAINING REPORT\n")
        f.write(f"ckpt md5(adapter) = {h}\n")
        f.write(f"steps={STEPS} lr={LR} n_train={len(train_wins)} n_val={len(val_wins)}\n")
        f.write(f"VERDICT: {verdict}\n\n")
        f.write(f"base  TRAIN ce_act={base_train['ce_act']:.4f} ce_all={base_train['ce_all']:.4f}\n")
        f.write(f"base  VAL   ce_act={base_val['ce_act']:.4f} ce_all={base_val['ce_all']:.4f}\n")
        f.write(f"final TRAIN ce_act={final_train['ce_act']:.4f} ce_all={final_train['ce_all']:.4f}\n")
        f.write(f"final VAL   ce_act={final_val['ce_act']:.4f} ce_all={final_val['ce_all']:.4f}\n\n")
        f.write("trace (step, loss, tr_act, tr_all, val_act, val_all):\n")
        for t in trace:
            f.write("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % t)
        f.write("\nper-window final VAL:\n")
        for i, ce_act, ce_all in final_val["per_win"]:
            f.write(f"  val_win{i}: ce_act={ce_act:.3f} ce_all={ce_all:.3f}\n")
        f.write("\nper-window final TRAIN:\n")
        for i, ce_act, ce_all in final_train["per_win"]:
            f.write(f"  train_win{i}: ce_act={ce_act:.3f} ce_all={ce_all:.3f}\n")
    print("=== CHECK-C REPORT SALVATO: data/checkc_report.txt ===")
    print("=== CHECK-C COMPLETATO ===")
    sys.stdout.flush()

except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    print("CHECK-C FALLITA (eccezione rilanciata)")
    sys.stdout.flush()
    raise
