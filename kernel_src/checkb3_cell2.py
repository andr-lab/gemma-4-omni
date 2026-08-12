# -*- coding: utf-8 -*-
# ============================================================================
# CHECK-B3: CAPACITY PROBE (fix del bug requires_grad-in-InferenceMode di B2)
# Scopo: l'adapter (Linear 3072->2048 + LayerNorm) PUO' rappresentare la
# mappatura h_t -> codici parlato del sub-talker congelato?
#   - prendo UNA finestra train densa di parlato
#   - estraggo adapted_hidden dal base congelato UNA volta (frozen)
#   - ricostruisco l'adapter FRESCO (stessa architettura, pesi ep2 come init)
#     -> evita i tensori in inference mode ereditati dal load del ckpt
#   - ottimizzo SOLO questo adapter (Adam lr 1e-3, PROBE_STEPS step) sulla
#     coupling loss della finestra (teacher forcing, sub-talker congelato)
#   - misuro CE parlato (maschera energia) prima/dopo
# Verdetto:
#   CE act scende nettamente (-> ~1-2) : l'adapter PUO' rappresentare la
#       mappatura; il problema e' l'addestramento (H1/H-segnale)
#   CE act resta ~5-7              : l'adapter NON riesce nemmeno a
#       overfittare UNA finestra -> H2/H3 (informazione/capacita') ->
#       serve distillazione h_t o piu' capacita' (MLP/LoRA)
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

    PROBE_STEPS = int(os.environ.get("CHECKB3_PROBE_STEPS", "300"))
    PROBE_LR = float(os.environ.get("CHECKB3_PROBE_LR", "1e-3"))
    N_SCAN = int(os.environ.get("CHECKB3_N_SCAN", "16"))  # finestre train da scansionare

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
    print(f"ds_train: {len(ds_train)} finestre")
    sys.stdout.flush()

    # ---- modello + ckpt ep2 ----
    duplex_model, _ = KP.load_gemma4_duplex_system()
    ckpt = "checkpoints/gemma4_duplex_sota_ckpt"
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    h = hashlib.md5(open(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors"), "rb").read()).hexdigest()[:8]
    print(f"ckpt md5(adapter)={h} (atteso 86a854e9)")
    device = next(duplex_model.base_model.parameters()).device

    # ---- device-patch (come fix_cell2) ----
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
    # 1) scansiona le prime N_SCAN finestre train, trova quella densa di
    #    parlato, estrai adapted_hidden (frozen) + gt + maschera energia
    # =====================================================================
    print("=== SCAN finestre train per probe ===", flush=True)
    best = None
    for i in range(min(N_SCAN, len(ds_train))):
        item = ds_train[i]
        n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
        if n_real <= 0:
            continue
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
            gt = item["audio_targets"][:n_real].to(device)
            # CE baseline con adapter ep2 (inference)
            if next(duplex_model.ht_adapter.parameters()).device != adapted_hidden.device:
                duplex_model.ht_adapter.to(adapted_hidden.device)
            if next(sub.parameters()).device != adapted_hidden.device:
                sub.to(adapted_hidden.device)
            ad_dtype = next(duplex_model.ht_adapter.parameters()).dtype
            h_t0 = duplex_model.ht_adapter(adapted_hidden.to(ad_dtype)).to(next(sub.parameters()).dtype)
            sl0 = sub.forward_sub_talker_finetune(h_t0, gt.unsqueeze(0))
            per0 = F.cross_entropy(
                sl0.reshape(-1, V), gt[:, 1:].reshape(-1),
                reduction="none", ignore_index=PAD,
            ).reshape(n_real, 15)
            wav_gt = decoder.decode(torch.from_numpy(KP._trim_trailing_silence(gt.cpu().numpy())))
        active, rms_per = _frame_active_mask(wav_gt, n_real)
        n_act = int(active.sum())
        ce_act0 = float(per0[active].mean()) if n_act else float("nan")
        ce_all0 = float(per0.mean())
        print(f"  win{i}: n_act={n_act}/{n_real} CE act={ce_act0:.3f} CE all={ce_all0:.3f}", flush=True)
        if n_act >= 8 and (best is None or n_act > best[1]):
            # salva (senza mantenere il graph): tensori distaccati
            best = (i, n_act, adapted_hidden.detach().to("cpu"), gt.detach().cpu(),
                    active.copy(), per0.detach().cpu(), ce_act0, ce_all0)
    if best is None:
        print("NESSUNA finestra con >=8 frame parlati trovata: probe impossibile")
        raise SystemExit(0)
    idx, n_act, adapted_hidden_cpu, gt_cpu, active, per0_cpu, ce_act0, ce_all0 = best
    print(f"\nPROBE su win{idx}: n_act={n_act}/{gt_cpu.shape[0]} | "
          f"CE act BASE={ce_act0:.3f} CE all BASE={ce_all0:.3f}", flush=True)

    # =====================================================================
    # 2) costruisci adapter FRESCHI (tensori normali, addestrabili):
    #    A) LINEAR  = architettura attuale (Linear 3072->2048 + LayerNorm)
    #    B) MLP     = stessa dim ma con hidden ReLU (piu' capacita' non lineare)
    #    Discriminazione: se MLP scende e Linear no -> capacita' (B/C);
    #    se NESSUNO scende -> informazione assente in h_t (A).
    # =====================================================================
    src_adapter = duplex_model.ht_adapter
    lin, ln = src_adapter[0], src_adapter[1]
    H = lin.out_features

    def _build(name, mlp):
        if mlp:
            a = nn.Sequential(
                nn.Linear(lin.in_features, H, bias=True),
                nn.ReLU(),
                nn.Linear(H, H, bias=True),
                nn.LayerNorm(H, elementwise_affine=True),
            )
            with torch.no_grad():
                # init: prima layer = copia del Linear ep2, seconda = identita' scaled
                a[0].weight.copy_(lin.weight.detach())
                if lin.bias is not None:
                    a[0].bias.copy_(lin.bias.detach())
                nn.init.zeros_(a[2].weight)
                nn.init.zeros_(a[2].bias)
                a[3].weight.copy_(ln.weight.detach())
                a[3].bias.copy_(ln.bias.detach())
        else:
            a = nn.Sequential(
                nn.Linear(lin.in_features, H, bias=lin.bias is not None),
                nn.LayerNorm(H, elementwise_affine=ln.elementwise_affine),
            )
            with torch.no_grad():
                a[0].weight.copy_(lin.weight.detach())
                if lin.bias is not None:
                    a[0].bias.copy_(lin.bias.detach())
                a[1].weight.copy_(ln.weight.detach())
                a[1].bias.copy_(ln.bias.detach())
        a = a.to(device)
        a.train()
        for p in a.parameters():
            p.requires_grad = True
        npar = sum(p.numel() for p in a.parameters() if p.requires_grad)
        print(f"adapter {name} costruito: {npar} params trainable", flush=True)
        return a

    adapted_hidden = adapted_hidden_cpu.to(device)
    gt = gt_cpu.to(device)
    active_t = torch.from_numpy(active).to(device)
    targets = gt[:, 1:].reshape(-1).to(device)
    sub_dtype = next(sub.parameters()).dtype
    # NB: l'adapter del probe e' in fp32 (nuovo nn.Linear default) -> l'INPUT del
    # probe deve restare fp32; la conversione a sub_dtype avviene sull'OUTPUT
    # (h_t) prima della forward del sub-talker, esattamente come faceva
    # l'adapter originale (riga ~153: ht_adapter(...).to(next(sub.parameters()).dtype)).
    adapted_hidden = adapted_hidden.float()

    def _run_probe(name, adapter):
        opt = torch.optim.Adam(adapter.parameters(), lr=PROBE_LR)
        trace = []
        for step in range(PROBE_STEPS):
            opt.zero_grad()
            h_t = adapter(adapted_hidden)
            h_t = h_t.to(sub_dtype)
            sub_logits = sub.forward_sub_talker_finetune(h_t, gt.unsqueeze(0))
            loss = F.cross_entropy(sub_logits.reshape(-1, V), targets, ignore_index=PAD)
            loss.backward()
            opt.step()
            if step % 25 == 0 or step == PROBE_STEPS - 1:
                with torch.inference_mode():
                    h_t_e = adapter(adapted_hidden).to(sub_dtype)
                    sl = sub.forward_sub_talker_finetune(h_t_e, gt.unsqueeze(0))
                    per_e = F.cross_entropy(
                        sl.reshape(-1, V), targets, reduction="none", ignore_index=PAD
                    ).reshape(gt.shape[0], 15)
                    ce_act_e = float(per_e[active_t].mean()) if n_act else float("nan")
                    ce_all_e = float(per_e.mean())
                trace.append((step, float(loss.item()), ce_act_e, ce_all_e))
                print(f"  [{name}] step {step}: loss={loss.item():.3f} "
                      f"CE act={ce_act_e:.3f} CE all={ce_all_e:.3f}", flush=True)
        return trace

    results = {}
    for name, mlp in (("LINEAR", False), ("MLP", True)):
        print(f"\n=== PROBE {name} ==", flush=True)
        tr = _run_probe(name, _build(name, mlp))
        results[name] = {
            "ce_act_final": tr[-1][2] if tr else float("nan"),
            "ce_all_final": tr[-1][3] if tr else float("nan"),
            "trace": [(s, round(l, 4), round(a, 4), round(al, 4)) for (s, l, a, al) in tr],
        }

    # ---- verdetto combinato ----
    ce_lin = results["LINEAR"]["ce_act_final"]
    ce_mlp = results["MLP"]["ce_act_final"]
    delta_lin = ce_act0 - ce_lin
    delta_mlp = ce_act0 - ce_mlp
    if not math.isnan(ce_lin) and not math.isnan(ce_mlp):
        if delta_lin >= 1.5:
            verdict = ("PROBE OK (LINEAR): CE act %.2f -> %.2f (delta %.2f). L'adapter "
                       "attuale PUO' rappresentare la mappatura: il problema era "
                       "l'addestramento (H1/gradiente/distribuzione), non la capacita'."
                       % (ce_act0, ce_lin, delta_lin))
        elif delta_mlp >= 1.5:
            verdict = ("PROBE PARZIALE: Linear %.2f -> %.2f (delta %.2f) MA MLP %.2f -> %.2f "
                       "(delta %.2f). La capacita' non lineare SERVE: architettura adapter "
                       "insufficiente (H3). Fix: MLP piu' profondo o LoRA sul sub-talker."
                       % (ce_act0, ce_lin, delta_lin, ce_act0, ce_mlp, delta_mlp))
        else:
            verdict = ("PROBE KO (entrambi): Linear %.2f->%.2f, MLP %.2f->%.2f. Nemmeno un "
                       "MLP overfitta UNA finestra -> H2 (informazione assente in h_t): "
                       "serve distillazione/regressione h_t verso il teacher Qwen (FitNets)"
                       " o riprogettazione del segnale di condizionamento."
                       % (ce_act0, ce_lin, ce_act0, ce_mlp))
    else:
        verdict = f"PROBE INCONCLUSIVO (lin={ce_lin} mlp={ce_mlp})"

    print("\n" + "=" * 70)
    print("PROBE VERDICT:", verdict)
    print("=" * 70)
    print(json.dumps({
        "win": idx, "n_act": n_act, "n_real": int(gt.shape[0]),
        "ce_act_base": ce_act0, "ce_all_base": ce_all0,
        "linear": results["LINEAR"], "mlp": results["MLP"],
        "steps": PROBE_STEPS, "lr": PROBE_LR,
    }, indent=2))
    sys.stdout.flush()

    with open("data/checkb3_report.txt", "w", encoding="utf-8") as f:
        f.write("CHECK-B3 CAPACITY PROBE (LINEAR vs MLP)\n")
        f.write("ckpt md5(adapter) = %s\n" % h)
        f.write("win=%d n_act=%d n_real=%d\n" % (idx, n_act, gt.shape[0]))
        f.write("VERDICT: %s\n\n" % verdict)
        f.write("ce_act_base=%.4f | linear_final=%.4f | mlp_final=%.4f\n\n"
                % (ce_act0, results["LINEAR"]["ce_act_final"], results["MLP"]["ce_act_final"]))
        f.write("trace LINEAR (step, loss, ce_act, ce_all):\n")
        for t in results["LINEAR"]["trace"]:
            f.write("%d\t%.4f\t%.4f\t%.4f\n" % t)
        f.write("\ntrace MLP (step, loss, ce_act, ce_all):\n")
        for t in results["MLP"]["trace"]:
            f.write("%d\t%.4f\t%.4f\t%.4f\n" % t)
    print("=== CHECK-B3 REPORT SALVATO: data/checkb3_report.txt ===")
    print("=== CHECK-B3 COMPLETATO ===")
    sys.stdout.flush()
except SystemExit:
    print("EXIT senza probe (nessuna finestra valida)")
    sys.stdout.flush()
except Exception:
    tb = traceback.format_exc()
    print("CHECK-B3 FALLITA:\n" + tb)
    sys.stdout.flush()
    with open("checkb3_ERROR.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    raise
