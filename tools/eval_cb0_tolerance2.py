#!/usr/bin/env python3
"""eval_cb0_tolerance2.py — Tolleranza cb0 su finestre val CON parlato.
1) decode p=0 su 10 finestre val -> ASR -> seleziona quelle con trascrizione non vuota
2) sulle selezionate: corruzione p=0.1/0.3/0.5 -> mel cos-sim vs p=0 (oggettivo) + ASR
Fornisce la curva di tolleranza reale del sub-talker agli errori su cb0."""
import os, sys
sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")
import numpy as np
import torch
import soundfile as sf
import librosa

import kaggle_pipeline as KP


def mel_cos(a, b, sr=24000, n_mels=80):
    ma = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=n_mels, hop_length=320, n_fft=1024)
    mb = librosa.feature.melspectrogram(y=b, sr=sr, n_mels=n_mels, hop_length=320, n_fft=1024)
    la, lb = librosa.power_to_db(ma), librosa.power_to_db(mb)
    la, lb = la.ravel().astype(np.float64), lb.ravel().astype(np.float64)
    if la.std() < 1e-6 or lb.std() < 1e-6:
        return 0.0
    return float(np.dot(la, lb) / (np.linalg.norm(la) * np.linalg.norm(lb)))


def main():
    cache = "/workspace/cache_ars_full.pt"
    adapter_ckpt = "/workspace/checkpoints/ht_adapter_temporal2_final.pt"
    out = "/workspace/samples_tol2"

    print("[TOL2] 1/4 Sistema + adapter...", flush=True)
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, os.path.dirname(adapter_ckpt))
    for p in duplex_model.parameters():
        p.requires_grad = False
    duplex_model.eval()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)

    blob = torch.load(cache, weights_only=False)
    va_h, va_c = blob["va_h"], blob["va_c"]
    decoder = KP.Code2WavDecoder()
    os.makedirs(out, exist_ok=True)
    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cuda", compute_type="float16")

    sub_dev = next(duplex_model.subtalker.parameters()).device
    sub_dtype = next(duplex_model.subtalker.parameters()).dtype
    V = 2048

    def decode(cb0, i):
        T = cb0.shape[0]
        h_sub = va_h[i][:T].unsqueeze(0).to(sub_dev).to(sub_dtype)
        codes_anchor = torch.zeros((1, T, 16), dtype=torch.long, device=sub_dev)
        codes_anchor[:, :, 0] = cb0.unsqueeze(0).to(sub_dev)
        with torch.inference_mode():
            logits = duplex_model.subtalker.forward_sub_talker_finetune(h_sub, codes_anchor)
            pred_rest = torch.argmax(logits, dim=-1)
            full = torch.zeros((1, T, 16), dtype=torch.long)
            full[0, :, 0] = cb0
            full[0, :, 1:] = pred_rest.cpu()
            wav = decoder.decode(full[0].cpu())
        return wav

    def asr_text(wav, f):
        sf.write(f, wav, 24000)
        segs, _ = asr.transcribe(f, language="it", beam_size=5)
        return " ".join(s.text.strip() for s in segs)

    # 1) trova finestre con parlato (p=0 trascrivibile)
    print("[TOL2] 2/4 selezione finestre con parlato (p=0)...", flush=True)
    good = []
    for i in range(min(12, len(va_h))):
        T = min(64, va_c[i].shape[0], va_h[i].shape[0])
        cb0 = va_c[i][:T].clone()
        wav = decode(cb0, i)
        t = asr_text(wav, f"{out}/sel_w{i}.wav")
        if t:
            good.append(i)
            print(f"[TOL2] win{i} p=0 asr: {t[:110]}", flush=True)
        if len(good) >= 4:
            break
    if not good:
        print("[TOL2] nessuna finestra con parlato nei primi 12 val — allargo la ricerca", flush=True)
        for i in range(12, min(40, len(va_h))):
            T = min(64, va_c[i].shape[0], va_h[i].shape[0])
            cb0 = va_c[i][:T].clone()
            wav = decode(cb0, i)
            t = asr_text(wav, f"{out}/sel_w{i}.wav")
            if t:
                good.append(i)
                print(f"[TOL2] win{i} p=0 asr: {t[:110]}", flush=True)
            if len(good) >= 4:
                break

    # 2) sweep corruzione
    print(f"[TOL2] 3/4 sweep corruzione su {len(good)} finestre con parlato", flush=True)
    for i in good:
        T = min(64, va_c[i].shape[0], va_h[i].shape[0])
        gt = va_c[i][:T].clone()
        w0 = decode(gt, i)
        for p in [0.1, 0.3, 0.5]:
            rng = np.random.default_rng(42 + i)
            cb0 = gt.clone()
            mask = rng.random(T) < p
            for j in np.where(mask)[0]:
                cand = int(rng.integers(0, V))
                while cand == int(cb0[j]):
                    cand = int(rng.integers(0, V))
                cb0[j] = cand
            wav = decode(cb0, i)
            cs = mel_cos(wav, w0)
            t = asr_text(wav, f"{out}/tol_p{int(p*100):03d}_w{i}.wav")
            print(f"[TOL2] win{i} p={p:.1f} mel_cos_vs_p0={cs:.3f} asr: {t[:110]}", flush=True)
    print("[TOL2] FATTO.", flush=True)


if __name__ == "__main__":
    main()
