# -*- coding: utf-8 -*-
"""
Verifica oggettiva delle wav del syntest v2 (sampling ufficiale Qwen3-TTS).
Metriche per ogni wav: rms, energia >6kHz, flatness spettrale, autocorr frame,
CV energia. Genera spettrogrammi di confronto gt_ref vs oracle vs pred.
Uso: python _syntest_metrics.py <cartella_wav> [outdir]
"""
import os
import sys
import glob
import json
import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile

def metrics(wav_path, sr=24000, nfft=1024):
    sr_f, x = wavfile.read(wav_path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.float32:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x ** 2)))
    # spettrogramma
    f, t, Sxx = sp_signal.spectrogram(x, fs=sr, nperseg=nfft, noverlap=nfft // 2)
    S = Sxx ** 2  # potenza
    S_sum = S.sum(axis=0)
    total = float(S.sum())
    e6 = float(S[f > 6000].sum()) / max(total, 1e-12) if f.max() > 6000 else 0.0
    # flatness (media per frame)
    with np.errstate(divide='ignore', invalid='ignore'):
        gmean = np.exp(np.mean(np.log(S + 1e-12), axis=0))
    amean = S.mean(axis=0)
    flatness = float(np.mean(gmean / np.maximum(amean, 1e-12)))
    # autocorrelazione media sui frame (periodicita')
    frame = 2048
    hops = []
    for st in range(0, len(x) - frame, frame):
        seg = x[st:st + frame]
        seg = seg - seg.mean()
        if np.std(seg) < 1e-9:
            continue
        ac = np.correlate(seg, seg, 'full')[frame - 1:]
        ac = ac / ac[0]
        hops.append(float(ac[1]))
    autocorr = float(np.mean(hops)) if hops else 0.0
    # CV energia per frame
    fr_e = []
    for st in range(0, len(x) - frame, frame):
        fr_e.append(float(np.sum(x[st:st + frame] ** 2)))
    fr_e = np.array(fr_e)
    cv = float(np.std(fr_e) / max(np.mean(fr_e), 1e-12)) if fr_e.size else 0.0
    return {"rms": rms, "e_gt6k": e6, "flatness": flatness,
            "autocorr": autocorr, "cv_energy": cv, "sr": int(sr_f), "len_s": float(len(x) / sr_f)}

def spec_pil(wav_path, out_png, sr=24000):
    from PIL import Image, ImageDraw
    sr_f, x = wavfile.read(wav_path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.float32:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f, t, Sxx = sp_signal.spectrogram(x, fs=sr_f, nperseg=512, noverlap=256)
    db = 10 * np.log10(Sxx + 1e-12)
    db = np.clip(db, -100, 0)
    # normalizza 0..255
    img = ((db + 100) / 100.0 * 255.0).astype(np.uint8)
    img = np.flipud(img)  # freq crescente verso l'alto
    im = Image.fromarray(img, mode='L').resize((1000, 400), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), os.path.basename(wav_path), fill=255)
    im.save(out_png)
    return out_png

def main():
    folder = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(folder, "spec_met")
    os.makedirs(outdir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if not wavs:
        print("nessuna wav in", folder)
        return 1
    res = {}
    for w in wavs:
        m = metrics(w)
        res[os.path.basename(w)] = m
        base = os.path.splitext(os.path.basename(w))[0]
        spec_pil(w, os.path.join(outdir, base + ".png"))
        print(f"{os.path.basename(w):32s} rms={m['rms']:.4f} E>6k={m['e_gt6k']*100:.2f}% "
              f"flat={m['flatness']:.3f} autocorr={m['autocorr']:.4f} cv={m['cv_energy']:.2f} "
              f"len={m['len_s']:.2f}s")
    with open(os.path.join(outdir, "syntest_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nmetriche salvate in", os.path.join(outdir, "syntest_metrics.json"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
