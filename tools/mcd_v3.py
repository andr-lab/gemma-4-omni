# -*- coding: utf-8 -*-
"""MCD gold-standard (Kubichek) su frame voiced per wav syntest v3.
Mel log-spettrogramma (n_fft=512, n_mels=80, hop=128, log1p come repo),
DCT-II ortogonale per frame lungo le BANDE mel, coeff cepstrali 1..12 (c0 escluso),
MCD = (10/ln10)*sqrt(2*mean_i(c-c_ref)^2) medio sui frame voiced.
Calibrazione attesa (da re-audit): gt vs gt=0; gt vs parlato diverso ~16.6; gt vs rumore ~18.9.
"""
import os, glob, json, re
import numpy as np
from scipy.io import wavfile

SR = 24000
N_FFT = 512
N_MELS = 80
VOICE_THR = 0.15
MC_LO, MC_HI = 1, 13  # esclude c0 (energia): usa coeff 1..12

def load(p):
    sr, x = wavfile.read(p)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.float32:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)

def _wav_to_mel_l1(x, sr=SR, n_fft=N_FFT, n_mels=N_MELS):
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
        filt[m] = np.clip(
            np.minimum((mel_bins - lo) / (ctr - lo + 1e-9), (hi - mel_bins) / (hi - ctr + 1e-9)),
            0.0, 1.0,
        )
    win = np.hanning(n_fft).astype(np.float32)
    for t in range(n_frames):
        seg = xp[t * hop: t * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg)) ** 2
        mel[t] = filt @ spec
    return np.log1p(mel)  # (T, 80)

def _dct2_ortho(x):
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    if N == 0:
        return x.copy()
    n = np.arange(N)
    k = np.arange(N)[:, None]
    _cos = np.cos((np.pi / N) * (n[None, :] + 0.5) * k)
    _y = _cos @ x
    _y[0] /= np.sqrt(2.0)
    _y *= np.sqrt(2.0 / N)
    return _y

def cepstral(mel):
    """DCT-II lungo le bande per ogni frame -> (T, N_MELS) cepstral."""
    T, M = mel.shape
    out = np.zeros((T, M))
    for t in range(T):
        out[t] = _dct2_ortho(mel[t])
    return out

def mcd_voiced(x, ref, mask):
    mx = cepstral(_wav_to_mel_l1(x))[:, MC_LO:MC_HI]
    mr = cepstral(_wav_to_mel_l1(ref))[:, MC_LO:MC_HI]
    T = min(mx.shape[0], mr.shape[0])
    mx, mr = mx[:T], mr[:T]
    d = (10.0 / np.log(10.0)) * np.sqrt(2.0 * ((mx - mr) ** 2).mean(axis=1))
    m = mask[:T]
    return float(d[m].mean()) if m.any() else float(d.mean())

def frame_energy(x, n_fft=N_FFT, hop=N_FFT // 4):
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    e = np.zeros(n_frames)
    for t in range(n_frames):
        e[t] = float(np.sum(xp[t * hop: t * hop + n_fft] ** 2))
    return e

def main():
    folder = "eval_syntest_v3_out"
    refs = {k: load(os.path.join(folder, f"eval_r2_gt_ref_{k}.wav")) for k in "abc"}
    en_refs = {k: frame_energy(refs[k]) for k in "abc"}
    masks = {k: en_refs[k] > VOICE_THR * en_refs[k].max() for k in "abc"}
    print("frame voiced:", {k: int(m.sum()) for k, m in masks.items()})

    # calibrazione
    for k in "abc":
        print(f"[cal] gt_ref_{k} vs se stesso: MCD={mcd_voiced(refs[k], refs[k], masks[k]):.2f} dB (atteso 0)")
    print(f"[cal] gt_ref_a vs gt_ref_b (parlato diverso): {mcd_voiced(refs['a'], refs['b'], masks['a']):.2f} dB (atteso ~16.6)")
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(refs["a"])) * 0.05
    print(f"[cal] gt_ref_a vs rumore: {mcd_voiced(noise, refs['a'], masks['a']):.2f} dB (atteso ~18.9)")

    res = {}
    for w in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        base = os.path.basename(w)
        m_tag = re.search(r"_([abc])_?[A-Za-z0-9]*\.wav$", base)
        k = m_tag.group(1) if m_tag else None
        if k not in "abc":
            continue
        x = load(w)
        m = mcd_voiced(x, refs[k], masks[k])
        rms = float(np.sqrt(np.mean(x ** 2)))
        res[base] = {"mcd_voiced_dB": round(m, 2), "rms": round(rms, 5)}
        print(f"{base:34s} MCD_voiced={m:7.2f} dB | rms={rms:.5f}")

    with open(os.path.join(folder, "mcd_v3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\nsalvato mcd_v3.json")

if __name__ == "__main__":
    main()
