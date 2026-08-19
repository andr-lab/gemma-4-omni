# -*- coding: utf-8 -*-
"""Metriche di corruzione oggettive: LSD vs gt_ref, phase coherence, entropia spettrale, armonicita'."""
import os, glob, json
import numpy as np
from scipy.io import wavfile
from scipy import signal as sp

SR = 24000

def load(p):
    sr, x = wavfile.read(p)
    if x.dtype == np.int16: x = x.astype(np.float64)/32768.0
    elif x.dtype == np.float32: x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(axis=1)
    return x

def frame_energy(x, n=2048, h=1024):
    out=[]
    for st in range(0, len(x)-n, h):
        out.append(float(np.sum(x[st:st+n]**2)))
    return np.array(out)

def lsd(x, ref, nfft=1024, eps=1e-8):
    """Log-spectral distance media (dB) tra x e ref, frame allineati 1:1."""
    n = min(len(x), len(ref))
    x, ref = x[:n], ref[:n]
    f, t, Sx = sp.spectrogram(x, fs=SR, nperseg=nfft, noverlap=nfft//2)
    _, _, Sr = sp.spectrogram(ref, fs=SR, nperseg=nfft, noverlap=nfft//2)
    T = min(Sx.shape[1], Sr.shape[1])
    Sx, Sr = Sx[:, :T], Sr[:, :T]
    Px = Sx**2 + eps; Pr = Sr**2 + eps
    d = 10*np.log10(Px/Pr)  # per-frame log ratio
    # LSD medio = media su bande e frame
    return float(np.mean(np.abs(d))), float(np.mean(d))  # (media assoluta, bias)

def spectral_entropy_bands(x, bands=[(0,1000),(1000,4000),(4000,8000)]):
    f, t, S = sp.spectrogram(x, fs=SR, nperseg=1024, noverlap=512)
    P = S**2 + 1e-12
    P = P / P.sum(axis=0, keepdims=True)
    out = {}
    for lo, hi in bands:
        m = (f>=lo)&(f<hi)
        seg = P[m]
        if seg.size == 0: out[f"{lo}-{hi}"] = 0.0; continue
        ent = -(seg*np.log(seg+1e-12)).sum(axis=0)
        out[f"{lo}-{hi}"] = float(np.mean(ent))
    return out

def phase_coherence(x, n=512, h=256):
    """Coerenza di fase frame-to-frame: se i frame sono rumore caotico,
    la similarita' media tra frame adiacenti (correlazione) e' bassa."""
    frames=[]
    for st in range(0, len(x)-n, h):
        seg = x[st:st+n]; seg = seg - seg.mean()
        s = np.std(seg)
        if s > 1e-6: frames.append(seg/s)
    if len(frames) < 2: return 0.0
    corr = [float(np.dot(frames[i], frames[i+1])) for i in range(len(frames)-1)]
    return float(np.mean(corr))

def pitch_salience(x, n=1024):
    """Salienza armonica: autocorrelazione normalizzata max in lag 40-300 (f0 80-600Hz)."""
    best=[]
    for st in range(0, len(x)-n, n//2):
        seg = x[st:st+n]; seg = seg - seg.mean()
        if np.std(seg) < 1e-6: continue
        ac = np.correlate(seg, seg, 'full')[n-1:]
        ac = ac/ac[0]
        best.append(float(np.max(ac[40:300])))
    return float(np.mean(best)) if best else 0.0

def main():
    folder = "eval_syntest_v2_out"
    refs = {k: load(os.path.join(folder, f"eval_r2_gt_ref_{k}.wav")) for k in "abc"}
    res = {}
    for w in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        k = os.path.basename(w).split("_")[-1][0]
        x = load(w)
        lsd_abs, lsd_bias = lsd(x, refs[k])
        ent = spectral_entropy_bands(x)
        pc = phase_coherence(x)
        ps = pitch_salience(x)
        res[os.path.basename(w)] = {"lsd_abs_dB": round(lsd_abs,2), "lsd_bias_dB": round(lsd_bias,2),
                                    **{f"ent_{b}": round(v,3) for b,v in ent.items()},
                                    "phase_corr": round(pc,4), "pitch_sal": round(ps,3)}
        print(f"{os.path.basename(w):28s} LSD={lsd_abs:6.2f}dB (bias {lsd_bias:+6.2f}) "
              f"phase={pc:.3f} pitch={ps:.3f} | ent {[round(v,2) for v in ent.values()]}")
    with open(os.path.join(folder, "corruption_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nsalvato corruption_metrics.json")

if __name__ == "__main__":
    main()
