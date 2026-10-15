#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analyze_and_transcribe_sample.py
Measure acoustic quality (Crest factor, THD+N, HNR, Flatness, Rumble) and ASR WER using faster-whisper.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal as signal


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    return float(d[len(ref_words), len(hyp_words)] / len(ref_words))


def analyze_distortion(data: np.ndarray, sr: int) -> dict:
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 1. RMS and Crest Factor
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    crest = float(peak / (rms + 1e-9))

    # 2. Spectral Flatness
    f, t_spec, Zxx = signal.stft(data, sr, nperseg=1024, noverlap=512)
    psd = np.abs(Zxx)**2 + 1e-12
    geom_mean = np.exp(np.mean(np.log(psd), axis=0))
    arith_mean = np.mean(psd, axis=0)
    spectral_flatness = float(np.mean(geom_mean / arith_mean))

    # 3. Harmonics-to-Noise Ratio (HNR) via Autocorrelation
    frame_len = int(0.040 * sr)
    hop = int(0.020 * sr)
    hnr_list = []
    min_lag = int(sr / 400.0)
    max_lag = int(sr / 60.0)

    for start in range(0, len(data) - frame_len, hop):
        frame = data[start : start + frame_len]
        if np.sqrt(np.mean(frame**2)) < 0.01:
            continue

        frame_win = frame * np.hanning(len(frame))
        ac = np.correlate(frame_win, frame_win, mode="full")
        ac = ac[len(frame_win) - 1 :]
        r0 = ac[0] + 1e-12
        ac_norm = ac / r0

        if max_lag >= len(ac_norm):
            max_lag_cur = len(ac_norm) - 1
        else:
            max_lag_cur = max_lag

        if min_lag >= max_lag_cur:
            continue

        pitch_peak = float(np.max(ac_norm[min_lag:max_lag_cur]))
        pitch_peak = float(np.clip(pitch_peak, 1e-4, 0.9999))
        hnr = float(10.0 * np.log10(pitch_peak / (1.0 - pitch_peak)))
        hnr_list.append(hnr)

    avg_hnr = float(np.mean(hnr_list)) if len(hnr_list) > 0 else 0.0
    thd_n_pct = float(100.0 / (1.0 + 10.0 ** (avg_hnr / 10.0))) if avg_hnr > -50 else 100.0

    # 4. Low Frequency Rumble (< 80 Hz)
    fft_vals = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
    total_power = float(np.sum(fft_vals**2) + 1e-12)
    rumble_power = float(np.sum(fft_vals[freqs < 80] ** 2))
    rumble_pct = float((rumble_power / total_power) * 100.0)

    # 5. 12.5 Hz periodic modulation
    env = np.abs(signal.hilbert(data))
    env_fft = np.abs(np.fft.rfft(env - np.mean(env)))
    env_freqs = np.fft.rfftfreq(len(env), 1.0 / sr)
    idx_12hz = (env_freqs >= 11.5) & (env_freqs <= 13.5)
    mod_12hz_pct = float(np.sum(env_fft[idx_12hz]) / (np.sum(env_fft) + 1e-12) * 100.0)

    return {
        "rms": round(rms, 4),
        "crest_factor": round(crest, 2),
        "hnr_db": round(avg_hnr, 2),
        "thd_n_pct": round(thd_n_pct, 2),
        "spectral_flatness": round(spectral_flatness, 4),
        "rumble_pct": round(rumble_pct, 2),
        "mod_12hz_pct": round(mod_12hz_pct, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio distortion and ASR intelligibility.")
    parser.add_argument("wav_files", nargs="+", help="WAV files to evaluate")
    parser.add_argument("--ref", type=str, default=None, help="Ground truth reference text")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results = {}

    # Initialize Whisper if available
    asr_model = None
    try:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[ASR] Loading Faster-Whisper 'small' on {device} ({compute_type})...", flush=True)
        asr_model = WhisperModel("small", device=device, compute_type=compute_type)
    except Exception as exc:
        print(f"[ASR] Whisper not loaded: {exc}", flush=True)

    for filepath in args.wav_files:
        p = Path(filepath)
        if not p.is_file():
            print(f"File not found: {p}", file=sys.stderr)
            continue

        data, sr = sf.read(str(p))
        dist_metrics = analyze_distortion(data, sr)

        asr_text = ""
        wer = None
        if asr_model is not None:
            try:
                segments, _ = asr_model.transcribe(str(p), language="it")
                asr_text = " ".join(s.text.strip() for s in segments).strip()
                if args.ref:
                    wer = round(calculate_wer(args.ref, asr_text), 4)
            except Exception as exc:
                asr_text = f"ASR_ERROR: {exc}"

        results[p.name] = {
            "path": str(p),
            "distortion": dist_metrics,
            "asr_transcript": asr_text,
            "wer": wer,
        }

        print(f"\n=== Evaluation: {p.name} ===")
        for k, v in dist_metrics.items():
            print(f"  {k:20s}: {v}")
        print(f"  {'asr_transcript':20s}: \"{asr_text}\"")
        if wer is not None:
            print(f"  {'wer':20s}: {wer:.2%}")

    if args.json:
        out_p = Path(args.json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved report to {out_p}")


if __name__ == "__main__":
    main()
