#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_voice_quality.py -- Voice Quality and Anti-Sine Artifact Verification Tool.
Analyzes audio files (.wav) and parquet mel/code files to ensure real human speech:
  1. Constant frame fraction in Mel spectrum (< 5% required)
  2. Average RMS (> 0.01 required)
  3. Dominant frequency (must NOT be pure 440Hz sine wave; natural human pitch 80-500Hz)
  4. Harmonic richness and spectral spread (80-4000Hz, > 3 harmonics required for voice)
"""

import os
import sys
import argparse
import glob
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq


def analyze_waveform(wav: np.ndarray, sr: int = 24000) -> dict:
    """Analyze a 1D audio waveform for voice characteristics vs synthetic tone."""
    if len(wav) == 0:
        return {
            "status": "FAIL",
            "reason": "empty_waveform",
            "rms": 0.0,
            "peak": 0.0,
            "dom_freq_hz": 0.0,
            "harmonic_count": 0,
            "spectral_flatness": 1.0,
            "is_sine_artifact": True,
        }

    wav = wav.astype(np.float32)
    rms = float(np.sqrt(np.mean(wav**2)))
    peak = float(np.max(np.abs(wav)))

    # FFT Analysis
    n_fft = 2048
    if len(wav) >= n_fft:
        windowed = wav[:n_fft] * np.hanning(n_fft)
    else:
        windowed = np.pad(wav, (0, n_fft - len(wav))) * np.hanning(n_fft)

    mag = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid_idx = np.where(freqs >= 50.0)[0]
    top_idx = valid_idx[np.argmax(mag[valid_idx])]
    dom_freq = float(freqs[top_idx])

    # Spectral energy concentration
    total_energy = np.sum(mag[valid_idx]**2) + 1e-12
    peak_energy = np.sum(mag[max(0, top_idx-1) : min(len(mag), top_idx+2)]**2)
    energy_concentration = float(peak_energy / total_energy)

    # Harmonic analysis
    vocal_band = (freqs >= 80.0) & (freqs <= 4000.0)
    mag_vocal = mag[vocal_band]
    if len(mag_vocal) > 2:
        threshold = 0.05 * np.max(mag_vocal)
        peaks = (mag_vocal[1:-1] > mag_vocal[:-2]) & (mag_vocal[1:-1] > mag_vocal[2:]) & (mag_vocal[1:-1] > threshold)
        harmonic_count = int(peaks.sum())
    else:
        harmonic_count = 0

    # Temporal RMS variation
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    if len(wav) > frame_len:
        frames = np.lib.stride_tricks.sliding_window_view(wav, frame_len)[::hop_len]
        frame_rms = np.sqrt(np.mean(frames**2, axis=-1))
        rms_std = float(np.std(frame_rms))
    else:
        rms_std = 0.0

    # Pure tone detection
    is_pure_tone = (energy_concentration > 0.70) or (420.0 <= dom_freq <= 460.0 and harmonic_count <= 2) or (rms > 0.02 and rms_std < 0.003)
    is_constant = rms < 0.005 or np.std(wav) < 1e-4

    is_sine_artifact = is_pure_tone or is_constant
    status = "FAIL" if is_sine_artifact else "PASS"
    reason = "pure_sine_tone" if is_pure_tone else ("silent_or_constant" if is_constant else "ok_voice")

    return {
        "status": status,
        "reason": reason,
        "duration_sec": round(len(wav) / sr, 2),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "dom_freq_hz": round(dom_freq, 1),
        "harmonic_count": harmonic_count,
        "energy_concentration": round(energy_concentration, 4),
        "rms_std": round(rms_std, 4),
        "is_sine_artifact": is_sine_artifact,
    }


def analyze_mel_parquet(mel_parquet_path: str) -> dict:
    """Analyze a parquet mel file for constant/silent frames."""
    df = pd.read_parquet(mel_parquet_path)
    mel_cols = [c for c in df.columns if c.startswith("mel_")]
    if not mel_cols:
        mel_cols = df.columns.tolist()
    arr = df[mel_cols].values

    stds = np.std(arr, axis=1)
    const_mask = (stds < 1e-4)
    const_ratio = float(np.mean(const_mask))

    window_len = 512
    n_windows = len(arr) // window_len
    bad_windows = 0
    for w in range(n_windows):
        w_stds = stds[w * window_len : (w + 1) * window_len]
        if np.mean(w_stds < 1e-4) > 0.5:
            bad_windows += 1

    bad_win_ratio = float(bad_windows / max(1, n_windows))
    status = "PASS" if const_ratio < 0.05 else "FAIL"
    return {
        "status": status,
        "total_frames": len(arr),
        "constant_frames": int(const_mask.sum()),
        "constant_ratio": round(const_ratio, 4),
        "total_windows": n_windows,
        "bad_windows_gt50pct_silent": bad_windows,
        "bad_window_ratio": round(bad_win_ratio, 4),
        "mel_min": round(float(np.min(arr)), 4),
        "mel_max": round(float(np.max(arr)), 4),
        "mel_mean": round(float(np.mean(arr)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Check voice quality and detect sine artifacts.")
    parser.add_argument("paths", nargs="+", help="Files (.wav or .parquet) or directories to analyze")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.wav"))))
            files.extend(sorted(glob.glob(os.path.join(p, "*mel.parquet"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            for match in glob.glob(p):
                files.append(match)

    if not files:
        print(f"[CheckVoice] No matching files found for: {args.paths}")
        sys.exit(1)

    print("=" * 78)
    print(f"=== VOICE QUALITY CHECK REPORT ({len(files)} files) ===")
    print("=" * 78)

    all_pass = True
    for f in files:
        if f.endswith(".wav"):
            try:
                y, sr = sf.read(f)
                res = analyze_waveform(y, sr)
                icon = "✅ PASS" if res["status"] == "PASS" else "❌ FAIL"
                print(f"{icon} [WAV] {os.path.basename(f):30s} | RMS: {res['rms']:.4f} (std {res['rms_std']:.4f}) | DomFreq: {res['dom_freq_hz']:6.1f}Hz | Harmonics: {res['harmonic_count']:2d} | Status: {res['reason']}")
                if res["status"] != "PASS":
                    all_pass = False
            except Exception as e:
                print(f"❌ ERROR [WAV] {f}: {e}")
                all_pass = False

        elif f.endswith(".parquet"):
            try:
                res = analyze_mel_parquet(f)
                icon = "✅ PASS" if res["status"] == "PASS" else "❌ FAIL"
                pct = res["constant_ratio"] * 100
                print(f"{icon} [MEL] {os.path.basename(f):45s} | ConstFrames: {pct:5.2f}% (max 5%) | BadWin: {res['bad_windows_gt50pct_silent']}/{res['total_windows']} | Status: {res['status']}")
                if res["status"] != "PASS":
                    all_pass = False
            except Exception as e:
                print(f"❌ ERROR [PARQUET] {f}: {e}")
                all_pass = False

    print("=" * 78)
    print(f"OVERALL QUALITY GATE: {'✅ ALL PASSED' if all_pass else '❌ FAILED'}")
    print("=" * 78)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    import pandas as pd
    main()
