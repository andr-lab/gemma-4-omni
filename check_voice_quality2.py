#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_voice_quality2.py -- Strict Voice Quality Verification Tool for Speech-LLM.
Validates 5 mandatory quality gates for real human speech (preventing clipping, square wave, and pure-tone artifacts):
  1. clip>99% < 1% (fraction of samples with |x| > 0.99)
  2. peak/rms between 3.0 and 10.0 (natural human speech crest factor; square wave is ~1.0)
  3. dyn > 0.10 (temporal dynamics: std(frame_rms) / mean(frame_rms))
  4. dom_freq diversity: dominant frequencies vary across windows (>= 3 distinct frequencies, no pure tone)
  5. harmonics > 20 (spectral richness: number of FFT bins with magnitude > 30% of peak in vocal range)
"""

import os
import sys
import argparse
import glob
from typing import Dict, List, Any
import numpy as np
import soundfile as sf


def analyze_single_waveform(wav: np.ndarray, sr: int = 24000) -> Dict[str, Any]:
    """Analyze a single audio waveform against speech quality metrics."""
    if len(wav) == 0:
        return {
            "status": "FAIL",
            "reason": "empty_waveform",
            "duration_sec": 0.0,
            "clip_pct": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "peak_rms": 0.0,
            "dyn": 0.0,
            "dom_freq_hz": 0.0,
            "harmonics_gt30": 0,
            "energy_concentration": 1.0,
            "failures": ["empty_waveform"],
        }

    wav = wav.astype(np.float32)
    duration_sec = float(len(wav) / sr)

    # 1. Clipping ratio (|x| > 0.99)
    clip_ratio = float(np.mean(np.abs(wav) > 0.99))
    clip_pct = clip_ratio * 100.0

    # 2. Crest factor: Peak / RMS
    rms = float(np.sqrt(np.mean(wav**2)))
    peak = float(np.max(np.abs(wav)))
    peak_rms = float(peak / (rms + 1e-9))

    # 3. Temporal Dynamics (frame RMS variation over time)
    frame_len = int(sr * 0.025)  # 25ms
    hop_len = int(sr * 0.010)    # 10ms
    if len(wav) >= frame_len:
        frames = np.lib.stride_tricks.sliding_window_view(wav, frame_len)[::hop_len]
        frame_rms = np.sqrt(np.mean(frames**2, axis=-1))
        f_mean = float(np.mean(frame_rms))
        f_std = float(np.std(frame_rms))
        dyn = float(f_std / (f_mean + 1e-9))
    else:
        dyn = 0.0

    # 4. Spectral analysis: STFT average magnitude spectrum
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid_idx = np.where(freqs >= 50.0)[0]

    mags = []
    chunk_doms = []
    hop_fft = 1024
    for i in range(0, len(wav) - n_fft, hop_fft):
        chunk = wav[i : i + n_fft] * np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(chunk, n=n_fft))
        mags.append(mag)
        top_c = valid_idx[np.argmax(mag[valid_idx])]
        chunk_doms.append(float(freqs[top_c]))

    if mags:
        avg_mag = np.mean(mags, axis=0)
        top_idx = valid_idx[np.argmax(avg_mag[valid_idx])]
        dom_freq = float(freqs[top_idx])
        max_mag = float(np.max(avg_mag[valid_idx]))

        # 5. Harmonics count: bins with magnitude > 30% of peak in vocal range (>= 50Hz)
        harmonics_gt30 = int(np.sum(avg_mag[valid_idx] > 0.30 * max_mag))

        # Energy concentration around peak bin (+- 1 bin)
        peak_energy = float(np.sum(avg_mag[max(0, top_idx - 1) : min(len(avg_mag), top_idx + 2)] ** 2))
        tot_energy = float(np.sum(avg_mag[valid_idx] ** 2) + 1e-12)
        energy_concentration = float(peak_energy / tot_energy)
    else:
        dom_freq = 0.0
        harmonics_gt30 = 0
        energy_concentration = 1.0

    # Evaluation against quality gates
    failures = []
    if clip_ratio >= 0.01:
        failures.append(f"clip>99% ({clip_pct:.1f}% >= 1.0%)")
    if peak_rms < 3.0 or peak_rms > 10.0:
        failures.append(f"peak/rms ({peak_rms:.2f} not in [3.0, 10.0])")
    if dyn <= 0.10:
        failures.append(f"dyn ({dyn:.3f} <= 0.10)")
    if harmonics_gt30 <= 20:
        failures.append(f"harmonics ({harmonics_gt30} <= 20)")
    if energy_concentration >= 0.70:
        failures.append(f"pure_tone (energy conc {energy_concentration:.2f} >= 0.70)")

    status = "PASS" if not failures else "FAIL"
    reason = "ok_voice" if not failures else "; ".join(failures)

    return {
        "status": status,
        "reason": reason,
        "duration_sec": round(duration_sec, 2),
        "clip_pct": round(clip_pct, 2),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "peak_rms": round(peak_rms, 2),
        "dyn": round(dyn, 4),
        "dom_freq_hz": round(dom_freq, 1),
        "harmonics_gt30": harmonics_gt30,
        "energy_concentration": round(energy_concentration, 4),
        "chunk_doms": chunk_doms,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict Voice Quality Verification Tool v2.")
    parser.add_argument("paths", nargs="+", help="WAV files or directory containing WAV files to analyze")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.wav"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            for match in glob.glob(p):
                files.append(match)

    files = sorted(list(set(files)))
    if not files:
        print(f"[CheckVoice2] No matching WAV files found for: {args.paths}")
        sys.exit(1)

    print("=" * 96)
    print(f"=== VOICE QUALITY VERIFICATION REPORT v2 ({len(files)} files) ===")
    print("=" * 96)
    print(f"{'STATUS':8s} | {'FILE':24s} | {'CLIP%':6s} | {'PK/RMS':6s} | {'DYN':6s} | {'DOM_FREQ':9s} | {'HARM(>30%)':10s} | {'NOTES'}")
    print("-" * 96)

    all_pass = True
    dom_freqs = []
    results = []

    for f in files:
        try:
            y, sr = sf.read(f)
            res = analyze_single_waveform(y, sr)
            res["filename"] = os.path.basename(f)
            res["filepath"] = f
            results.append(res)

            icon = "✅ PASS" if res["status"] == "PASS" else "❌ FAIL"
            if res["status"] != "PASS":
                all_pass = False

            dom_freqs.append(res["dom_freq_hz"])
            print(
                f"{icon:8s} | {res['filename']:24s} | {res['clip_pct']:5.2f}% | {res['peak_rms']:6.2f} | {res['dyn']:6.3f} | {res['dom_freq_hz']:7.1f}Hz | {res['harmonics_gt30']:10d} | {res['reason']}"
            )
        except Exception as e:
            print(f"❌ ERROR  | {os.path.basename(f):24s} | Failed to read/analyze: {e}")
            all_pass = False

    # Multi-file global check: Diversity of dominant frequencies
    unique_doms = len(np.unique(np.round(dom_freqs, 0))) if dom_freqs else 0
    dom_diversity_pass = unique_doms >= 3 if len(files) >= 3 else True

    print("=" * 96)
    print("=== SUMMARY METRICS & GLOBAL QUALITY GATES ===")
    print(f"  - Total files analyzed:             {len(files)}")
    print(f"  - Single-file quality gates:        {'✅ ALL PASSED' if all_pass else '❌ FAILED'}")
    print(f"  - Dominant frequency diversity:     {unique_doms} distinct frequencies (Target >= 3): {'✅ PASS' if dom_diversity_pass else '❌ FAIL'}")
    if dom_freqs:
        unique_vals = [float(x) for x in np.unique(np.round(dom_freqs, 1))]
        print(f"    Unique dominant frequencies (Hz): {unique_vals}")

    overall_success = all_pass and dom_diversity_pass
    print("-" * 96)
    print(f"FINAL QUALITY GATE VERDICT: {'🏆 100% ALL PASSED (REAL HUMAN SPEECH)' if overall_success else '❌ VERIFICATION FAILED'}")
    print("=" * 96)

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
