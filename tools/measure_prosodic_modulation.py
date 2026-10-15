#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/measure_prosodic_modulation.py
Objective acoustic prosody and dynamics analyzer for synthesized speech waveforms.

Measures:
1. Fundamental Frequency (F0) Pitch Contour & Dynamics:
   - Mean F0, Median F0, Min/Max F0 (5th-95th percentiles)
   - Pitch Standard Deviation (intonation variability)
   - Pitch Range (semitones: 12 * log2(F0_95 / F0_05))
   - Voiced frame fraction
2. Energy & Dynamic Range:
   - Frame-level RMS energy (mean, std)
   - Dynamic Range (dB: 20 * log10(RMS_95 / (RMS_05 + eps)))
   - Crest Factor (Peak / RMS)
   - Clipping percentage (|x| >= 0.999)
3. Differential Prosodic Modulation:
   - Compares unconditioned baseline vs. continuously conditioned speech.
   - Measures modulation intensity, pitch contour shifts, and dynamic expansion.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf


def extract_f0_contour(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    f0_min: float = 65.0,
    f0_max: float = 450.0,
    voicing_threshold: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract frame-level F0 contour using normalized autocorrelation.
    Returns:
        times: timestamps of frame centers (seconds)
        f0_hz: pitch in Hz (0.0 for unvoiced frames)
    """
    frame_len = int(sr * (frame_ms / 1000.0))
    hop_len = int(sr * (hop_ms / 1000.0))
    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)

    num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
    f0_hz = np.zeros(num_frames, dtype=np.float32)
    times = np.zeros(num_frames, dtype=np.float32)

    window = np.hanning(frame_len)

    for i in range(num_frames):
        start = i * hop_len
        times[i] = (start + frame_len // 2) / sr
        frame = audio[start : start + frame_len]
        if len(frame) < frame_len:
            break

        frame_rms = float(np.sqrt(np.mean(frame**2)))
        if frame_rms < 0.005:  # silence gate
            continue

        win_frame = frame * window
        # Autocorrelation
        corr = np.correlate(win_frame, win_frame, mode="full")
        corr = corr[len(win_frame) - 1 :]
        r0 = corr[0] + 1e-12
        norm_corr = corr / r0

        if max_lag >= len(norm_corr):
            max_lag_cur = len(norm_corr) - 1
        else:
            max_lag_cur = max_lag

        if min_lag >= max_lag_cur:
            continue

        search_corr = norm_corr[min_lag:max_lag_cur]
        peak_idx = int(np.argmax(search_corr)) + min_lag
        peak_val = float(norm_corr[peak_idx])

        # Parabolic interpolation for sub-sample lag accuracy
        if 0 < peak_idx < len(norm_corr) - 1:
            alpha = float(norm_corr[peak_idx - 1])
            beta = float(norm_corr[peak_idx])
            gamma = float(norm_corr[peak_idx + 1])
            denom = 2.0 * (alpha - 2.0 * beta + gamma)
            if abs(denom) > 1e-12:
                refined_lag = peak_idx + (alpha - gamma) / denom
            else:
                refined_lag = float(peak_idx)
        else:
            refined_lag = float(peak_idx)

        if peak_val >= voicing_threshold and refined_lag > 0:
            pitch = float(sr / refined_lag)
            if f0_min <= pitch <= f0_max:
                f0_hz[i] = pitch

    return times, f0_hz


def extract_prosodic_features(
    audio_path_or_array: Union[str, Path, np.ndarray],
    sr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive prosodic and dynamic features from an audio file or numpy array.
    """
    if isinstance(audio_path_or_array, (str, Path)):
        filepath = Path(audio_path_or_array)
        if not filepath.is_file():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        data, file_sr = sf.read(str(filepath))
        sr = file_sr
    else:
        data = np.asarray(audio_path_or_array, dtype=np.float32)
        if sr is None:
            raise ValueError("sample rate must be provided when passing a numpy array")

    if data.ndim > 1:
        data = data.mean(axis=1)

    total_samples = len(data)
    duration_s = float(total_samples / sr)

    # 1. Peak and clipping checks
    peak_amp = float(np.max(np.abs(data)))
    clipping_samples = int(np.sum(np.abs(data) >= 0.999))
    clipping_pct = float((clipping_samples / max(1, total_samples)) * 100.0)

    # 2. Overall RMS and Crest Factor
    overall_rms = float(np.sqrt(np.mean(data**2)))
    crest_factor = float(peak_amp / (overall_rms + 1e-12))

    # 3. Frame-level RMS energy (20ms frames, 10ms hop)
    frame_len_energy = int(sr * 0.020)
    hop_len_energy = int(sr * 0.010)
    num_energy_frames = max(1, (len(data) - frame_len_energy) // hop_len_energy + 1)
    frame_rms = np.zeros(num_energy_frames, dtype=np.float32)

    for i in range(num_energy_frames):
        start = i * hop_len_energy
        chunk = data[start : start + frame_len_energy]
        frame_rms[i] = float(np.sqrt(np.mean(chunk**2)))

    active_rms = frame_rms[frame_rms > 0.005]
    if len(active_rms) > 0:
        rms_p05 = float(np.percentile(active_rms, 5))
        rms_p95 = float(np.percentile(active_rms, 95))
        dynamic_range_db = float(20.0 * np.log10((rms_p95 + 1e-12) / (rms_p05 + 1e-12)))
        energy_mean = float(np.mean(active_rms))
        energy_std = float(np.std(active_rms))
    else:
        dynamic_range_db = 0.0
        energy_mean = 0.0
        energy_std = 0.0

    # 4. Fundamental Frequency (F0)
    times, f0_hz = extract_f0_contour(data, sr)
    voiced_f0 = f0_hz[f0_hz > 0.0]
    total_frames = len(f0_hz)
    voiced_frames = len(voiced_f0)
    voiced_fraction = float(voiced_frames / max(1, total_frames))

    if voiced_frames > 2:
        mean_f0 = float(np.mean(voiced_f0))
        std_f0 = float(np.std(voiced_f0))
        median_f0 = float(np.median(voiced_f0))
        f0_p05 = float(np.percentile(voiced_f0, 5))
        f0_p95 = float(np.percentile(voiced_f0, 95))
        pitch_range_hz = float(f0_p95 - f0_p05)
        pitch_range_semitones = float(12.0 * np.log2((f0_p95 + 1e-12) / (f0_p05 + 1e-12)))
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
        median_f0 = 0.0
        f0_p05 = 0.0
        f0_p95 = 0.0
        pitch_range_hz = 0.0
        pitch_range_semitones = 0.0

    return {
        "duration_seconds": round(duration_s, 3),
        "sample_rate": sr,
        "f0_mean_hz": round(mean_f0, 2),
        "f0_std_hz": round(std_f0, 2),
        "f0_median_hz": round(median_f0, 2),
        "f0_p05_hz": round(f0_p05, 2),
        "f0_p95_hz": round(f0_p95, 2),
        "pitch_range_hz": round(pitch_range_hz, 2),
        "pitch_range_semitones": round(pitch_range_semitones, 2),
        "voiced_fraction": round(voiced_fraction, 3),
        "energy_mean_rms": round(energy_mean, 4),
        "energy_std_rms": round(energy_std, 4),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "peak_amplitude": round(peak_amp, 4),
        "crest_factor": round(crest_factor, 2),
        "clipping_pct": round(clipping_pct, 4),
    }


def compare_prosody(
    baseline_audio: Union[str, Path, np.ndarray],
    conditioned_audio: Union[str, Path, np.ndarray],
    sr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare baseline unconditioned audio vs conditioned audio.
    """
    base_feat = extract_prosodic_features(baseline_audio, sr)
    cond_feat = extract_prosodic_features(conditioned_audio, sr)

    delta_f0_mean = round(cond_feat["f0_mean_hz"] - base_feat["f0_mean_hz"], 2)
    delta_f0_std = round(cond_feat["f0_std_hz"] - base_feat["f0_std_hz"], 2)
    delta_pitch_range_st = round(cond_feat["pitch_range_semitones"] - base_feat["pitch_range_semitones"], 2)
    delta_dynamic_range_db = round(cond_feat["dynamic_range_db"] - base_feat["dynamic_range_db"], 2)
    delta_crest = round(cond_feat["crest_factor"] - base_feat["crest_factor"], 2)

    return {
        "baseline": base_feat,
        "conditioned": cond_feat,
        "deltas": {
            "delta_f0_mean_hz": delta_f0_mean,
            "delta_f0_std_hz": delta_f0_std,
            "delta_pitch_range_semitones": delta_pitch_range_st,
            "delta_dynamic_range_db": delta_dynamic_range_db,
            "delta_crest_factor": delta_crest,
        },
        "healthy_speech_checks": {
            "clipping_safe": bool(cond_feat["clipping_pct"] <= 0.05),
            "crest_factor_natural": bool(4.0 <= cond_feat["crest_factor"] <= 9.0),
            "pitch_range_active": bool(cond_feat["pitch_range_semitones"] >= 2.0),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Measure objective acoustic prosody on synthesized speech.")
    parser.add_argument("audio", type=str, help="Path to target audio file (WAV)")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline WAV for comparative analysis")
    parser.add_argument("--json", type=str, default=None, help="Save output JSON metrics to file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"Error: Target file {audio_path} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.baseline:
        base_path = Path(args.baseline)
        if not base_path.is_file():
            print(f"Error: Baseline file {base_path} does not exist", file=sys.stderr)
            sys.exit(1)
        result = compare_prosody(base_path, audio_path)
        print("=== Comparative Prosodic Analysis ===")
        print(f"Baseline:    {base_path.name}")
        print(f"Conditioned: {audio_path.name}")
        print("\n[Baseline Features]")
        for k, v in result["baseline"].items():
            print(f"  {k:28s}: {v}")
        print("\n[Conditioned Features]")
        for k, v in result["conditioned"].items():
            print(f"  {k:28s}: {v}")
        print("\n[Deltas (Conditioned - Baseline)]")
        for k, v in result["deltas"].items():
            print(f"  {k:28s}: {v:+}")
        print("\n[Acoustic Health Checks]")
        for k, v in result["healthy_speech_checks"].items():
            print(f"  {k:28s}: {'PASS' if v else 'FAIL'}")
    else:
        result = extract_prosodic_features(audio_path)
        print(f"=== Prosodic Features: {audio_path.name} ===")
        for k, v in result.items():
            print(f"  {k:28s}: {v}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
