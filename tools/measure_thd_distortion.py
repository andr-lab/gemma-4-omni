#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure THD+N, HNR (Harmonics-to-Noise Ratio), Spectral Flatness, and Distortion
"""

import os, sys
import numpy as np
import soundfile as sf
import scipy.signal as signal

def analyze_audio_distortion(filepath):
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    # 1. RMS and Peak
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    crest = float(peak / (rms + 1e-9))
    
    # 2. Spectral Flatness (Wiener Entropy)
    f, t_spec, Zxx = signal.stft(data, sr, nperseg=1024, noverlap=512)
    psd = np.abs(Zxx)**2 + 1e-12
    geom_mean = np.exp(np.mean(np.log(psd), axis=0))
    arith_mean = np.mean(psd, axis=0)
    spectral_flatness = float(np.mean(geom_mean / arith_mean))
    
    # 3. Harmonics-to-Noise Ratio (HNR) via Autocorrelation
    frame_len = int(0.040 * sr) # 40ms frames
    hop = int(0.020 * sr)
    hnr_list = []
    
    min_lag = int(sr / 400.0) # 400 Hz max pitch
    max_lag = int(sr / 60.0)  # 60 Hz min pitch
    
    for start in range(0, len(data) - frame_len, hop):
        frame = data[start:start+frame_len]
        if np.sqrt(np.mean(frame**2)) < 0.01:
            continue # skip silence
        
        frame_win = frame * np.hanning(len(frame))
        ac = np.correlate(frame_win, frame_win, mode='full')
        ac = ac[len(frame_win)-1:]
        r0 = ac[0] + 1e-12
        ac_norm = ac / r0
        
        pitch_peak = float(np.max(ac_norm[min_lag:max_lag]))
        pitch_peak = float(np.clip(pitch_peak, 1e-4, 0.9999))
        hnr = float(10.0 * np.log10(pitch_peak / (1.0 - pitch_peak)))
        hnr_list.append(hnr)
        
    avg_hnr = float(np.mean(hnr_list)) if len(hnr_list) > 0 else 0.0
    
    # 4. Total Harmonic Distortion + Noise (THD+N)
    thd_n_pct = float(100.0 / (1.0 + 10.0**(avg_hnr / 10.0))) if avg_hnr > -50 else 100.0
    
    # 5. Low Frequency Rumble (< 80 Hz)
    fft_vals = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0/sr)
    total_power = float(np.sum(fft_vals**2) + 1e-12)
    rumble_power = float(np.sum(fft_vals[freqs < 80]**2))
    rumble_pct = float((rumble_power / total_power) * 100.0)
    
    # 6. High frequency noise (> 4000 Hz)
    high_power = float(np.sum(fft_vals[freqs > 4000]**2))
    high_pct = float((high_power / total_power) * 100.0)
    
    # 7. 12.5 Hz Periodic Subharmonic Modulation (Pulsation)
    env = np.abs(signal.hilbert(data))
    env_fft = np.abs(np.fft.rfft(env - np.mean(env)))
    env_freqs = np.fft.rfftfreq(len(env), 1.0/sr)
    idx_12hz = (env_freqs >= 11.5) & (env_freqs <= 13.5)
    mod_12hz_ratio = float(np.sum(env_fft[idx_12hz]) / (np.sum(env_fft) + 1e-12) * 100.0)
    
    return {
        'rms': rms,
        'crest_factor': crest,
        'hnr_db': avg_hnr,
        'thd_n_pct': thd_n_pct,
        'spectral_flatness': spectral_flatness,
        'rumble_pct': rumble_pct,
        'high_pct': high_pct,
        'mod_12hz_pct': mod_12hz_ratio
    }

files = [
    ('Native Qwen Reference', 'samples_eval/sample_1_native_qwen_reference.wav'),
    ('Oracle GT Residuals', 'samples_eval/exp_b_clean_gt_residuals.wav'),
    ('V9 Step 1000 Omni', 'samples_eval/sample_1_omni_v9_step1000.wav'),
    ('Exp A CodePredictor', 'samples_eval/exp_a_official_codepredictor.wav'),
    ('Sample 1 Calib RMS', 'samples_eval/sample_1_calib_rms_t09.wav'),
    ('Sample 3 Calib T03', 'samples_eval/sample_3_calib_rms_t03.wav'),
    ('Sample 4 Calib Greedy', 'samples_eval/sample_4_calib_rms_greedy.wav'),
]

print("| File | RMS | Crest | HNR (dB) | THD+N (%) | Flatness | Rumble (<80Hz) | 12.5Hz Pulse |")
print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

for label, f in files:
    if os.path.exists(f):
        m = analyze_audio_distortion(f)
        print(f"| **{label}** | {m['rms']:.3f} | {m['crest_factor']:.2f} | **{m['hnr_db']:.1f} dB** | **{m['thd_n_pct']:.1f}%** | {m['spectral_flatness']:.4f} | {m['rumble_pct']:.1f}% | {m['mod_12hz_pct']:.2f}% |")
    else:
        print(f"| {label} | File non trovato | | | | | | |")
