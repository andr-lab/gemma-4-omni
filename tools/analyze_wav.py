#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi spettrale e acustica del file WAV generato
"""
import numpy as np
import soundfile as sf

wav, sr = sf.read('samples/v7_sampled_00.wav')
print(f"File: samples/v7_sampled_00.wav")
print(f"Sample rate: {sr} Hz")
print(f"Shape: {wav.shape}")
print(f"Duration: {len(wav)/sr:.2f} s")
print(f"Min: {np.min(wav):.4f}, Max: {np.max(wav):.4f}")
print(f"RMS: {np.sqrt(np.mean(wav**2)):.4f}")
print(f"Zero Crossing Rate: {np.mean(np.abs(np.diff(np.sign(wav))))/2:.4f}")

# Calcolo FFT per spettro frequenziale medio
fft_vals = np.abs(np.fft.rfft(wav))
freqs = np.fft.rfftfreq(len(wav), 1.0/sr)

# Bande di frequenza:
# Sub-bass (<100Hz), Voce fondamentale (100-300Hz), Formanti (300-3000Hz), Alte (>3000Hz)
b_sub = np.sum(fft_vals[freqs < 100]**2)
b_f0 = np.sum(fft_vals[(freqs >= 100) & (freqs < 300)]**2)
b_form = np.sum(fft_vals[(freqs >= 300) & (freqs < 3000)]**2)
b_high = np.sum(fft_vals[freqs >= 3000]**2)
total_e = b_sub + b_f0 + b_form + b_high + 1e-12

print("\nRipartizione Spettrale dell'Energia:")
print(f"  • Sub-bass (<100 Hz)       : {b_sub/total_e*100:.2f}%")
print(f"  • Frequenza Fondamentale F0: {b_f0/total_e*100:.2f}%")
print(f"  • Formanti Vocali (0.3-3k) : {b_form/total_e*100:.2f}%")
print(f"  • Alte frequenze (>3 kHz)  : {b_high/total_e*100:.2f}%")
