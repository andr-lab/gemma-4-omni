# -*- coding: utf-8 -*-
"""dialogue_eval/audio_metrics.py

Audio quality and physical fidelity metrics for full-duplex conversations.

Measures:
  - clipping_pct: percentage of samples where |x| >= 0.999 (gate: <= 0.1%)
  - rms: root-mean-square amplitude
  - crest_factor: peak / (RMS + eps) (target: 5.0 - 8.5; alarm if < 3.5)
  - dc_offset: mean amplitude offset
  - peak_abs: maximum absolute amplitude
  - anomalous_silences: detects gaps > 320 ms in active speech regions
  - packet_discontinuities: detects sample level jumps across packet boundaries
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np


def compute_physical_audio_metrics(
    pcm: np.ndarray,
    sample_rate: int = 16_000,
    *,
    packet_boundary_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Computes physical audio metrics on a PCM signal."""
    array = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return {
            "samples": 0,
            "duration_s": 0.0,
            "finite": True,
            "clipping_pct": 0.0,
            "rms": 0.0,
            "crest_factor": 0.0,
            "dc_offset": 0.0,
            "peak_abs": 0.0,
            "anomalous_silences_count": 0,
            "packet_discontinuities_count": 0,
            "max_discontinuity_jump": 0.0,
        }

    is_finite = bool(np.isfinite(array).all())
    if not is_finite:
        return {
            "samples": int(array.size),
            "duration_s": round(float(array.size) / sample_rate, 4),
            "finite": False,
            "clipping_pct": 100.0,
            "rms": 999.0,
            "crest_factor": 0.0,
            "dc_offset": 999.0,
            "peak_abs": 999.0,
            "anomalous_silences_count": 0,
            "packet_discontinuities_count": 0,
            "max_discontinuity_jump": 999.0,
        }

    peak_abs = float(np.max(np.abs(array)))
    clipping_count = int(np.sum(np.abs(array) >= 0.999))
    clipping_pct = round(100.0 * float(clipping_count) / float(array.size), 4)

    dc_offset = float(np.mean(array))
    # Centered array for true AC RMS
    ac_signal = array - dc_offset
    rms = float(np.sqrt(np.mean(ac_signal ** 2)))
    crest_factor = round(peak_abs / (rms + 1e-9), 3) if rms > 1e-7 else 0.0

    # Anomalous silence detection: window size = 80 ms (frame size), step = 40 ms
    frame_len = int(0.080 * sample_rate)
    hop_len = int(0.040 * sample_rate)
    silence_threshold = 0.005  # below -46 dBFS

    silence_anomalies = 0
    consecutive_silent_frames = 0
    max_allowed_consecutive_silence = int(0.320 / 0.040)  # 320 ms gap threshold

    if array.size >= frame_len:
        # Determine speech envelope
        num_windows = (array.size - frame_len) // hop_len + 1
        for i in range(num_windows):
            start = i * hop_len
            chunk = array[start:start + frame_len]
            chunk_rms = np.sqrt(np.mean(chunk ** 2))
            if chunk_rms < silence_threshold:
                consecutive_silent_frames += 1
                if consecutive_silent_frames == max_allowed_consecutive_silence + 1:
                    silence_anomalies += 1
            else:
                consecutive_silent_frames = 0

    # Packet boundary discontinuities
    packet_discontinuities = 0
    max_jump = 0.0
    jump_threshold = 0.35  # sudden jump in normalized amplitude > 0.35 indicates phase click / drop

    if packet_boundary_indices:
        for idx in packet_boundary_indices:
            if 0 < idx < array.size:
                jump = float(abs(array[idx] - array[idx - 1]))
                max_jump = max(max_jump, jump)
                if jump > jump_threshold:
                    packet_discontinuities += 1

    return {
        "samples": int(array.size),
        "duration_s": round(float(array.size) / sample_rate, 4),
        "finite": is_finite,
        "clipping_pct": clipping_pct,
        "rms": round(rms, 5),
        "crest_factor": crest_factor,
        "dc_offset": round(dc_offset, 6),
        "peak_abs": round(peak_abs, 5),
        "anomalous_silences_count": silence_anomalies,
        "packet_discontinuities_count": packet_discontinuities,
        "max_discontinuity_jump": round(max_jump, 4),
    }
