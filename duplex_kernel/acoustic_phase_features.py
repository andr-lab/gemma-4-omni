"""Causal acoustic phase features computed from observed PCM only.

The input history ends at the current user tick.  No labels, controller state,
future PCM, or outgoing audio are accepted by this helper.  RMS is computed on
16 kHz float PCM in [-1, 1] using float64 accumulation, then converted with
log10(max(rms, epsilon)) and clipped to [-6, 0].
"""
from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE_RATE_HZ = 16_000
TICK_MS = 80
TICK_SAMPLES = SAMPLE_RATE_HZ * TICK_MS // 1000
HISTORY_MS = 400
HISTORY_SAMPLES = SAMPLE_RATE_HZ * HISTORY_MS // 1000
EPSILON = 1e-6

ACOUSTIC_PHASE_FEATURE_NAMES = (
    "user_log10_rms_current_80ms",
    "user_log10_rms_past_400ms_inclusive",
    "assistant_log10_rms_prior_80ms",
    "user_log_rms_delta_clip6_div6",
)

ACOUSTIC_PHASE_FEATURE_CONTRACT = {
    "schema_version": 1,
    "names": list(ACOUSTIC_PHASE_FEATURE_NAMES),
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "tick_ms": TICK_MS,
    "history_ms": HISTORY_MS,
    "epsilon": EPSILON,
    "log10_rms_clip": [-6.0, 0.0],
    "log_rms_delta_clip": [-6.0, 6.0],
    "log_rms_delta_divisor": 6.0,
    "history_padding": "left_zero_pad_at_stream_start_to_400ms",
    "assistant_reference": "previous_input_tick_t_minus_1_only",
    "pcm_domain": "mono_float32_normalized_pcm",
}


def _pcm_vector(value: Any, *, name: str, maximum_samples: int | None = None) -> np.ndarray:
    try:
        pcm = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric PCM vector") from exc
    if pcm.ndim != 1:
        raise ValueError(f"{name} must be a mono 1D PCM vector, got shape {pcm.shape}")
    if maximum_samples is not None and len(pcm) > maximum_samples:
        raise ValueError(f"{name} exceeds the {maximum_samples}-sample causal window")
    if not np.isfinite(pcm).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return pcm


def _log10_rms(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        rms = 0.0
    else:
        values = pcm.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(values * values, dtype=np.float64)))
    return float(np.clip(np.log10(max(rms, EPSILON)), -6.0, 0.0))


def acoustic_phase_features(
    user_history_pcm: Any,
    assistant_previous_pcm: Any | None,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    tick_ms: int = TICK_MS,
) -> np.ndarray:
    """Return four causal acoustic features for input tick ``t``.

    ``user_history_pcm`` is the chronological user PCM ending at tick ``t``;
    it may contain 1--5 ticks so startup history can be represented without
    future samples.  Missing history before the first observed tick is padded
    on the left with zeros.  ``assistant_previous_pcm`` is exactly the
    assistant-reference PCM for input tick ``t-1`` or ``None`` when unavailable.

    The output order is ``ACOUSTIC_PHASE_FEATURE_NAMES``.  Its first three
    values are clipped log10 RMS values; the fourth is the clipped difference
    between current and previous user-tick log10 RMS, divided by six.
    """
    if sample_rate_hz != SAMPLE_RATE_HZ or tick_ms != TICK_MS:
        raise ValueError(
            f"acoustic phase ABI requires {SAMPLE_RATE_HZ} Hz/{TICK_MS} ms, "
            f"got {sample_rate_hz} Hz/{tick_ms} ms"
        )
    history = _pcm_vector(
        user_history_pcm,
        name="user_history_pcm",
        maximum_samples=HISTORY_SAMPLES,
    )
    if not TICK_SAMPLES <= len(history) <= HISTORY_SAMPLES:
        raise ValueError(
            f"user_history_pcm must contain the current {TICK_SAMPLES}-sample tick "
            f"and at most {HISTORY_SAMPLES} samples"
        )
    if assistant_previous_pcm is None:
        assistant_previous = np.zeros(TICK_SAMPLES, dtype=np.float32)
    else:
        assistant_previous = _pcm_vector(assistant_previous_pcm, name="assistant_previous_pcm")
        if len(assistant_previous) != TICK_SAMPLES:
            raise ValueError(
                f"assistant_previous_pcm must contain exactly {TICK_SAMPLES} samples, "
                f"got {len(assistant_previous)}"
            )

    current_user = history[-TICK_SAMPLES:]
    previous_user = (
        history[-2 * TICK_SAMPLES : -TICK_SAMPLES]
        if len(history) >= 2 * TICK_SAMPLES
        else np.zeros(TICK_SAMPLES, dtype=np.float32)
    )
    padded_history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)
    padded_history[-len(history) :] = history

    current_log_rms = _log10_rms(current_user)
    previous_log_rms = _log10_rms(previous_user)
    delta = float(np.clip(current_log_rms - previous_log_rms, -6.0, 6.0) / 6.0)
    result = np.asarray(
        (
            current_log_rms,
            _log10_rms(padded_history),
            _log10_rms(assistant_previous),
            delta,
        ),
        dtype=np.float32,
    )
    if result.shape != (len(ACOUSTIC_PHASE_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("acoustic phase feature helper produced an invalid output")
    return result


__all__ = [
    "ACOUSTIC_PHASE_FEATURE_CONTRACT",
    "ACOUSTIC_PHASE_FEATURE_NAMES",
    "EPSILON",
    "HISTORY_MS",
    "HISTORY_SAMPLES",
    "SAMPLE_RATE_HZ",
    "TICK_MS",
    "TICK_SAMPLES",
    "acoustic_phase_features",
]
