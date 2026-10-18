"""Causal four-value phase input owned only by native Control6 LoRA."""
from __future__ import annotations

from typing import Any

import numpy as np

from duplex_kernel.acoustic_phase_features import EPSILON, TICK_SAMPLES


NATIVE_CONTROL6_PHASE_T_NAMES = (
    "assistant_log10_rms_prior_80ms",
    "assistant_speaking_prior_80ms",
    "assistant_log_rms_delta_clip6_div6",
    "user_log10_rms_current_80ms",
)
NATIVE_CONTROL6_PHASE_T_WIDTH = 4


def _frame(value: Any | None, *, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(TICK_SAMPLES, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (TICK_SAMPLES,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite 80 ms mono PCM frame")
    return result


def _log10_rms(pcm: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    return float(np.clip(np.log10(max(rms, EPSILON)), -6.0, 0.0))


def native_control6_phase_t(
    *,
    listener_frame_id: int,
    user_pcm_current: Any,
    assistant_pcm_prior: Any | None,
    assistant_prior_frame_id: int | None,
    assistant_pcm_preceding: Any | None,
    assistant_preceding_frame_id: int | None,
) -> np.ndarray:
    """Build ``phase_t`` from user tick ``t`` and assistant t-1/t-2 only."""
    if isinstance(listener_frame_id, bool) or not isinstance(listener_frame_id, int) or listener_frame_id < 0:
        raise ValueError("listener_frame_id must be a non-negative integer")
    user = _frame(user_pcm_current, name="user_pcm_current")
    prior = _frame(assistant_pcm_prior, name="assistant_pcm_prior")
    preceding = _frame(assistant_pcm_preceding, name="assistant_pcm_preceding")
    if assistant_pcm_prior is None:
        if assistant_prior_frame_id is not None:
            raise ValueError("missing assistant prior PCM cannot have a frame ID")
    elif (
        isinstance(assistant_prior_frame_id, bool)
        or not isinstance(assistant_prior_frame_id, int)
        or assistant_prior_frame_id != listener_frame_id - 1
    ):
        raise ValueError("assistant prior PCM must be exactly the t-1 visible frame")
    if assistant_pcm_preceding is None:
        if assistant_preceding_frame_id is not None:
            raise ValueError("missing preceding assistant PCM cannot have a frame ID")
    elif (
        isinstance(assistant_preceding_frame_id, bool)
        or not isinstance(assistant_preceding_frame_id, int)
        or assistant_preceding_frame_id != listener_frame_id - 2
    ):
        raise ValueError("preceding assistant PCM must be exactly the t-2 visible frame")

    prior_rms = float(np.sqrt(np.mean(np.square(prior, dtype=np.float64))))
    preceding_log_rms = _log10_rms(preceding)
    prior_log_rms = _log10_rms(prior)
    prior_speaking = float(prior_rms >= 0.001 and float(np.max(np.abs(prior))) >= 0.005)
    assistant_delta = float(np.clip(prior_log_rms - preceding_log_rms, -6.0, 6.0) / 6.0)
    result = np.asarray(
        (prior_log_rms, prior_speaking, assistant_delta, _log10_rms(user)),
        dtype=np.float32,
    )
    if result.shape != (NATIVE_CONTROL6_PHASE_T_WIDTH,) or not np.isfinite(result).all():
        raise RuntimeError("native Control6 phase_t must contain exactly four finite values")
    return result


__all__ = ["NATIVE_CONTROL6_PHASE_T_NAMES", "NATIVE_CONTROL6_PHASE_T_WIDTH", "native_control6_phase_t"]
