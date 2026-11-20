from __future__ import annotations

import math
import sys
import threading
from typing import Any

_TALKER_CUDA_TIMING_PROBE_LOCK = threading.Lock()
_TALKER_CUDA_TIMING_PROBE_CLAIMED = False


def _claim_talker_cuda_timing_probe() -> bool:
    state_owner = sys.modules.get("duplex_kernel.engine", sys.modules[__name__])
    lock = getattr(state_owner, "_TALKER_CUDA_TIMING_PROBE_LOCK", _TALKER_CUDA_TIMING_PROBE_LOCK)
    with lock:
        if getattr(state_owner, "_TALKER_CUDA_TIMING_PROBE_CLAIMED", _TALKER_CUDA_TIMING_PROBE_CLAIMED):
            return False
        state_owner._TALKER_CUDA_TIMING_PROBE_CLAIMED = True
        return True


def _summarize_timing_values_ms(values: list[float], *, pending_count: int = 0) -> dict[str, Any]:
    """Summarize completed GPU event timings without synchronizing the device."""
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))

    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)

    return {
        "sample_count": len(ordered),
        "pending_count": int(pending_count),
        "p50_ms": quantile(0.50),
        "p95_ms": quantile(0.95),
        "max_ms": round(max(ordered), 3) if ordered else None,
    }


def _completed_cuda_event_elapsed_ms(start_event: Any, end_event: Any) -> float | None:
    """Return a GPU duration only when both events completed; never synchronize."""
    try:
        if not start_event.query() or not end_event.query():
            return None
        elapsed_ms = float(start_event.elapsed_time(end_event))
        if math.isfinite(elapsed_ms) and elapsed_ms >= 0.0:
            return elapsed_ms
    except Exception:
        # Optional telemetry must never fail an otherwise valid audio response.
        return None
    return None


def _record_completed_cuda_timing(
    metrics: dict[str, Any], key: str, start_event: Any, end_event: Any
) -> None:
    elapsed_ms = _completed_cuda_event_elapsed_ms(start_event, end_event)
    if elapsed_ms is None:
        metrics[f"{key}_pending"] = True
        return
    metrics[key] = elapsed_ms
    metrics[f"{key}_pending"] = False


def _summarize_cuda_event_pairs(event_pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Read only completed event pairs; query/elapsed_time never wait for CUDA."""
    values: list[float] = []
    pending = 0
    for start_event, end_event in event_pairs:
        elapsed_ms = _completed_cuda_event_elapsed_ms(start_event, end_event)
        if elapsed_ms is None:
            pending += 1
        else:
            values.append(elapsed_ms)
    return _summarize_timing_values_ms(values, pending_count=pending)
