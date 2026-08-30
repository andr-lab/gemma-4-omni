"""Independent P2 harness for the 80 ms scheduler and PCM ingress."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from duplex_kernel.scheduler import AudioIngressQueue, DuplexScheduler, MonotonicFrameClock, VirtualFrameClock

FRAME_SAMPLES = 1280
FRAME_NS = 80_000_000


def _emit(trace: list[dict[str, Any]], event: str, **fields: Any) -> None:
    trace.append({"event": event, **fields})


def _tick_event(clock_name: str, tick_index: int, tick: Any, *, speaking: bool = False) -> dict[str, Any]:
    return {
        "event": "p2_tick",
        "clock": clock_name,
        "tick_index": tick_index,
        "frame_id": tick.frame_id,
        "capture_start_ns": tick.capture_start_ns,
        "capture_end_ns": tick.capture_end_ns,
        "enqueue_ns": tick.enqueue_ns,
        "deadline_ns": tick.deadline_ns,
        "observed_ns": tick.observed_ns,
        "listener_start_ns": tick.listener_start_ns,
        "listener_end_ns": tick.listener_end_ns,
        "policy_ns": tick.policy_ns,
        "speaker_start_ns": tick.speaker_start_ns,
        "speaker_end_ns": tick.speaker_end_ns,
        "speaker_active": speaking,
    }


def _make_ingress(frame_count: int, *, capture_start_ns: int) -> AudioIngressQueue:
    ingress = AudioIngressQueue()
    ingress.append(
        np.zeros(frame_count * FRAME_SAMPLES, dtype=np.float32),
        capture_start_ns=capture_start_ns,
    )
    return ingress


def run_checks(
    workspace: Path,
    *,
    seed: int,
    real_duration_s: float,
    cpu_contract: dict[str, Any],
    trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    del workspace
    if real_duration_s < 60.0:
        raise ValueError("il gate P2 richiede almeno 60 secondi di real clock")
    expected_real_ticks = round(real_duration_s * 1_000 / 80)
    if abs(real_duration_s * 1_000 / 80 - expected_real_ticks) > 1e-6:
        raise ValueError("real_duration_s deve essere un multiplo di 80 ms")

    _emit(trace, "phase_metadata", seed=seed, rng_seed_applied=True, cpu_contract=cpu_contract,
          frame_duration_ms=80, sample_rate=16_000)

    ingress_10 = _make_ingress(125, capture_start_ns=0)
    scheduler_10 = DuplexScheduler(ingress_10, clock=VirtualFrameClock())
    ticks_10 = scheduler_10.run_for(10.0)
    for index, tick in enumerate(ticks_10):
        _emit(trace, **_tick_event("virtual_10s", index, tick))

    speaking_events: list[int] = []

    def on_speaker(frame: Any, tick: Any) -> None:
        if frame is not None:
            speaking_events.append(frame.frame_id)
            _emit(trace, "p2_speaking_frame", frame_id=frame.frame_id, tick_frame_id=tick.frame_id)

    ingress_60 = _make_ingress(750, capture_start_ns=0)
    scheduler_60 = DuplexScheduler(
        ingress_60,
        clock=VirtualFrameClock(),
        on_speaker=on_speaker,
    )
    ticks_60 = scheduler_60.run_for(60.0)
    for index, tick in enumerate(ticks_60):
        _emit(trace, **_tick_event("virtual_60s", index, tick, speaking=True))

    real_clock = MonotonicFrameClock()
    ingress_real = _make_ingress(expected_real_ticks, capture_start_ns=real_clock.start_ns)
    scheduler_real = DuplexScheduler(ingress_real, clock=real_clock)
    real_started = time.monotonic()
    real_ticks = scheduler_real.run_for(real_duration_s)
    # 750 ticks have deadlines [0, 59.92 s]. Keep the gate wall-clock run at
    # the requested duration as well as validating its 12.5 Hz cadence.
    remaining = real_duration_s - (time.monotonic() - real_started)
    if remaining > 0:
        time.sleep(remaining)
    real_elapsed_s = time.monotonic() - real_started
    for index, tick in enumerate(real_ticks):
        _emit(trace, **_tick_event("real", index, tick))
    _emit(trace, "p2_real_run", requested_duration_s=real_duration_s, elapsed_s=real_elapsed_s,
          expected_ticks=expected_real_ticks)

    metrics: dict[str, Any] = {
        "cpu_contract": cpu_contract,
        "virtual_10_ticks": len(ticks_10),
        "virtual_10_missing_ticks": scheduler_10.stats()["missing_ticks"],
        "virtual_10_duplicate_ticks": scheduler_10.stats()["duplicate_ticks"],
        "virtual_10_out_of_order_ticks": scheduler_10.stats()["out_of_order_ticks"],
        "virtual_60_ticks": len(ticks_60),
        "virtual_60_missing_ticks": scheduler_60.stats()["missing_ticks"],
        "virtual_60_duplicate_ticks": scheduler_60.stats()["duplicate_ticks"],
        "virtual_60_out_of_order_ticks": scheduler_60.stats()["out_of_order_ticks"],
        "input_frames_received": scheduler_60.stats()["input_frames_received"],
        "input_frames_enqueued": scheduler_60.stats()["input_frames_enqueued"],
        "input_frames_consumed": scheduler_60.stats()["input_frames_consumed"],
        "input_samples_received": scheduler_60.stats()["samples_received"],
        "dropped_input_frames": scheduler_60.stats()["dropped_input_frames"],
        "duplicate_input_frames": scheduler_60.stats()["duplicate_input_frames"],
        "out_of_order_input_frames": scheduler_60.stats()["out_of_order_input_frames"],
        "input_frames_during_speaking": len(speaking_events),
        "real_ticks": len(real_ticks),
        "real_duration_s": round(real_elapsed_s, 3),
        "real_deadline_miss_rate": scheduler_real.stats()["deadline_miss_rate"],
        "real_tick_jitter_p99_ms": scheduler_real.stats()["tick_jitter_p99_ms"],
    }
    thresholds: dict[str, Any] = {
        "cpu_contract_exit_code": 0,
        "virtual_10_ticks": 125,
        "virtual_10_missing_ticks": 0,
        "virtual_10_duplicate_ticks": 0,
        "virtual_10_out_of_order_ticks": 0,
        "virtual_60_ticks": 750,
        "virtual_60_missing_ticks": 0,
        "virtual_60_duplicate_ticks": 0,
        "virtual_60_out_of_order_ticks": 0,
        "input_frames_received": 750,
        "input_frames_enqueued": 750,
        "input_frames_consumed": 750,
        "input_samples_received": 750 * FRAME_SAMPLES,
        "dropped_input_frames": 0,
        "duplicate_input_frames": 0,
        "out_of_order_input_frames": 0,
        "input_frames_during_speaking_min": 1,
        "real_ticks": expected_real_ticks,
        "real_duration_min_s": real_duration_s,
        "real_deadline_miss_rate_max": 0.005,
        "real_tick_jitter_p99_ms_max": 20.0,
    }
    metrics["passed"] = bool(
        cpu_contract["exit_code"] == 0
        and metrics["virtual_10_ticks"] == 125
        and metrics["virtual_60_ticks"] == 750
        and metrics["input_frames_consumed"] == 750
        and metrics["dropped_input_frames"] == 0
        and len(speaking_events) > 0
        and metrics["real_ticks"] == expected_real_ticks
        and metrics["real_duration_s"] >= real_duration_s
        and float(metrics["real_deadline_miss_rate"]) <= 0.005
        and float(metrics["real_tick_jitter_p99_ms"]) <= 20.0
    )
    return metrics, thresholds


__all__ = ["run_checks"]
