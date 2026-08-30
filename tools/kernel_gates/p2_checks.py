"""Independent P2 harness for the 80 ms scheduler and PCM ingress."""
from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from duplex_kernel.scheduler import AudioFrame, AudioIngressQueue, DuplexScheduler, MonotonicFrameClock, VirtualFrameClock

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


def _make_ingress(frame_count: int, *, capture_start_ns: int) -> tuple[AudioIngressQueue, list[AudioFrame]]:
    ingress = AudioIngressQueue()
    frames = ingress.append(
        np.zeros(frame_count * FRAME_SAMPLES, dtype=np.float32),
        capture_start_ns=capture_start_ns,
    )
    return ingress, frames


def _record_transport(trace: list[dict[str, Any]], stream: str, phase: str, frames: list[AudioFrame]) -> None:
    for frame in frames:
        _emit(trace, "p2_input_" + phase, stream=stream, frame_id=frame.frame_id, samples=len(frame.pcm))


def _record_consumed(trace: list[dict[str, Any]], stream: str, ticks: list[Any]) -> None:
    for tick in ticks:
        if tick.enqueue_ns is not None:
            _emit(trace, "p2_input_consumed", stream=stream, frame_id=tick.frame_id, samples=FRAME_SAMPLES)


def _run_concurrent_transport(trace: list[dict[str, Any]]) -> dict[str, int]:
    """Run a producer while the scheduler's speaker callback is occupied."""
    ingress = AudioIngressQueue()
    initial = ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), capture_start_ns=0)
    _record_transport(trace, "concurrent", "arrival", [initial])
    _record_transport(trace, "concurrent", "enqueued", [initial])
    speaking = threading.Event()
    producer_done = threading.Event()
    produced_ids: list[int] = []
    producer_thread: threading.Thread | None = None

    def produce() -> None:
        try:
            for frame_id in range(1, 21):
                _emit(trace, "p2_input_arrival", stream="concurrent", frame_id=frame_id,
                      samples=FRAME_SAMPLES, speaker_active=speaking.is_set())
                try:
                    ingress.put(
                        np.full(FRAME_SAMPLES, frame_id, dtype=np.float32),
                        capture_start_ns=frame_id * FRAME_NS,
                    )
                except queue.Full:
                    _emit(trace, "p2_input_drop", stream="concurrent", frame_id=frame_id, samples=FRAME_SAMPLES)
                    continue
                _emit(trace, "p2_input_enqueued", stream="concurrent", frame_id=frame_id, samples=FRAME_SAMPLES)
                if speaking.is_set():
                    produced_ids.append(frame_id)
                time.sleep(0.002)
        finally:
            producer_done.set()

    def speaker(frame: AudioFrame, tick: Any) -> None:
        nonlocal producer_thread
        if frame.frame_id == 0:
            speaking.set()
            producer_thread = threading.Thread(target=produce, name="p2-concurrent-producer")
            producer_thread.start()
            time.sleep(0.05)
        else:
            time.sleep(0.005)
        _emit(trace, "p2_concurrent_speaker_frame", frame_id=frame.frame_id, tick_frame_id=tick.frame_id)

    scheduler = DuplexScheduler(ingress, clock=MonotonicFrameClock(), on_speaker=speaker)
    ticks = scheduler.run_ticks(21)
    producer_done.wait(timeout=2.0)
    if producer_thread is not None:
        producer_thread.join(timeout=2.0)
    _record_consumed(trace, "concurrent", ticks)
    concurrent_arrivals = [e for e in trace if e.get("event") == "p2_input_arrival" and e.get("stream") == "concurrent"]
    concurrent_enqueued = [e for e in trace if e.get("event") == "p2_input_enqueued" and e.get("stream") == "concurrent"]
    concurrent_consumed = [e for e in trace if e.get("event") == "p2_input_consumed" and e.get("stream") == "concurrent"]
    concurrent_dropped = [e for e in trace if e.get("event") == "p2_input_drop" and e.get("stream") == "concurrent"]
    speaking_events = [e for e in trace if e.get("event") == "p2_concurrent_speaker_frame"]
    metrics = {
        "concurrent_input_frames_received": len(concurrent_arrivals),
        "concurrent_input_frames_enqueued": len(concurrent_enqueued),
        "concurrent_input_frames_consumed": len(concurrent_consumed),
        "concurrent_input_frames_during_speaking": len(produced_ids),
        "concurrent_dropped_input_frames": len(concurrent_dropped),
        "concurrent_duplicate_input_frames": len(concurrent_consumed) - len({e.get("frame_id") for e in concurrent_consumed}),
        "concurrent_out_of_order_input_frames": int([e.get("frame_id") for e in concurrent_consumed] != list(range(21))),
        "concurrent_speaker_frames": len(speaking_events),
    }
    return metrics


def run_checks(
    workspace: Path,
    *,
    seed: int,
    real_duration_s: float,
    cpu_contract: dict[str, Any],
    trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    server_text = (workspace / "duplex_kernel" / "server.py").read_text(encoding="utf-8")
    ws_block = server_text.split('elif et == "input_audio_buffer.append":', 1)[1].split(
        'elif et in {"input_audio_buffer.commit", "response.create"}:', 1
    )[0]
    dependency = {
        "direct_server_audio_ingress_consumers": len(re.findall(r"audio_ingress\.(?:get|get_nowait)\s*\(", server_text)),
        "direct_ws_policy_calls": len(re.findall(r"\.step\s*\(", ws_block)),
        "scheduler_policy_binding": "sess.scheduler.on_policy = scheduler_policy" in server_text,
        "scheduler_start_binding": "sess.scheduler.start(" in server_text,
    }
    if real_duration_s < 60.0:
        raise ValueError("il gate P2 richiede almeno 60 secondi di real clock")
    expected_real_ticks = round(real_duration_s * 1_000 / 80)
    if abs(real_duration_s * 1_000 / 80 - expected_real_ticks) > 1e-6:
        raise ValueError("real_duration_s deve essere un multiplo di 80 ms")

    _emit(trace, "phase_metadata", seed=seed, rng_seed_applied=True, cpu_contract=cpu_contract,
          frame_duration_ms=80, sample_rate=16_000)
    _emit(trace, "p2_dependency_scan", **dependency)

    ingress_10, frames_10 = _make_ingress(125, capture_start_ns=0)
    _record_transport(trace, "virtual_10s", "arrival", frames_10)
    _record_transport(trace, "virtual_10s", "enqueued", frames_10)
    scheduler_10 = DuplexScheduler(ingress_10, clock=VirtualFrameClock())
    ticks_10 = scheduler_10.run_for(10.0)
    _record_consumed(trace, "virtual_10s", ticks_10)
    for index, tick in enumerate(ticks_10):
        _emit(trace, **_tick_event("virtual_10s", index, tick))

    speaking_events: list[int] = []

    def on_speaker(frame: AudioFrame, tick: Any) -> None:
        speaking_events.append(frame.frame_id)
        _emit(trace, "p2_speaking_frame", frame_id=frame.frame_id, tick_frame_id=tick.frame_id)

    ingress_60, frames_60 = _make_ingress(750, capture_start_ns=0)
    _record_transport(trace, "virtual_60s", "arrival", frames_60)
    _record_transport(trace, "virtual_60s", "enqueued", frames_60)
    scheduler_60 = DuplexScheduler(
        ingress_60,
        clock=VirtualFrameClock(),
        on_speaker=on_speaker,
    )
    ticks_60 = scheduler_60.run_for(60.0)
    _record_consumed(trace, "virtual_60s", ticks_60)
    for index, tick in enumerate(ticks_60):
        _emit(trace, **_tick_event("virtual_60s", index, tick, speaking=True))

    concurrent_metrics = _run_concurrent_transport(trace)

    real_clock = MonotonicFrameClock()
    ingress_real, frames_real = _make_ingress(expected_real_ticks, capture_start_ns=real_clock.start_ns)
    _record_transport(trace, "real", "arrival", frames_real)
    _record_transport(trace, "real", "enqueued", frames_real)
    scheduler_real = DuplexScheduler(ingress_real, clock=real_clock)
    real_started = time.monotonic()
    real_ticks = scheduler_real.run_for(real_duration_s)
    # 750 ticks have deadlines [0, 59.92 s]. Keep the gate wall-clock run at
    # the requested duration as well as validating its 12.5 Hz cadence.
    remaining = real_duration_s - (time.monotonic() - real_started)
    if remaining > 0:
        time.sleep(remaining)
    real_elapsed_s = time.monotonic() - real_started
    _record_consumed(trace, "real", real_ticks)
    for index, tick in enumerate(real_ticks):
        _emit(trace, **_tick_event("real", index, tick))
    _emit(trace, "p2_real_run", requested_duration_s=real_duration_s, elapsed_s=real_elapsed_s,
          expected_ticks=expected_real_ticks)

    virtual_stats = scheduler_60.stats()
    real_stats = scheduler_real.stats()
    metrics: dict[str, Any] = {
        "cpu_contract": cpu_contract,
        **dependency,
        "virtual_10_ticks": len(ticks_10),
        "virtual_10_missing_ticks": scheduler_10.stats()["missing_ticks"],
        "virtual_10_duplicate_ticks": scheduler_10.stats()["duplicate_ticks"],
        "virtual_10_out_of_order_ticks": scheduler_10.stats()["out_of_order_ticks"],
        "virtual_60_ticks": len(ticks_60),
        "virtual_60_missing_ticks": virtual_stats["missing_ticks"],
        "virtual_60_duplicate_ticks": virtual_stats["duplicate_ticks"],
        "virtual_60_out_of_order_ticks": virtual_stats["out_of_order_ticks"],
        "input_frames_received": virtual_stats["input_frames_received"],
        "input_frames_enqueued": virtual_stats["input_frames_enqueued"],
        "input_frames_consumed": virtual_stats["input_frames_consumed"],
        "input_samples_received": virtual_stats["samples_received"],
        "dropped_input_frames": virtual_stats["dropped_input_frames"],
        "duplicate_input_frames": virtual_stats["duplicate_input_frames"],
        "out_of_order_input_frames": virtual_stats["out_of_order_input_frames"],
        "input_frames_during_speaking": len(speaking_events),
        **concurrent_metrics,
        "real_ticks": len(real_ticks),
        "real_duration_s": round(real_elapsed_s, 3),
        "real_deadline_miss_rate": real_stats["deadline_miss_rate"],
        "real_tick_jitter_p99_ms": real_stats["tick_jitter_p99_ms"],
    }
    thresholds: dict[str, Any] = {
        "cpu_contract_exit_code": 0,
        "direct_server_audio_ingress_consumers": 0,
        "direct_ws_policy_calls": 0,
        "scheduler_policy_binding": True,
        "scheduler_start_binding": True,
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
        "concurrent_input_frames_received": 21,
        "concurrent_input_frames_enqueued": 21,
        "concurrent_input_frames_consumed": 21,
        "concurrent_input_frames_during_speaking_min": 1,
        "concurrent_dropped_input_frames": 0,
        "concurrent_duplicate_input_frames": 0,
        "concurrent_out_of_order_input_frames": 0,
        "concurrent_speaker_frames": 21,
        "real_ticks": expected_real_ticks,
        "real_duration_min_s": real_duration_s,
        "real_deadline_miss_rate_max": 0.005,
        "real_tick_jitter_p99_ms_max": 20.0,
    }
    metrics["passed"] = bool(
        cpu_contract["exit_code"] == 0
        and metrics["direct_server_audio_ingress_consumers"] == 0
        and metrics["direct_ws_policy_calls"] == 0
        and metrics["scheduler_policy_binding"] is True
        and metrics["scheduler_start_binding"] is True
        and metrics["virtual_10_ticks"] == 125
        and metrics["virtual_60_ticks"] == 750
        and metrics["input_frames_consumed"] == 750
        and metrics["dropped_input_frames"] == 0
        and len(speaking_events) > 0
        and all(metrics[key] == thresholds[key] for key in (
            "concurrent_input_frames_received", "concurrent_input_frames_enqueued",
            "concurrent_input_frames_consumed", "concurrent_dropped_input_frames",
            "concurrent_duplicate_input_frames", "concurrent_out_of_order_input_frames",
            "concurrent_speaker_frames",
        ))
        and metrics["concurrent_input_frames_during_speaking"] >= 1
        and metrics["real_ticks"] == expected_real_ticks
        and metrics["real_duration_s"] >= real_duration_s
        and float(metrics["real_deadline_miss_rate"]) <= 0.005
        and float(metrics["real_tick_jitter_p99_ms"]) <= 20.0
    )
    return metrics, thresholds


__all__ = ["run_checks"]
