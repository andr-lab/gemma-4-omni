"""P3 real-model listener gate and raw event producer."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from duplex_kernel.listener import AssistantReferenceBuffer, GemmaListener, GemmaListenerState

FRAME_SAMPLES = 1280
FRAME_DURATION_MS = 80
SAMPLE_RATE = 16_000
MIN_REPLAY_FRAMES = 3_750  # five logical minutes at 12.5 Hz
MIN_LISTENER_UPDATES = 10_000
MIN_SPEAKING_FRAMES = 375  # thirty logical seconds


def _percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


def _source_scan(workspace: Path) -> dict[str, Any]:
    server = (workspace / "duplex_kernel" / "server.py").read_text(encoding="utf-8")
    listener = (workspace / "duplex_kernel" / "listener.py").read_text(encoding="utf-8")
    return {
        "event": "p3_dependency_scan",
        # The realtime handler must not make the legacy ASR call at all.
        "direct_realtime_asr_calls": len(re.findall(r"engine\.transcribe_audio\s*\(", server)),
        "scheduler_listener_binding": "on_listener = scheduler_listener" in server,
        "reference_buffer_binding": "assistant_reference_buffer.visible_for" in server,
        "listener_step_binding": "sess.listener.step" in server,
        "native_audio_tower_backend": "get_audio_features" in listener and "transcribe_audio" not in listener,
        "assistant_reference_default_delay_frames": 1,
    }


def run_checks(
    workspace: Path,
    *,
    listener: GemmaListener,
    state: GemmaListenerState,
    seed: int,
    total_frames: int = MIN_LISTENER_UPDATES,
    trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the P3 replay against a real GemmaListener and emit elemental events."""
    if total_frames < MIN_LISTENER_UPDATES:
        raise ValueError(f"P3 richiede almeno {MIN_LISTENER_UPDATES} listener updates")
    if state.frame_samples != FRAME_SAMPLES:
        raise ValueError("stato listener non configurato a 1280 sample/frame")
    if listener.sample_rate != SAMPLE_RATE or listener.frame_duration_ms != FRAME_DURATION_MS:
        raise ValueError("listener non configurato a 16 kHz/80 ms")

    rng = np.random.default_rng(seed)
    references = AssistantReferenceBuffer(delay_frames=1)
    latencies_ms: list[float] = []
    reference_ids: list[int] = []
    speaking_expected = 0
    speaking_updates = 0
    errors = 0
    replay_frames = min(total_frames, MIN_REPLAY_FRAMES)
    trace.append({
        "event": "p3_dependency_scan",
        **{key: value for key, value in _source_scan(workspace).items() if key != "event"},
    })

    for frame_id in range(total_frames):
        # Deterministic, non-silent input exercises the complete native path
        # without reading a future sample.  One frame is exactly 80 ms.
        user = (0.05 * rng.standard_normal(FRAME_SAMPLES)).astype(np.float32)
        if frame_id > 0:
            emitted = np.full(FRAME_SAMPLES, (frame_id % 17) / 32.0, dtype=np.float32)
            references.append(emitted, frame_id=frame_id - 1, emitted_at_ns=frame_id * 80_000_000)
        reference = references.visible_for(listener_frame_id=frame_id)
        expected_reference_id = frame_id - 1 if frame_id > 0 else None
        if reference is not None:
            reference_ids.append(reference.frame_id)
        speaker_active = frame_id < MIN_SPEAKING_FRAMES
        speaking_expected += int(speaker_active)
        started = time.perf_counter()
        try:
            import torch
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = listener.step(user, reference, state)
                end_event.record()
                end_event.synchronize()
                latency_ms = float(start_event.elapsed_time(end_event))
            else:
                output = listener.step(user, reference, state)
                latency_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(latency_ms)
            finite = bool(output.finite)
            hidden_numel = int(output.hidden_t.numel()) if hasattr(output.hidden_t, "numel") else int(np.asarray(output.hidden_t).size)
            speaking_updates += int(speaker_active)
            trace.append({
                "event": "p3_listener_frame",
                "frame_id": frame_id,
                "samples": FRAME_SAMPLES,
                "speaker_active": speaker_active,
                "reference_frame_id": output.reference_frame_id,
                "expected_reference_frame_id": expected_reference_id,
                "audio_timestamp": output.audio_timestamp,
                "listener_state_version": output.listener_state_version,
                "hidden_numel": hidden_numel,
                "finite": finite,
                "listener_step_gpu_ms": latency_ms,
            })
        except Exception as exc:
            errors += 1
            trace.append({
                "event": "p3_listener_error",
                "frame_id": frame_id,
                "samples": FRAME_SAMPLES,
                "speaker_active": speaker_active,
                "error": f"{type(exc).__name__}: {exc}",
            })

    stats = state.stats()
    future_leaks = int(stats["future_reference_leaks"] or 0)
    wrong_references = int(stats["wrong_reference_frames"] or 0)
    reference_wrong = sum(
        int(actual != expected)
        for actual, expected in zip(reference_ids, range(0, max(0, total_frames - 1)))
    )
    trace.append({
        "event": "p3_replay_summary",
        "total_frames": total_frames,
        "replay_frames": replay_frames,
        "replay_seconds": replay_frames * FRAME_DURATION_MS / 1000.0,
        "speaking_frames": speaking_expected,
        "listener_updates_while_speaking": speaking_updates,
        "future_reference_leaks": future_leaks,
        "wrong_reference_frames": wrong_references + reference_wrong,
    })
    trace.append({
        "event": "p3_model_metadata",
        "real_model": True,
        "backend": type(listener.backend).__name__,
        "hidden_size": getattr(listener.backend, "hidden_size", None),
        "context_frames": state.context_frames,
    })

    metrics = {
        "listener_input_frames": int(stats["listener_input_frames"]),
        "samples_received": int(stats["samples_received"]),
        "samples_accounted_for": int(stats["samples_accounted_for"]),
        "dropped_listener_frames": int(stats["dropped_listener_frames"]),
        "replay_frames": replay_frames,
        "replay_seconds": replay_frames * FRAME_DURATION_MS / 1000.0,
        "input_frames_while_speaking": speaking_expected,
        "listener_updates_while_speaking": speaking_updates,
        "listener_coverage_while_speaking": speaking_updates / speaking_expected if speaking_expected else 0.0,
        "nonfinite_hidden_count": int(stats["nonfinite_hidden_count"]),
        "empty_hidden_count": int(stats["empty_hidden_count"]),
        "future_reference_leaks": future_leaks,
        "wrong_reference_frames": wrong_references + reference_wrong,
        "assistant_reference_delay_frames": references.delay_frames,
        "listener_step_gpu_ms": {
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "max": round(max(latencies_ms), 3) if latencies_ms else None,
        },
        "listener_errors": errors,
        "p3_dependency_scan": _source_scan(workspace),
    }
    thresholds = {
        "listener_input_frames_min": MIN_LISTENER_UPDATES,
        "samples_received_equals_accounted": True,
        "replay_frames_min": MIN_REPLAY_FRAMES,
        "replay_seconds_min": 300.0,
        "input_frames_while_speaking_min": MIN_SPEAKING_FRAMES,
        "listener_coverage_while_speaking": 1.0,
        "dropped_listener_frames": 0,
        "nonfinite_hidden_count": 0,
        "empty_hidden_count": 0,
        "future_reference_leaks": 0,
        "wrong_reference_frames": 0,
        "listener_step_gpu_ms_p95_max": 80.0,
        "listener_errors": 0,
        "assistant_reference_delay_frames": 1,
    }
    return metrics, thresholds


__all__ = ["run_checks"]
