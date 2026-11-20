"""Realtime dialogue harness with paced speech input and FIFO-consumption ACKs.

The tool never creates responses or injects transcripts/actions.  It sends
recorded speech through the normal automatic Control6 path and treats absent
playback, ACK, KV, VRAM, or server-log evidence as incomplete.

The fixture manifest contains four entries: ``long_request``, ``interruption``,
``memory_followup``, and ``closure``. Each entry supplies a relative WAV path,
an offline ASR JSON file with model/device/language metadata, and an explicit
``transcript_verified`` flag. The stop fixture must contain the ASR-verified
phrase ``Aspetta, fermati un momento`` in acceptance mode.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import ssl
import sys
import time
import wave
from typing import Any, Mapping, MutableMapping, Sequence
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
from scipy.signal import resample_poly
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from client.openai_duplex_client import (  # noqa: E402
    BoundedPCMPlayback,
    CODEC_FRAME_SAMPLES,
    CODEC_SAMPLE_RATE,
    MIC_FRAME_NS,
    MIC_FRAME_SAMPLES,
    MIC_SAMPLE_RATE,
    decode_response_audio,
)


TURN_POLICY = "gemma_native_control6"
FRAME_MS = 80
FRAME_DURATION_NS = MIC_FRAME_NS
NOISE_RMS_MIN = 0.0015
NOISE_RMS_MAX = 0.0025
INPUT_RMS_MIN = 0.015
MAX_INPUT_INFLIGHT = 16
DEFAULT_PACING_TOLERANCE_MS = 60.0
DEFAULT_MAX_GAP_MS = 160.0
LONG_RESPONSE_MIN_SAMPLES = 40 * CODEC_SAMPLE_RATE
LONG_RESPONSE_MIN_VOCAL_FRAMES = 151
PLAYOUT_BLOCK_SAMPLES = CODEC_SAMPLE_RATE // 100  # 10 ms device callback.
# Mirrors MAX_PLAYBACK_AHEAD_SECONDS in server/static/index.html:enqueueAudio.
BROWSER_PLAYBACK_MAX_SECONDS = 4.0
BROWSER_PLAYBACK_MAX_SAMPLES = int(BROWSER_PLAYBACK_MAX_SECONDS * CODEC_SAMPLE_RATE)


class EvidenceError(RuntimeError):
    """A required input/protocol/evidence contract failed closed."""


def _long_response_input_window_metrics(
    *,
    pcm_samples: int,
    first_audio_ns: int,
    done_ns: int,
    playout_started_ns: int,
    playout_completed_ns: int,
    final_observation_ns: int,
    input_frame_sends: list[Mapping[str, Any]],
    input_frame_acks: list[Mapping[str, Any]],
    fifo_peak_samples: int,
    input_ack_cutoff_ns: int | None = None,
    minimum_frames: int = 500,
    minimum_ack_ratio: float = 0.99,
    minimum_playout_seconds: float = 40.0,
    max_input_gap_ms: float = DEFAULT_MAX_GAP_MS,
    metrics_sink: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure PCM, production, and playout independently; verify mic continuity during playout."""
    ack_cutoff_ns = (
        int(input_ack_cutoff_ns)
        if input_ack_cutoff_ns is not None
        else int(final_observation_ns)
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in (
            pcm_samples, first_audio_ns, done_ns, playout_started_ns,
            playout_completed_ns, final_observation_ns,
        ))
        or pcm_samples < 0
        or done_ns < first_audio_ns
        or playout_completed_ns < playout_started_ns
        or ack_cutoff_ns < playout_completed_ns
    ):
        raise EvidenceError("long response has invalid monotonic timestamps or PCM sample count")
    if isinstance(fifo_peak_samples, bool) or not isinstance(fifo_peak_samples, int) or fifo_peak_samples < 0:
        raise EvidenceError("long response FIFO peak must be a non-negative sample count")
    pcm_duration_seconds = int(pcm_samples) / CODEC_SAMPLE_RATE
    production_window_seconds = (done_ns - first_audio_ns) / 1_000_000_000
    playout_window_seconds = (playout_completed_ns - playout_started_ns) / 1_000_000_000
    if playout_window_seconds < float(minimum_playout_seconds):
        raise EvidenceError(
            f"long response playout lasted {playout_window_seconds:.3f}s; "
            f"need at least {minimum_playout_seconds:.1f}s"
        )
    if pcm_duration_seconds < float(minimum_playout_seconds):
        raise EvidenceError(
            f"long response contains {pcm_duration_seconds:.3f}s PCM; "
            f"need at least {minimum_playout_seconds:.1f}s"
        )
    continuity_end_ns = int(playout_completed_ns)

    window_sends: dict[int, int] = {}
    window_backpressure_waits: dict[int, int] = {}
    for item in input_frame_sends:
        frame_id = item.get("local_frame_id")
        sent_ns = item.get("sent_ns")
        if (
            isinstance(frame_id, bool) or not isinstance(frame_id, int)
            or isinstance(sent_ns, bool) or not isinstance(sent_ns, int)
        ):
            continue
        if playout_started_ns <= sent_ns <= continuity_end_ns:
            if frame_id in window_sends:
                raise EvidenceError("long response input frame sends contain duplicate frame IDs")
            window_sends[frame_id] = sent_ns
            wait_ns = item.get("backpressure_wait_ns", 0)
            window_backpressure_waits[frame_id] = (
                wait_ns if isinstance(wait_ns, int) and not isinstance(wait_ns, bool) and wait_ns > 0 else 0
            )

    acked_ids: set[int] = set()
    for item in input_frame_acks:
        frame_id = item.get("local_frame_id")
        received_ns = item.get("received_ns")
        if (
            isinstance(frame_id, bool) or not isinstance(frame_id, int)
            or isinstance(received_ns, bool) or not isinstance(received_ns, int)
        ):
            continue
        # ACK delivery can trail the playout boundary. Count it through the
        # cutoff, but only for frames whose send overlapped playback.
        if playout_started_ns <= received_ns <= ack_cutoff_ns and frame_id in window_sends:
            acked_ids.add(frame_id)

    sent_count = len(window_sends)
    acked_count = len(acked_ids)
    ack_ratio = acked_count / sent_count if sent_count else 0.0
    ack_rows = sorted(
        (int(item["received_ns"]), int(item["local_frame_id"]))
        for item in input_frame_acks
        if isinstance(item.get("local_frame_id"), int)
        and not isinstance(item.get("local_frame_id"), bool)
        and int(item["local_frame_id"]) in acked_ids
        and isinstance(item.get("received_ns"), int)
        and not isinstance(item.get("received_ns"), bool)
    )
    send_rows = sorted((sent_ns, frame_id) for frame_id, sent_ns in window_sends.items())
    send_gaps = [
        (
            right_ns - left_ns,
            max(0, right_ns - left_ns - window_backpressure_waits.get(right_frame_id, 0)),
        )
        for (left_ns, _left_frame_id), (right_ns, right_frame_id)
        in zip(send_rows, send_rows[1:])
    ]
    max_send_gap_ms = max((raw_gap / 1e6 for raw_gap, _net_gap in send_gaps), default=0.0)
    max_send_net_gap_ms = max((net_gap / 1e6 for _raw_gap, net_gap in send_gaps), default=0.0)
    ack_gaps = [
        (right[0] - left[0], left, right)
        for left, right in zip(ack_rows, ack_rows[1:])
    ]
    max_ack_gap_ns, max_ack_left, max_ack_right = max(
        ack_gaps,
        key=lambda item: item[0],
        default=(0, None, None),
    )
    max_ack_gap_ms = max_ack_gap_ns / 1e6
    max_ack_gap_pair = None if max_ack_left is None else {
        "gap_ms": max_ack_gap_ms,
        "left": {
            "local_frame_id": int(max_ack_left[1]),
            "received_ns": int(max_ack_left[0]),
        },
        "right": {
            "local_frame_id": int(max_ack_right[1]),
            "received_ns": int(max_ack_right[0]),
        },
    }
    metrics: dict[str, Any] = {
        "pcm_samples": int(pcm_samples),
        "pcm_duration_seconds": pcm_duration_seconds,
        "first_audio_ns": int(first_audio_ns),
        "response_done_ns": int(done_ns),
        "production_window_seconds": production_window_seconds,
        "playout_started_ns": int(playout_started_ns),
        "playout_completed_ns": int(playout_completed_ns),
        "playout_window_seconds": playout_window_seconds,
        "final_observation_ns": int(final_observation_ns),
        "input_ack_cutoff_ns": int(ack_cutoff_ns),
        "microphone_continuity_start_ns": int(playout_started_ns),
        "microphone_continuity_end_ns": continuity_end_ns,
        "microphone_continuity_seconds": (continuity_end_ns - playout_started_ns) / 1e9,
        "input_frames_sent": sent_count,
        "input_frames_acked": acked_count,
        "ack_ratio": ack_ratio,
        "max_input_send_gap_ms": max_send_gap_ms,
        "max_input_send_net_gap_ms": max_send_net_gap_ms,
        "max_input_ack_gap_ms": max_ack_gap_ms,
        "max_input_ack_gap_pair": max_ack_gap_pair,
        "minimum_input_frames": int(minimum_frames),
        "minimum_ack_ratio": float(minimum_ack_ratio),
        "minimum_playout_seconds": float(minimum_playout_seconds),
        "max_input_gap_ms": float(max_input_gap_ms),
        "fifo_peak_samples": int(fifo_peak_samples),
        "fifo_capacity_samples": BROWSER_PLAYBACK_MAX_SAMPLES,
        "validation_status": "PENDING",
    }

    def save_metrics() -> None:
        if metrics_sink is not None:
            metrics_sink.clear()
            metrics_sink.update(metrics)

    def fail(message: str) -> None:
        metrics["validation_status"] = "FAIL"
        metrics["validation_error"] = message
        save_metrics()
        raise EvidenceError(message)

    save_metrics()
    if sent_count < int(minimum_frames):
        fail(
            f"long response carried only {sent_count} input frames; need at least {minimum_frames}"
        )
    if ack_ratio < float(minimum_ack_ratio):
        fail(
            f"long response input ACK ratio {ack_ratio:.3%} is below {minimum_ack_ratio:.1%} "
            f"({acked_count}/{sent_count})"
        )
    if max_send_net_gap_ms >= float(max_input_gap_ms):
        offending = max(send_gaps, key=lambda pair: pair[1], default=(0, 0))
        fail(
            f"long response microphone send gap after backpressure accounting "
            f"{offending[1] / 1e6:.1f}ms (raw {offending[0] / 1e6:.1f}ms) reached "
            f"the {float(max_input_gap_ms):.1f}ms limit"
        )
    if max_ack_gap_ms >= float(max_input_gap_ms):
        fail(
            f"long response microphone ACK gap {max_ack_gap_ms:.1f}ms reached "
            f"the {float(max_input_gap_ms):.1f}ms limit"
        )
    if fifo_peak_samples > BROWSER_PLAYBACK_MAX_SAMPLES:
        fail(
            f"long response FIFO peak {fifo_peak_samples} samples exceeds "
            f"{BROWSER_PLAYBACK_MAX_SAMPLES}"
        )
    metrics["validation_status"] = "PASS"
    save_metrics()
    return metrics


def _device_samples_due_since_epoch(
    now_ns: int,
    epoch_ns: int,
    device_samples_already_elapsed: int,
    *,
    sample_rate: int = CODEC_SAMPLE_RATE,
) -> int:
    """Return wall-clock device samples due without resetting fractional time."""
    if sample_rate <= 0 or device_samples_already_elapsed < 0:
        raise ValueError("invalid device clock state")
    target = max(0, (int(now_ns) - int(epoch_ns)) * int(sample_rate) // 1_000_000_000)
    return max(0, target - int(device_samples_already_elapsed))


def _playback_capacity_samples(mode: str, requested_seconds: float) -> int:
    seconds = float(requested_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise EvidenceError("playback FIFO duration must be finite and positive")
    if seconds > BROWSER_PLAYBACK_MAX_SECONDS:
        raise EvidenceError(
            f"playback FIFO cannot exceed the browser's {BROWSER_PLAYBACK_MAX_SECONDS:.1f}s scheduling horizon"
        )
    if mode == "acceptance" and not math.isclose(
        seconds, BROWSER_PLAYBACK_MAX_SECONDS, rel_tol=0.0, abs_tol=1e-9
    ):
        raise EvidenceError(
            f"acceptance mode must match the browser's {BROWSER_PLAYBACK_MAX_SECONDS:.1f}s FIFO limit"
        )
    return int(seconds * CODEC_SAMPLE_RATE)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rms(values: np.ndarray) -> float:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        return math.nan
    return float(np.sqrt(np.mean(np.square(samples))))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_manifest_asr(asr_path: Path, source_name: str) -> tuple[str, dict[str, Any]]:
    if not asr_path.is_file() or asr_path.stat().st_size <= 0:
        raise EvidenceError(f"ASR evidence is missing or empty: {asr_path}")
    try:
        payload = json.loads(asr_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"ASR evidence is not valid JSON: {asr_path}") from exc
    candidates: list[Mapping[str, Any]] = []
    model = str(payload.get("model", "")).strip() if isinstance(payload, Mapping) else ""
    device = str(payload.get("device", "")).strip() if isinstance(payload, Mapping) else ""
    compute_type = str(payload.get("compute_type", "")).strip() if isinstance(payload, Mapping) else ""
    transcript = ""
    language = ""
    if isinstance(payload, Mapping):
        rows = payload.get("results")
        if isinstance(rows, list):
            candidates.extend(row for row in rows if isinstance(row, Mapping))
        if isinstance(payload.get("transcript"), str) and not candidates:
            transcript = str(payload["transcript"]).strip()
        if isinstance(payload.get("text"), str) and not candidates:
            transcript = str(payload["text"]).strip()
        if not language:
            language = str(payload.get("language", "")).strip()
    for row in candidates:
        file_value = str(row.get("file", ""))
        if file_value and Path(file_value).name != Path(source_name).name:
            continue
        text_value = row.get("text")
        if isinstance(text_value, str) and text_value.strip():
            transcript = text_value.strip()
        else:
            segments = row.get("segments")
            if isinstance(segments, list):
                transcript = "".join(
                    str(segment.get("text", ""))
                    for segment in segments
                    if isinstance(segment, Mapping)
                ).strip()
        language = str(row.get("language", language)).strip()
        if transcript:
            return transcript, {
                "model": model,
                "device": device,
                "compute_type": compute_type,
                "language": language,
            }
    if transcript:
        return transcript, {
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "language": language,
        }
    raise EvidenceError(f"ASR transcript for {source_name} is absent in {asr_path}")


@dataclass(frozen=True)
class FixtureSpec:
    key: str
    path: Path
    asr_path: Path
    transcript: str
    asr_model: str
    asr_device: str
    language: str
    transcript_verified: bool
    asr_compute_type: str = "not_reported"


@dataclass
class PreparedFixture:
    spec: FixtureSpec
    source_rate: int
    source_channels: int
    source_samples: int
    resampled_samples: int
    padded_samples: int
    vocal_onset_frame: int
    vocal_end_frame: int
    pcm16: np.ndarray
    frame_metrics: list[dict[str, Any]]
    metrics: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return len(self.frame_metrics)


def load_fixture_manifest(path: Path) -> tuple[dict[str, FixtureSpec], dict[str, Any]]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise EvidenceError(f"scenario manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"scenario manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("fixtures"), dict):
        raise EvidenceError("scenario manifest requires a fixtures object")
    required = ("long_request", "interruption", "memory_followup", "closure")
    specs: dict[str, FixtureSpec] = {}
    for key in required:
        item = manifest["fixtures"].get(key)
        if not isinstance(item, dict):
            raise EvidenceError(f"scenario manifest is missing fixture {key!r}")
        for field_name in ("path", "asr_file", "asr_model", "asr_device", "language"):
            if not isinstance(item.get(field_name), str) or not item[field_name].strip():
                raise EvidenceError(f"fixture {key!r} requires {field_name}")
        fixture_path = (manifest_path.parent / item["path"]).resolve()
        asr_path = (manifest_path.parent / item["asr_file"]).resolve()
        transcript, asr_evidence = _read_manifest_asr(asr_path, fixture_path.name)
        declared = item.get("transcript")
        verified = item.get("transcript_verified") is True
        declared_model = item["asr_model"].strip()
        declared_device = item["asr_device"].strip()
        declared_language = item["language"].strip()
        if not asr_evidence.get("model") or asr_evidence["model"].casefold() != declared_model.casefold():
            raise EvidenceError(f"ASR model metadata does not match manifest for {key}")
        if not asr_evidence.get("device") or asr_evidence["device"].casefold() != declared_device.casefold():
            raise EvidenceError(f"ASR device metadata does not match manifest for {key}")
        if not asr_evidence.get("language") or asr_evidence["language"].casefold() != declared_language.casefold():
            raise EvidenceError(f"ASR language metadata does not match manifest for {key}")
        if isinstance(declared, str) and declared.strip():
            # Keep ASR evidence authoritative; a manifest annotation cannot
            # silently substitute a different transcript.
            if re.sub(r"\W+", " ", declared.casefold()).strip() != re.sub(
                r"\W+", " ", transcript.casefold()
            ).strip():
                raise EvidenceError(f"manifest transcript does not match ASR for {key}")
        specs[key] = FixtureSpec(
            key=key,
            path=fixture_path,
            asr_path=asr_path,
            transcript=transcript,
            asr_model=declared_model,
            asr_device=declared_device,
            language=declared_language,
            transcript_verified=verified,
            asr_compute_type=str(asr_evidence.get("compute_type", "not_reported")),
        )
    return specs, {**manifest, "manifest_path": str(manifest_path)}


def prepare_fixture(
    spec: FixtureSpec,
    *,
    seed: int,
    noise_rms: float,
) -> PreparedFixture:
    """Load real PCM16 speech, resample if needed, noise, and pad without trimming."""
    if not NOISE_RMS_MIN <= float(noise_rms) <= NOISE_RMS_MAX:
        raise ValueError(
            f"noise RMS must be in [{NOISE_RMS_MIN:.4f}, {NOISE_RMS_MAX:.4f}]"
        )
    if not spec.path.is_file():
        raise EvidenceError(f"speech fixture not found: {spec.path}")
    with wave.open(str(spec.path), "rb") as source:
        source_channels = source.getnchannels()
        source_width = source.getsampwidth()
        source_rate = source.getframerate()
        source_samples = source.getnframes()
        if source.getcomptype() != "NONE" or source_channels != 1 or source_width != 2:
            raise EvidenceError("speech fixture must be mono, uncompressed PCM16 WAV")
        raw = source.readframes(source_samples)
    if source_samples <= 0:
        raise EvidenceError("speech fixture is empty")
    input_i16 = np.frombuffer(raw, dtype="<i2")
    speech = input_i16.astype(np.float64) / 32768.0
    if not np.isfinite(speech).all():
        raise EvidenceError("speech fixture has NaN or Inf samples")
    resampled = speech
    if source_rate != MIC_SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), MIC_SAMPLE_RATE)
        resampled = resample_poly(
            speech,
            MIC_SAMPLE_RATE // divisor,
            int(source_rate) // divisor,
        ).astype(np.float64, copy=False)
    if resampled.size <= 0 or _rms(resampled) < INPUT_RMS_MIN:
        raise EvidenceError("speech fixture RMS is below 0.015 before PCM16 quantization")
    padded_samples = int(math.ceil(resampled.size / MIC_FRAME_SAMPLES) * MIC_FRAME_SAMPLES)
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(padded_samples).astype(np.float64)
    noise *= float(noise_rms) / _rms(noise)
    padded_speech = np.zeros(padded_samples, dtype=np.float64)
    padded_speech[: resampled.size] = resampled
    mixed = padded_speech + noise
    if not np.isfinite(mixed).all():
        raise EvidenceError("mixed speech fixture has NaN or Inf samples")
    scaled = mixed * 32768.0
    clipping_rail_samples = int(np.count_nonzero((scaled <= -32768.0) | (scaled >= 32767.0)))
    out_of_range_samples = int(np.count_nonzero((scaled < -32768.0) | (scaled > 32767.0)))
    if clipping_rail_samples:
        raise EvidenceError(
            f"input would clip before PCM16 quantization: reached {clipping_rail_samples} rail samples"
        )
    quantized = np.rint(scaled).clip(-32768, 32767).astype("<i2")
    frame_metrics: list[dict[str, Any]] = []
    clean_frame_rms: list[float] = []
    for index in range(padded_samples // MIC_FRAME_SAMPLES):
        start = index * MIC_FRAME_SAMPLES
        end = start + MIC_FRAME_SAMPLES
        clean_rms = _rms(padded_speech[start:end])
        clean_frame_rms.append(clean_rms)
        frame_metrics.append({
            "frame_index": index,
            "rms_before_quantization": _rms(mixed[start:end]),
            "noise_rms_before_quantization": _rms(noise[start:end]),
            "source_speech_rms_before_quantization": clean_rms,
            "prequantization_pcm16_rail_samples": int(np.count_nonzero(
                (scaled[start:end] <= -32768.0) | (scaled[start:end] >= 32767.0)
            )),
            "prequantization_pcm16_out_of_range_samples": int(np.count_nonzero(
                (scaled[start:end] < -32768.0) | (scaled[start:end] > 32767.0)
            )),
        })
    vocal_frames = [i for i, value in enumerate(clean_frame_rms) if value >= INPUT_RMS_MIN]
    if not vocal_frames:
        raise EvidenceError("speech fixture has no 80 ms vocal frame with RMS >= 0.015")
    return PreparedFixture(
        spec=spec,
        source_rate=int(source_rate),
        source_channels=int(source_channels),
        source_samples=int(source_samples),
        resampled_samples=int(resampled.size),
        padded_samples=padded_samples,
        vocal_onset_frame=vocal_frames[0],
        vocal_end_frame=vocal_frames[-1],
        pcm16=quantized,
        frame_metrics=frame_metrics,
        metrics={
            "source_path": str(spec.path),
            "source_sha256": _sha256_bytes(raw),
            "asr_path": str(spec.asr_path),
            "asr_sha256": _sha256_bytes(spec.asr_path.read_bytes()),
            "asr_model": spec.asr_model,
            "asr_device": spec.asr_device,
            "asr_language": spec.language,
            "asr_compute_type": spec.asr_compute_type,
            "asr_transcript": spec.transcript,
            "transcript_verified": spec.transcript_verified,
            "source_sample_rate": int(source_rate),
            "input_sample_rate": MIC_SAMPLE_RATE,
            "channels": 1,
            "source_samples_preserved": int(source_samples),
            "resampled_samples": int(resampled.size),
            "padded_samples_total": padded_samples,
            "padding_samples": padded_samples - int(resampled.size),
            "frame_samples": MIC_FRAME_SAMPLES,
            "frame_duration_ms": FRAME_MS,
            "fixture_rms_before_quantization": _rms(mixed),
            "source_speech_rms_before_noise": _rms(resampled),
            "noise_rms_before_quantization": _rms(noise),
            "peak_before_quantization": float(np.max(np.abs(mixed))),
            "prequantization_pcm16_rail_samples": clipping_rail_samples,
            "prequantization_pcm16_out_of_range_samples": out_of_range_samples,
            "vocal_onset_frame": vocal_frames[0],
            "vocal_onset_ms": round(vocal_frames[0] * FRAME_MS, 3),
            "vocal_end_frame_exclusive": vocal_frames[-1] + 1,
            "vocal_end_ms": round((vocal_frames[-1] + 1) * FRAME_MS, 3),
            "low_rms_frames": sum(value < INPUT_RMS_MIN for value in clean_frame_rms),
            "frame_metrics": frame_metrics,
        },
    )


def make_noise_frame(rng: np.random.Generator, target_rms: float) -> np.ndarray:
    return make_noise_frame_with_metrics(rng, target_rms)[0]


def make_noise_frame_with_metrics(
    rng: np.random.Generator, target_rms: float
) -> tuple[np.ndarray, float, int]:
    samples = rng.standard_normal(MIC_FRAME_SAMPLES).astype(np.float64)
    samples *= target_rms / _rms(samples)
    scaled = samples * 32768.0
    if not np.isfinite(scaled).all():
        raise EvidenceError("idle noise frame has non-finite PCM")
    rails = int(np.count_nonzero((scaled <= -32768) | (scaled >= 32767)))
    out_of_range = int(np.count_nonzero((scaled < -32768) | (scaled > 32767)))
    if rails:
        raise EvidenceError("idle noise frame reached a PCM16 clipping rail")
    if out_of_range:
        raise EvidenceError("idle noise frame is outside PCM16 range")
    return np.rint(scaled).clip(-32768, 32767).astype("<i2"), _rms(samples), rails


class PacingMonitor:
    """80 ms sender clock with no catch-up bursts and explicit jitter bounds."""

    def __init__(
        self,
        *,
        tolerance_ms: float = DEFAULT_PACING_TOLERANCE_MS,
        max_gap_ms: float = DEFAULT_MAX_GAP_MS,
    ) -> None:
        if tolerance_ms < 0 or max_gap_ms <= FRAME_MS:
            raise ValueError("invalid pacing bounds")
        self.tolerance_ns = int(tolerance_ms * 1_000_000)
        self.max_gap_ns = int(max_gap_ms * 1_000_000)
        self.previous_send_ns: int | None = None
        self.deadline_ns: int | None = None
        self.max_lateness_ns = 0
        self.max_gap_observed_ns = 0
        self.samples: list[dict[str, int]] = []

    def start(self, now_ns: int) -> None:
        if self.deadline_ns is None:
            self.deadline_ns = int(now_ns)

    def record_send(self, sent_ns: int, backpressure_wait_ns: int = 0) -> int:
        if self.deadline_ns is None:
            self.start(sent_ns)
        assert self.deadline_ns is not None
        sent = int(sent_ns)
        backpressure_wait = int(backpressure_wait_ns)
        lateness = max(0, sent - self.deadline_ns)
        gap = 0 if self.previous_send_ns is None else sent - self.previous_send_ns
        net_gap = max(0, gap - backpressure_wait)
        self.max_lateness_ns = max(self.max_lateness_ns, lateness)
        if lateness > self.tolerance_ns:
            raise EvidenceError(
                f"input scheduling lateness {lateness / 1e6:.1f}ms exceeds "
                f"{self.tolerance_ns / 1e6:.1f}ms"
            )
        if backpressure_wait <= 0:
            self.max_gap_observed_ns = max(self.max_gap_observed_ns, gap)
            if gap >= self.max_gap_ns:
                raise EvidenceError(
                    f"input frame gap {gap / 1e6:.1f}ms reached the {self.max_gap_ns / 1e6:.1f}ms limit"
                )
        else:
            self.max_gap_observed_ns = max(self.max_gap_observed_ns, net_gap)
            if net_gap >= self.max_gap_ns:
                raise EvidenceError(
                    f"input frame gap after backpressure accounting {net_gap / 1e6:.1f}ms "
                    f"(raw {gap / 1e6:.1f}ms, wait {backpressure_wait / 1e6:.1f}ms) "
                    f"reached the {self.max_gap_ns / 1e6:.1f}ms limit"
                )
        self.samples.append({
            "lateness_ns": lateness,
            "gap_ns": gap,
            "net_gap_ns": net_gap,
            "backpressure_wait_ns": backpressure_wait,
        })
        self.previous_send_ns = sent
        # Anchor the next deadline to the actual send to avoid compressing a
        # late frame and creating a catch-up burst.
        self.deadline_ns = sent + FRAME_DURATION_NS
        return lateness

    def report(self) -> dict[str, Any]:
        lateness_ms = [row["lateness_ns"] / 1e6 for row in self.samples]
        gaps_ms = [row["gap_ns"] / 1e6 for row in self.samples if row["gap_ns"]]
        return {
            "nominal_frame_ms": FRAME_MS,
            "tolerance_ms": self.tolerance_ns / 1e6,
            "max_gap_ms": self.max_gap_ns / 1e6,
            "no_catch_up_burst": True,
            "frames_sent": len(self.samples),
            "max_lateness_ms": max(lateness_ms) if lateness_ms else None,
            "p95_lateness_ms": float(np.percentile(lateness_ms, 95)) if lateness_ms else None,
            "max_interframe_gap_ms": max(gaps_ms) if gaps_ms else None,
            "p95_interframe_gap_ms": float(np.percentile(gaps_ms, 95)) if gaps_ms else None,
            "max_net_interframe_gap_ms": max((
                row["net_gap_ns"] / 1e6 for row in self.samples if row["gap_ns"]
            ), default=None),
        }


@dataclass
class StimulusDelivery:
    delivery_id: int
    fixture: PreparedFixture
    scheduled_ns: int = field(default_factory=time.monotonic_ns)
    frame_index: int = 0
    frames_sent: int = 0
    first_sent_ns: int | None = None
    first_vocal_sent_ns: int | None = None
    last_sent_ns: int | None = None
    last_vocal_sent_ns: int | None = None


class StimulusQueue:
    def __init__(self, *, noise_rms: float, seed: int) -> None:
        self.noise_rms = float(noise_rms)
        self.rng = np.random.default_rng(int(seed))
        self.pending: deque[StimulusDelivery] = deque()
        self.active: StimulusDelivery | None = None
        self.next_delivery_id = 1
        self.deliveries: dict[int, StimulusDelivery] = {}

    def schedule(self, fixture: PreparedFixture) -> StimulusDelivery:
        delivery = StimulusDelivery(self.next_delivery_id, fixture)
        self.next_delivery_id += 1
        self.deliveries[delivery.delivery_id] = delivery
        self.pending.append(delivery)
        return delivery

    def next_frame(self) -> tuple[np.ndarray, StimulusDelivery | None, int | None, float, int]:
        if self.active is None and self.pending:
            self.active = self.pending.popleft()
        if self.active is None:
            samples, frame_rms, rails = make_noise_frame_with_metrics(self.rng, self.noise_rms)
            return samples, None, None, frame_rms, rails
        delivery = self.active
        index = delivery.frame_index
        start = index * MIC_FRAME_SAMPLES
        end = start + MIC_FRAME_SAMPLES
        samples = delivery.fixture.pcm16[start:end]
        if samples.size != MIC_FRAME_SAMPLES:
            raise EvidenceError("prepared input frame is not exactly 1280 PCM16 samples")
        delivery.frame_index += 1
        if delivery.frame_index >= delivery.fixture.frame_count:
            self.active = None
        frame = delivery.fixture.frame_metrics[index]
        return (
            samples,
            delivery,
            index,
            float(frame["rms_before_quantization"]),
            int(frame["prequantization_pcm16_rail_samples"]),
        )


@dataclass
class ResponseRecord:
    response_id: str
    created_event: dict[str, Any]
    created_ns: int
    scenario_tag: str = "unassigned"
    delivery_id: int | None = None
    delivery_started_ns: int | None = None
    delivery_first_sent_ns: int | None = None
    delivery_attempt_index: int | None = None
    delivery_attempt_status: str | None = None
    recovered_by_response_id: str | None = None
    prompt_speech_active_at_cancel: bool | None = None
    audible_overlap_with_prompt_speech: bool | None = None
    action: str | None = None
    text_parts: list[str] = field(default_factory=list)
    received_pcm: list[np.ndarray] = field(default_factory=list)
    played_pcm: list[np.ndarray] = field(default_factory=list)
    audio_samples: int = 0
    codec_frames: int = 0
    vocal_frames: int = 0
    segment_ids: set[str] = field(default_factory=set)
    marker_ids: set[str] = field(default_factory=set)
    ack_sent_ids: set[str] = field(default_factory=set)
    audio_telemetry_complete: bool = True
    audio_quality_events: int = 0
    first_audio_ns: int | None = None
    playout_started_ns: int | None = None
    last_pcm_consumed_ns: int | None = None
    playout_completed_ns: int | None = None
    pcm_consumed_samples: int = 0
    playout_onset_ns: int | None = None
    done_event: dict[str, Any] | None = None
    done_ns: int | None = None
    cancelled_event: dict[str, Any] | None = None
    cancelled_ns: int | None = None
    flush_ns: int | None = None
    finalizer_event: dict[str, Any] | None = None
    final_observation_ns: int | None = None
    long_response_input_window: dict[str, Any] | None = None
    created_signal: asyncio.Event = field(default_factory=asyncio.Event)
    terminal_signal: asyncio.Event = field(default_factory=asyncio.Event)
    playout_onset_signal: asyncio.Event = field(default_factory=asyncio.Event)
    finalizer_signal: asyncio.Event = field(default_factory=asyncio.Event)


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.count = 0

    def write(
        self,
        direction: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        monotonic_ns: int | None = None,
    ) -> None:
        row = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns),
            "direction": direction,
            "type": kind,
            "payload": _json_safe(payload),
        }
        self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handle.flush()
        self.count += 1

    def close(self) -> None:
        self.handle.close()


class RealDialogueSimulator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.playback_buffer_seconds = float(args.playback_buffer_seconds)
        self.playback_capacity_samples = _playback_capacity_samples(
            str(args.mode), self.playback_buffer_seconds
        )
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = EventLog(self.output_dir / "events.jsonl")
        self.specs, self.input_manifest = load_fixture_manifest(Path(args.fixture_manifest))
        self.fixtures = {
            name: prepare_fixture(
                spec,
                seed=int(args.seed) + index * 101,
                noise_rms=float(args.noise_rms),
            )
            for index, (name, spec) in enumerate(self.specs.items())
        }
        self.stimuli = StimulusQueue(noise_rms=args.noise_rms, seed=args.seed + 991)
        self.playback = BoundedPCMPlayback(
            max_samples=self.playback_capacity_samples
        )
        self.pacing = PacingMonitor(
            tolerance_ms=args.pacing_tolerance_ms,
            max_gap_ms=min(float(args.max_input_gap_ms), DEFAULT_MAX_GAP_MS),
        )
        if float(args.max_input_gap_ms) > DEFAULT_MAX_GAP_MS:
            raise EvidenceError(
                f"--max-input-gap-ms cannot exceed {DEFAULT_MAX_GAP_MS:.0f}ms"
            )
        self.responses: dict[str, ResponseRecord] = {}
        self.response_created_queue: asyncio.Queue[ResponseRecord] = asyncio.Queue()
        self.input_sent: deque[dict[str, Any]] = deque()
        self.input_ack_ids: set[int] = set()
        self.input_frame_sends: list[dict[str, int]] = []
        self.input_frame_acks: list[dict[str, int]] = []
        self.input_ack_latency_ms: list[float] = []
        self.input_local_frame_id = 0
        self.input_inflight_high_water = 0
        self.input_ack_expected: int | None = 0
        self.input_dropped_frames = 0
        self.input_wav_bytes = bytearray()
        self.session_playout_parts: list[np.ndarray] = []
        self.current_response_id: str | None = None
        self.socket: Any = None
        self.send_lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.input_pause = asyncio.Event()
        self.fatal_event = asyncio.Event()
        self.fatal_error: str | None = None
        self.error_categories: dict[str, list[str]] = {
            "runtime_error": [],
            "transport_error": [],
            "producer_limit": [],
            "acoustic_limit": [],
            "harness_interruption": [],
        }
        self.tasks: list[asyncio.Task[Any]] = []
        self.ack_sent_count = 0
        self.ack_rejected_count = 0
        self.ack_sent_events: list[dict[str, Any]] = []
        self.playback_callback_gaps_ms: list[float] = []
        self.playback_callback_max_lateness_ms = 0.0
        self.playback_callback_previous_ns: int | None = None
        self.playback_callback_count = 0
        self.playback_total_device_samples = 0
        self.playback_actual_pcm_samples_consumed = 0
        self.playback_queue_peak_samples = 0
        self.playback_queue_trace: list[dict[str, Any]] = []
        self.playback_epoch_ns: int | None = None
        self.playback_onsets: list[dict[str, Any]] = []
        self.receiver_events = 0
        self.ws_errors: list[str] = []
        self.inflight_limit: int | None = None
        self.session_created: dict[str, Any] | None = None
        self.session_updated: dict[str, Any] | None = None
        self.runtime_kv_evidence: list[dict[str, Any]] = []
        self.runtime_audio_telemetry_complete = True
        self.scenario_errors: list[str] = []
        self.evidence_gaps: list[str] = []
        self.outcome = "INCOMPLETE"
        self.run_started_at_utc: datetime | None = None
        self.run_finished_at_utc: datetime | None = None
        self.tls_evidence: dict[str, Any] = {
            "scheme": None,
            "peer_certificate_verified": None,
            "ca_certificate": None,
            "server_hostname_override": args.tls_server_name,
            "insecure_loopback_opt_in": bool(args.insecure_loopback),
        }

    def set_fatal(self, exc: BaseException) -> None:
        if self.fatal_error is None:
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.ws_errors.append(self.fatal_error)
            if isinstance(exc, (ConnectionError, OSError, websockets.exceptions.WebSocketException)):
                category = "transport_error"
            elif isinstance(exc, asyncio.CancelledError):
                category = "harness_interruption"
            else:
                category = "runtime_error"
            self.error_categories[category].append(self.fatal_error)
            self.fatal_event.set()
            self.stop_event.set()
            self.log.write("harness", "fatal_error", {"message": self.fatal_error})

    def _record_finish_limits(self, response_id: str, event: Mapping[str, Any]) -> None:
        runtime_metrics = event.get("runtime_metrics")
        completion = (
            runtime_metrics.get("response_completion")
            if isinstance(runtime_metrics, Mapping) else None
        )
        if not isinstance(completion, Mapping):
            return
        if completion.get("text_terminal_reason") == "producer_limit":
            self.error_categories["producer_limit"].append(str(response_id))
        if completion.get("acoustic_limit_reached") is True:
            self.error_categories["acoustic_limit"].append(str(response_id))

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            if self.socket is None or bool(getattr(self.socket, "closed", False)):
                raise EvidenceError("WebSocket closed before outbound protocol event")
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            await self.socket.send(raw)
        self.log.write("client", str(payload.get("type", "unknown")), {
            key: value for key, value in payload.items() if key != "audio"
        })

    async def _recv_initial(
        self,
        *,
        phase: str,
        phase_started_at: float | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        started_at = loop.time() if phase_started_at is None else phase_started_at
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else "<no-current-task>"
        try:
            raw = await asyncio.wait_for(
                self.socket.recv(), timeout=self.args.handshake_timeout_s
            )
        except TimeoutError as exc:
            elapsed_s = max(0.0, loop.time() - started_at)
            raise TimeoutError(
                f"timed out waiting for {phase} after {elapsed_s:.3f}s "
                f"(task={task_name!r})"
            ) from exc
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise EvidenceError("WebSocket handshake event is not a JSON object")
        return event

    @staticmethod
    def _voice_from_created(event: Mapping[str, Any]) -> str | None:
        voice = event.get("voice")
        if isinstance(voice, str) and voice.strip():
            return voice.strip()
        session = event.get("session")
        if isinstance(session, Mapping):
            nested = session.get("voice")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return None

    async def _wait_ready(self) -> None:
        loop = asyncio.get_running_loop()
        created_wait_started_at = loop.time()
        self.session_created = await self._recv_initial(
            phase="session.created",
            phase_started_at=created_wait_started_at,
        )
        self.log.write("server", "session.created", self.session_created)
        if self.session_created.get("type") != "session.created":
            raise EvidenceError("expected session.created before microphone audio")
        if self.session_created.get("turn_policy") != TURN_POLICY:
            raise EvidenceError(
                "session.created did not announce gemma_native_control6; refusing a stale policy"
            )
        announced_voice = self._voice_from_created(self.session_created)
        if not announced_voice:
            self.evidence_gaps.append("session.created did not announce its effective voice")
        capacity = self.session_created.get("input_audio_buffer_capacity_frames")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            self.evidence_gaps.append("session.created omitted a positive input frame capacity")
            self.inflight_limit = MAX_INPUT_INFLIGHT - 1
        else:
            self.inflight_limit = min(MAX_INPUT_INFLIGHT - 1, capacity)
        await self._send({
            "type": "session.update",
            "session": {
                "auto_response": True,
                "turn_policy": TURN_POLICY,
                "input_audio_buffer_acknowledgements": True,
                "input_audio_transcription": False,
                "instructions": self.args.instructions,
                "response_profile": getattr(self.args, "response_profile", "long"),
            },
        })
        updated_wait_started_at = loop.time()
        deadline = updated_wait_started_at + self.args.handshake_timeout_s
        while loop.time() < deadline:
            event = await self._recv_initial(
                phase="session.updated",
                phase_started_at=updated_wait_started_at,
            )
            kind = event.get("type")
            self.log.write("server", str(kind), event)
            if kind == "session.updated":
                self.session_updated = event
                self.log.write("harness", "transport_verified", self.tls_evidence)
                created_voice = self._voice_from_created(self.session_created)
                updated_voice = event.get("voice")
                if event.get("turn_policy") != TURN_POLICY:
                    raise EvidenceError("session.updated did not confirm gemma_native_control6")
                if event.get("response_profile") != getattr(
                    self.args, "response_profile", "long"
                ):
                    raise EvidenceError("session.updated did not confirm the requested response_profile")
                if created_voice and updated_voice != created_voice:
                    raise EvidenceError(
                        f"server changed voice after session.created: {created_voice!r} -> {updated_voice!r}"
                    )
                if not created_voice:
                    self.evidence_gaps.append("effective voice could not be matched to session.created")
                return
            if kind == "error":
                raise EvidenceError(f"session.update failed: {event.get('error')}")
        raise EvidenceError("session.updated handshake timed out")

    def _websocket_tls_options(self) -> dict[str, Any]:
        parsed = urlsplit(self.args.url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        tls_server_name = self.args.tls_server_name
        if tls_server_name is not None:
            tls_server_name = str(tls_server_name).strip()
            if (
                not tls_server_name
                or any(char in tls_server_name for char in "\r\n\t /?#@")
            ):
                raise EvidenceError("--tls-server-name must be a hostname or IP address, not a URL")
        try:
            loopback = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
        if scheme not in {"ws", "wss"} or not hostname:
            raise EvidenceError("realtime URL must be an absolute ws:// or wss:// URL")
        if self.args.insecure_loopback:
            if tls_server_name is not None:
                raise EvidenceError("--tls-server-name cannot be combined with --insecure-loopback")
            if self.args.mode != "diagnostic" or not loopback:
                raise EvidenceError("--insecure-loopback is allowed only for diagnostic loopback runs")
            self.tls_evidence.update({
                "scheme": scheme,
                "peer_certificate_verified": False if scheme == "wss" else None,
            })
            self.evidence_gaps.append(
                "insecure loopback diagnostic transport cannot certify acceptance"
            )
            if scheme == "wss":
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                return {"ssl": context}
            return {}
        if scheme != "wss":
            if tls_server_name is not None:
                raise EvidenceError("--tls-server-name requires wss:// TLS")
            raise EvidenceError("unencrypted ws:// requires --insecure-loopback in diagnostic loopback mode")
        ca_path = self.args.ca_cert
        if ca_path:
            ca = Path(ca_path).expanduser().resolve()
            if not ca.is_file():
                raise EvidenceError(f"local CA certificate file not found: {ca}")
            context = ssl.create_default_context(cafile=str(ca))
            self.tls_evidence.update({
                "scheme": scheme,
                "peer_certificate_verified": True,
                "ca_certificate": str(ca),
                "server_hostname_override": tls_server_name,
            })
        else:
            context = ssl.create_default_context()
            self.tls_evidence.update({
                "scheme": scheme,
                "peer_certificate_verified": True,
                "ca_certificate": "system trust store",
                "server_hostname_override": tls_server_name,
            })
        options = {"ssl": context}
        if tls_server_name is not None:
            options["server_hostname"] = tls_server_name
        return options

    async def _sender_loop(self) -> None:
        self.pacing.start(time.monotonic_ns())
        deadline_ns = self.pacing.deadline_ns or time.monotonic_ns()
        while not self.stop_event.is_set():
            if self.input_pause.is_set():
                await asyncio.sleep(0.01)
                continue
            delay_ns = deadline_ns - time.monotonic_ns()
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            if self.stop_event.is_set():
                return
            scheduler_delay_ns = max(0, time.monotonic_ns() - deadline_ns)
            inflight = len(self.input_sent)
            wait_deadline_ns = time.monotonic_ns() + 3_500_000_000
            waited_backpressure = False
            backpressure_started_ns: int | None = None
            while inflight >= int(self.inflight_limit or MAX_INPUT_INFLIGHT - 1) and not self.stop_event.is_set():
                if backpressure_started_ns is None:
                    backpressure_started_ns = time.monotonic_ns()
                waited_backpressure = True
                if time.monotonic_ns() > wait_deadline_ns:
                    raise EvidenceError(
                        f"input backpressure stalled: in_flight={inflight} unacknowledged for >3.5s; audio was not dropped"
                    )
                await asyncio.sleep(0.005)
                inflight = len(self.input_sent)
            if self.stop_event.is_set():
                return
            backpressure_wait_ns = (
                max(0, time.monotonic_ns() - backpressure_started_ns)
                if backpressure_started_ns is not None else 0
            )
            if waited_backpressure:
                self.pacing.deadline_ns = time.monotonic_ns()
            if inflight + 1 >= MAX_INPUT_INFLIGHT:
                raise EvidenceError("input in_flight reached the fail-closed threshold 16")
            (
                samples,
                delivery,
                fixture_frame,
                prequantization_rms,
                prequantization_rails,
            ) = self.stimuli.next_frame()
            if samples.size != MIC_FRAME_SAMPLES or samples.dtype != np.dtype("<i2"):
                raise EvidenceError("input frame must be exactly 1280 PCM16LE samples")
            pcm_bytes = np.ascontiguousarray(samples, dtype="<i2").tobytes()
            pcm_float = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float64) / 32768.0
            if not np.isfinite(pcm_float).all():
                raise EvidenceError("input frame contains NaN or Inf")
            frame_rms_after_quantization = _rms(pcm_float)
            if (
                not math.isfinite(prequantization_rms)
                or not math.isfinite(frame_rms_after_quantization)
                or prequantization_rms <= 0
            ):
                raise EvidenceError("input frame has invalid prequantization or PCM16 RMS")
            if prequantization_rails:
                raise EvidenceError(
                    f"input frame would clip before PCM16 quantization ({prequantization_rails} samples)"
                )
            if np.any((samples == -32768) | (samples == 32767)):
                # This check is telemetry only; the actual fail-closed rail
                # count was measured before quantization in prepare_fixture.
                frame_rail_count = int(np.count_nonzero((samples == -32768) | (samples == 32767)))
            else:
                frame_rail_count = 0
            send_started_ns = time.monotonic_ns()
            pending = {
                "local_frame_id": self.input_local_frame_id,
                "sent_ns": send_started_ns,
                "delivery_id": delivery.delivery_id if delivery else None,
                "fixture": delivery.fixture.spec.key if delivery else "idle_noise_floor",
                "fixture_frame_index": fixture_frame,
                "frame_rms_before_quantization": prequantization_rms,
                "frame_rms_after_pcm16_quantization": frame_rms_after_quantization,
                "prequantization_pcm16_rail_samples": prequantization_rails,
                "noise_target_rms": self.args.noise_rms,
                "pcm16_rail_samples_after_quantization_telemetry_only": frame_rail_count,
                "sender_scheduler_delay_ns": int(scheduler_delay_ns),
                "backpressure_wait_ns": int(backpressure_wait_ns),
                "waited_for_backpressure": bool(waited_backpressure),
            }
            # Reserve the in-flight slot before awaiting the network send so a
            # fast server ACK can never race an empty local pending queue.
            self.input_sent.append(pending)
            self.input_inflight_high_water = max(
                self.input_inflight_high_water, len(self.input_sent)
            )
            pending["inflight_after_reservation"] = len(self.input_sent)
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_bytes).decode("ascii"),
            }
            await self._send(event)
            sent_ns = time.monotonic_ns()
            pending["sent_ns"] = sent_ns
            self.pacing.record_send(
                sent_ns,
                backpressure_wait_ns=int(backpressure_wait_ns),
            )
            if delivery is not None:
                if delivery.first_sent_ns is None:
                    delivery.first_sent_ns = sent_ns
                delivery.frames_sent += 1
                delivery.last_sent_ns = sent_ns
                assert fixture_frame is not None
                source_speech_rms = float(
                    delivery.fixture.frame_metrics[fixture_frame].get(
                        "source_speech_rms_before_quantization", 0.0
                    )
                )
                if source_speech_rms >= INPUT_RMS_MIN:
                    if delivery.first_vocal_sent_ns is None:
                        delivery.first_vocal_sent_ns = sent_ns
                    delivery.last_vocal_sent_ns = sent_ns
            self.input_wav_bytes.extend(pcm_bytes)
            self.log.write("client", "input_frame_sent", pending)
            self.input_frame_sends.append({
                "local_frame_id": int(pending["local_frame_id"]),
                "sent_ns": int(sent_ns),
                "sender_scheduler_delay_ns": int(scheduler_delay_ns),
                "backpressure_wait_ns": int(backpressure_wait_ns),
                "waited_for_backpressure": int(bool(waited_backpressure)),
            })
            self.input_local_frame_id += 1
            # Anchor the next deadline at this send time; late frames never
            # cause a compressed catch-up burst.
            deadline_ns = sent_ns + FRAME_DURATION_NS

    def _set_active_response(self, record: ResponseRecord | None) -> None:
        self.current_response_id = None if record is None else record.response_id

    def _clear_active_response_if_current(self, record: ResponseRecord) -> None:
        if self.current_response_id == record.response_id:
            self._set_active_response(None)

    def _trace_playback_queue(self, event: str, **fields: Any) -> None:
        self.playback_queue_trace.append({
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        })

    def _record_audio_quality(
        self,
        event: Mapping[str, Any],
        record: ResponseRecord,
        *,
        expected_samples: int,
    ) -> None:
        quality = event.get("raw_source_audio_quality")
        if not isinstance(quality, Mapping):
            record.audio_telemetry_complete = False
            self.runtime_audio_telemetry_complete = False
            return
        finite = quality.get("finite")
        rms = quality.get("rms")
        peak = quality.get("peak")
        rails = quality.get("clipping_rail_samples")
        count = quality.get("sample_count")
        if finite is False or any(
            isinstance(value, float) and not math.isfinite(value)
            for value in (rms, peak)
        ):
            raise EvidenceError("server raw-source PCM telemetry reports NaN or Inf")
        if (
            finite is not True
            or isinstance(rms, bool) or not isinstance(rms, (int, float)) or not math.isfinite(rms)
            or rms < 0
            or isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak)
            or peak < 0 or rms > peak + 1e-8
            or isinstance(rails, bool) or not isinstance(rails, int) or rails < 0
            or isinstance(count, bool) or not isinstance(count, int) or count != expected_samples
        ):
            record.audio_telemetry_complete = False
            self.runtime_audio_telemetry_complete = False
            return
        if rails > 0:
            raise EvidenceError("server raw-source PCM telemetry reports clipped rail samples")
        record.audio_quality_events += 1

    def _record_kv_values(self, event: Mapping[str, Any]) -> None:
        metrics = event.get("runtime_metrics")
        if not isinstance(metrics, Mapping):
            return
        keys = {
            "client_consumed_pcm", "unified_native_rollback", "unified_native_finalizer",
            "confirmed_exclusive_kv_cursor", "spoken_prefix_exclusive_cursor",
        }
        found: dict[str, Any] = {}

        def visit(node: Any) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    if key in keys and value is not None:
                        found[str(key)] = value
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(metrics)
        if found:
            self.runtime_kv_evidence.append({
                "event_type": event.get("type"),
                "response_id": event.get("response_id"),
                "values": found,
            })

    async def _receiver_loop(self) -> None:
        async for raw in self.socket:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise EvidenceError("server sent a non-object realtime event")
            self.receiver_events += 1
            kind = str(event.get("type", ""))
            received_ns = time.monotonic_ns()
            if kind == "response.audio.delta":
                log_payload = {key: value for key, value in event.items() if key != "delta"}
                packet = decode_response_audio(event)
                log_payload.update({
                    "sample_count": int(packet.pcm.size),
                    "pcm_sha256": _sha256_bytes(np.ascontiguousarray(packet.pcm).tobytes()),
                })
            else:
                log_payload = dict(event)
            self.log.write(
                "server", kind or "unknown", log_payload, monotonic_ns=int(received_ns)
            )
            if kind == "error":
                detail = event.get("error", {})
                message = str(detail)
                if re.search(r"cuda|cublas|cudnn|device-side|out of memory", message, re.I):
                    raise EvidenceError(f"CUDA/runtime error: {message}")
                if detail.get("code") == "INPUT_AUDIO_OVERLOAD":
                    raise EvidenceError(f"input audio overload: {message}")
                if detail.get("code") == "STALE_CLIENT_CONSUMPTION_ACK":
                    self.ack_rejected_count += 1
                    raise EvidenceError(f"server rejected client playback ACK: {message}")
                raise EvidenceError(f"server realtime error: {message}")
            if kind == "input_audio_buffer.consumed":
                frame_id = event.get("frame_id")
                if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                    raise EvidenceError("input_audio_buffer.consumed omitted a valid frame_id")
                if frame_id in self.input_ack_ids:
                    raise EvidenceError(f"duplicate input ACK for frame {frame_id}")
                if self.input_ack_expected is not None and frame_id != self.input_ack_expected:
                    raise EvidenceError(
                        f"input ACK gap/reorder: expected frame {self.input_ack_expected}, got {frame_id}"
                    )
                if not self.input_sent:
                    raise EvidenceError(f"unsolicited input ACK for frame {frame_id}")
                pending = self.input_sent.popleft()
                self.input_ack_ids.add(frame_id)
                self.input_frame_acks.append({
                    "local_frame_id": int(pending["local_frame_id"]),
                    "received_ns": int(received_ns),
                })
                self.input_ack_expected = frame_id + 1
                self.input_ack_latency_ms.append((received_ns - pending["sent_ns"]) / 1e6)
                self.log.write("client", "input_frame_ack_received", {
                    "server_frame_id": frame_id,
                    "local_frame_id": pending["local_frame_id"],
                    "ack_latency_ms": self.input_ack_latency_ms[-1],
                }, monotonic_ns=int(received_ns))
                continue
            if kind == "response.created":
                response_id = event.get("response_id")
                if not isinstance(response_id, str) or not response_id or response_id in self.responses:
                    raise EvidenceError("response.created has a missing or reused response_id")
                record = ResponseRecord(
                    response_id=response_id,
                    created_event=event,
                    created_ns=received_ns,
                    action=str(event.get("effective_action", event.get("action", ""))).upper() or None,
                )
                self.responses[response_id] = record
                previous_response_id = self.current_response_id
                pending_before = self.playback.pending_samples
                self.playback_queue_peak_samples = max(self.playback_queue_peak_samples, pending_before)
                self.playback.begin_response(response_id)
                pending_after = self.playback.pending_samples
                self.playback_queue_peak_samples = max(self.playback_queue_peak_samples, pending_after)
                self._trace_playback_queue(
                    "response_begin",
                    response_id=response_id,
                    previous_response_id=previous_response_id,
                    generation_id=event.get("generation_id"),
                    pending_before_samples=pending_before,
                    pending_after_samples=pending_after,
                )
                self._set_active_response(record)
                self.response_created_queue.put_nowait(record)
                record.created_signal.set()
                continue
            response_id = event.get("response_id")
            record = self.responses.get(str(response_id)) if response_id is not None else None
            self._record_kv_values(event)
            if kind == "response.output_text.delta" and record is not None:
                if record.cancelled_event is not None:
                    raise EvidenceError(
                        f"response {record.response_id} emitted text after cancellation"
                    )
                delta = event.get("delta")
                if isinstance(delta, str):
                    record.text_parts.append(delta)
            elif kind == "response.audio.delta":
                packet = decode_response_audio(event)
                if record is None:
                    raise EvidenceError("response audio arrived before response.created")
                self._record_audio_quality(
                    event, record, expected_samples=int(packet.pcm.size)
                )
                if record.first_audio_ns is None:
                    record.first_audio_ns = received_ns
                record.segment_ids.add(packet.speaker_segment_id)
                record.received_pcm.append(packet.pcm.copy())
                record.audio_samples += int(packet.pcm.size)
                record.codec_frames += int(packet.last_frame_id - packet.first_frame_id + 1)
                frame_count = packet.pcm.size // CODEC_FRAME_SAMPLES
                framed = packet.pcm[: frame_count * CODEC_FRAME_SAMPLES].reshape(
                    frame_count, CODEC_FRAME_SAMPLES
                )
                record.vocal_frames += sum(
                    _rms(frame) >= INPUT_RMS_MIN for frame in framed
                )
                pending_before = self.playback.pending_samples
                self.playback_queue_peak_samples = max(self.playback_queue_peak_samples, pending_before)
                try:
                    accepted = self.playback.enqueue_packet(
                        response_id=packet.response_id,
                        generation_id=packet.generation_id,
                        speaker_segment_id=packet.speaker_segment_id,
                        packet_id=packet.packet_id,
                        first_frame_id=packet.first_frame_id,
                        last_frame_id=packet.last_frame_id,
                        start_sample=packet.start_sample,
                        end_sample=packet.end_sample,
                        pcm=packet.pcm,
                        sample_rate=packet.sample_rate,
                    )
                except BufferError:
                    self._trace_playback_queue(
                        "enqueue_rejected_capacity",
                        response_id=packet.response_id,
                        generation_id=packet.generation_id,
                        speaker_segment_id=packet.speaker_segment_id,
                        packet_id=packet.packet_id,
                        attempted_samples=int(packet.pcm.size),
                        pending_before_samples=pending_before,
                        capacity_samples=self.playback.max_samples,
                    )
                    raise
                if not accepted:
                    raise EvidenceError("playback FIFO rejected current-generation server PCM")
                pending_after = self.playback.pending_samples
                self.playback_queue_peak_samples = max(self.playback_queue_peak_samples, pending_after)
                self._trace_playback_queue(
                    "enqueue",
                    response_id=packet.response_id,
                    generation_id=packet.generation_id,
                    speaker_segment_id=packet.speaker_segment_id,
                    packet_id=packet.packet_id,
                    start_sample=packet.start_sample,
                    end_sample=packet.end_sample,
                    enqueued_samples=int(packet.pcm.size),
                    pending_before_samples=pending_before,
                    pending_after_samples=pending_after,
                )
            elif kind == "response.audio.segment_done":
                if record is None:
                    raise EvidenceError("response.audio.segment_done arrived without response.created")
                segment_id = event.get("speaker_segment_id")
                accepted = self.playback.complete_segment(
                    response_id=str(response_id),
                    generation_id=_required_protocol_int(event, "generation_id"),
                    speaker_segment_id=str(segment_id or ""),
                    start_sample=_required_protocol_int(event, "start_sample"),
                    end_sample=_required_protocol_int(event, "end_sample"),
                    sample_rate=_required_protocol_int(event, "sample_rate"),
                )
                if not accepted:
                    raise EvidenceError("playback FIFO rejected segment_done identity/range")
                record.marker_ids.add(str(segment_id))
                self._trace_playback_queue(
                    "segment_done",
                    response_id=str(response_id),
                    generation_id=event.get("generation_id"),
                    speaker_segment_id=str(segment_id or ""),
                    start_sample=event.get("start_sample"),
                    end_sample=event.get("end_sample"),
                    pending_samples=self.playback.pending_samples,
                )
            elif kind == "response.done":
                if record is None:
                    raise EvidenceError("response.done arrived without response.created")
                if record.cancelled_event is not None:
                    raise EvidenceError(
                        f"response {record.response_id} emitted response.done after cancellation"
                    )
                record.done_event = event
                record.done_ns = received_ns
                self._record_finish_limits(record.response_id, event)
                if record.marker_ids == record.segment_ids:
                    # All segments were explicitly closed by the canonical
                    # marker before the response terminal; this does not invent
                    # a marker or a consumption ACK.
                    self.playback.complete_response(record.response_id)
                self._trace_playback_queue(
                    "response_done",
                    response_id=record.response_id,
                    generation_id=event.get("generation_id"),
                    segment_ids=sorted(record.segment_ids),
                    segment_done_ids=sorted(record.marker_ids),
                    pending_samples=self.playback.pending_samples,
                )
                record.terminal_signal.set()
            elif kind == "response.cancelled":
                if record is not None:
                    record.cancelled_event = event
                    record.cancelled_ns = received_ns
                    pending_before = self.playback.pending_samples
                    self.playback.cancel_response(
                        record.response_id,
                        event.get("generation_id")
                        if isinstance(event.get("generation_id"), int)
                        and not isinstance(event.get("generation_id"), bool)
                        else None,
                    )
                    record.flush_ns = time.monotonic_ns()
                    pending_after = self.playback.pending_samples
                    self._trace_playback_queue(
                        "response_cancel",
                        response_id=record.response_id,
                        generation_id=event.get("generation_id"),
                        reason=event.get("reason"),
                        pending_before_samples=pending_before,
                        pending_after_samples=pending_after,
                    )
                    self.log.write("harness", "response_cancel_observed", {
                        "response_id": record.response_id,
                        "reason": event.get("reason"),
                        "generation_id": event.get("generation_id"),
                        "cancelled_ns": record.cancelled_ns,
                        "flush_ns": record.flush_ns,
                    })
                    record.terminal_signal.set()
                    self._clear_active_response_if_current(record)
            elif kind == "response.unified_native_finalized" and record is not None:
                record.finalizer_event = event
                record.final_observation_ns = int(received_ns)
                record.finalizer_signal.set()
            if kind == "response.done" and record is not None:
                # Keep the response active until its FIFO samples have drained;
                # the input sender continues independently throughout.
                pass

    async def _ack_sender_loop(self) -> None:
        while not self.stop_event.is_set():
            acknowledgements = self.playback.drain_client_consumed_pcm_acks()
            if not acknowledgements:
                await asyncio.sleep(0.005)
                continue
            for ack in acknowledgements:
                async with self.send_lock:
                    if self.stop_event.is_set():
                        return
                    if not self.playback.accept_client_consumed_pcm_ack(
                        ack, current_generation_id=int(ack["generation_id"])
                    ):
                        self.ack_rejected_count += 1
                        self.log.write("client", "client_consumed_pcm_ack_discarded", ack)
                        continue
                    if self.socket is None:
                        raise EvidenceError("WebSocket closed before client playback ACK could be sent")
                    await self.socket.send(json.dumps(ack, separators=(",", ":")))
                self.ack_sent_count += 1
                self.ack_sent_events.append(dict(ack))
                record = self.responses.get(str(ack.get("response_id")))
                if record is not None:
                    record.ack_sent_ids.add(str(ack.get("speaker_segment_id")))
                self.log.write("client", "client_consumed_pcm", ack)

    async def _playout_loop(self) -> None:
        epoch_ns = time.monotonic_ns()
        self.playback_epoch_ns = epoch_ns
        block_duration_ns = PLAYOUT_BLOCK_SAMPLES * 1_000_000_000 // CODEC_SAMPLE_RATE
        callback_index = 1
        while not self.stop_event.is_set():
            next_deadline_ns = epoch_ns + callback_index * block_duration_ns
            delay_ns = next_deadline_ns - time.monotonic_ns()
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            if self.stop_event.is_set():
                return
            callback_ns = time.monotonic_ns()
            if self.playback_callback_previous_ns is not None:
                gap_ms = (callback_ns - self.playback_callback_previous_ns) / 1e6
                self.playback_callback_gaps_ms.append(gap_ms)
                if gap_ms >= DEFAULT_MAX_GAP_MS:
                    raise EvidenceError(f"simulated 24 kHz playout callback gap reached {gap_ms:.1f}ms")
            lateness_ms = max(0.0, (callback_ns - next_deadline_ns) / 1e6)
            self.playback_callback_max_lateness_ms = max(
                self.playback_callback_max_lateness_ms, lateness_ms
            )
            self.playback_callback_previous_ns = callback_ns
            samples_due = _device_samples_due_since_epoch(
                callback_ns,
                epoch_ns,
                self.playback_total_device_samples,
                sample_rate=CODEC_SAMPLE_RATE,
            )
            pending_before = self.playback.pending_samples
            output = np.zeros(samples_due, dtype=np.float32)
            pcm_samples_consumed = self.playback.read_into(output) if samples_due else 0
            self.playback_total_device_samples += samples_due
            self.playback_actual_pcm_samples_consumed += pcm_samples_consumed
            self.playback_callback_count += 1
            pending_after = self.playback.pending_samples
            self.playback_queue_peak_samples = max(self.playback_queue_peak_samples, pending_before)
            if samples_due:
                self.session_playout_parts.append(output.copy())
            record = self.responses.get(self.current_response_id or "")
            if record is not None and samples_due:
                record.played_pcm.append(output.copy())
                if pcm_samples_consumed > 0:
                    record.pcm_consumed_samples += int(pcm_samples_consumed)
                    record.last_pcm_consumed_ns = int(callback_ns)
                    if record.playout_started_ns is None:
                        record.playout_started_ns = int(callback_ns)
                        self.log.write("harness", "playout_started", {
                            "response_id": record.response_id,
                            "scenario_tag": record.scenario_tag,
                            "playout_started_ns": record.playout_started_ns,
                            "basis": "first PCM sample consumed by the 24 kHz device clock",
                        }, monotonic_ns=callback_ns)
                if record.playout_onset_ns is None and _rms(output) >= INPUT_RMS_MIN:
                    record.playout_onset_ns = callback_ns
                    record.playout_onset_signal.set()
                    self.playback_onsets.append({
                        "response_id": record.response_id,
                        "scenario_tag": record.scenario_tag,
                        "playout_onset_ns": callback_ns,
                        "onset_basis": "24kHz FIFO read callback with frame RMS >= 0.015",
                    })
                    self.log.write("harness", "playout_onset", self.playback_onsets[-1])
            if pending_before or pending_after or pcm_samples_consumed:
                self.playback_queue_trace.append({
                    "event": "consume",
                    "monotonic_ns": callback_ns,
                    "active_response_id": self.current_response_id,
                    "callback_index": self.playback_callback_count,
                    "requested_samples": samples_due,
                    "pcm_samples_consumed": pcm_samples_consumed,
                    "pending_before_samples": pending_before,
                    "pending_after_samples": pending_after,
                })
            callback_index += 1

    async def _task_guard(self, name: str, awaitable: Any) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self.set_fatal(EvidenceError(f"{name}: {type(exc).__name__}: {exc}"))

    async def _wait_queue(self, queue: asyncio.Queue[Any], timeout_s: float) -> Any:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if self.fatal_error:
                raise EvidenceError(self.fatal_error)
            try:
                return await asyncio.wait_for(queue.get(), timeout=min(0.2, max(
                    0.001, deadline - asyncio.get_running_loop().time()
                )))
            except asyncio.TimeoutError:
                continue
        raise EvidenceError(f"scenario event timed out after {timeout_s:.1f}s")

    async def _wait_signal(
        self,
        signal: asyncio.Event,
        *,
        timeout_s: float,
        label: str,
    ) -> None:
        if signal.is_set():
            return
        signal_task = asyncio.create_task(signal.wait())
        fatal_task = asyncio.create_task(self.fatal_event.wait())
        try:
            done, pending = await asyncio.wait(
                (signal_task, fatal_task),
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if signal_task in done and signal.is_set():
                return
            if fatal_task in done and self.fatal_error:
                raise EvidenceError(self.fatal_error)
            raise EvidenceError(f"timed out waiting for {label} after {timeout_s:.1f}s")
        finally:
            for task in (signal_task, fatal_task):
                if not task.done():
                    task.cancel()

    async def _await_response_finalizer(self, record: ResponseRecord) -> None:
        await self._wait_signal(
            record.finalizer_signal,
            timeout_s=min(15.0, self.args.response_timeout_s),
            label=f"response {record.response_id} runtime finalizer",
        )
        status = str((record.finalizer_event or {}).get("status", "")).lower()
        if status != "completed":
            raise EvidenceError(
                f"response {record.response_id} runtime finalizer status was {status or 'missing'}"
            )

    async def _next_speaking_response(
        self,
        timeout_s: float,
        *,
        after_ns: int | None = None,
    ) -> ResponseRecord:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            record = await self._wait_queue(
                self.response_created_queue,
                max(0.001, deadline - asyncio.get_running_loop().time()),
            )
            if after_ns is not None and record.created_ns < after_ns:
                self.log.write("harness", "stale_response_rejected", {
                    "response_id": record.response_id,
                    "action": record.action,
                    "created_ns": record.created_ns,
                    "delivery_started_ns": after_ns,
                })
                continue
            if record.action != "SPEAK":
                self.log.write("harness", "non_speaking_response_skipped", {
                    "response_id": record.response_id,
                    "action": record.action,
                })
                continue
            return record
        raise EvidenceError("no response.created arrived for the real speech fixture")

    async def _await_response_terminal(
        self,
        record: ResponseRecord,
        *,
        want_done: bool,
        timeout_s: float,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if record.cancelled_event is not None:
                if record.done_event is not None:
                    raise EvidenceError(
                        f"response {record.response_id} reported both completion and cancellation"
                    )
                return
            if record.done_event is not None:
                if not want_done:
                    raise EvidenceError(
                        f"response {record.response_id} completed instead of cancelling"
                    )
                return
            if self.fatal_error:
                raise EvidenceError(self.fatal_error)
            try:
                await asyncio.wait_for(
                    record.terminal_signal.wait(),
                    timeout=min(0.2, max(0.001, deadline - asyncio.get_running_loop().time())),
                )
            except asyncio.TimeoutError:
                pass
        raise EvidenceError(f"response {record.response_id} terminal event timed out")

    async def _wait_playback_drain(self, record: ResponseRecord, timeout_s: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if self.fatal_error:
                raise EvidenceError(self.fatal_error)
            if self.playback.pending_samples == 0 and record.marker_ids == record.ack_sent_ids:
                if record.audio_samples and record.playout_completed_ns is None:
                    if record.last_pcm_consumed_ns is None:
                        raise EvidenceError(
                            f"response {record.response_id} drained without a final PCM consumption timestamp"
                        )
                    record.playout_completed_ns = int(record.last_pcm_consumed_ns)
                    self.log.write("harness", "playout_completed", {
                        "response_id": record.response_id,
                        "scenario_tag": record.scenario_tag,
                        "playout_started_ns": record.playout_started_ns,
                        "playout_completed_ns": record.playout_completed_ns,
                        "pcm_samples_consumed": record.pcm_consumed_samples,
                    }, monotonic_ns=record.playout_completed_ns)
                return
            await asyncio.sleep(0.01)
        raise EvidenceError(
            f"response {record.response_id} did not drain through the client-consumed PCM ACK frontier"
        )

    async def _pause_and_drain_input(self, timeout_s: float = 10.0) -> None:
        self.input_pause.set()
        deadline = asyncio.get_running_loop().time() + timeout_s
        while self.input_sent and asyncio.get_running_loop().time() < deadline:
            if self.fatal_error:
                raise EvidenceError(self.fatal_error)
            await asyncio.sleep(0.02)
        if self.input_sent:
            raise EvidenceError(
                f"{len(self.input_sent)} input frames remained unacknowledged after sender pause"
            )
        if len(self.input_ack_ids) != self.input_local_frame_id:
            raise EvidenceError(
                "input ACK count does not match sent frames after sender pause: "
                f"{len(self.input_ack_ids)} != {self.input_local_frame_id}"
            )

    def _assign_fixture(self, fixture_key: str, scenario_tag: str) -> StimulusDelivery:
        fixture = self.fixtures[fixture_key]
        delivery = self.stimuli.schedule(fixture)
        self.log.write("client", "fixture_scheduled", {
            "fixture": fixture_key,
            "scenario_tag": scenario_tag,
            "delivery_id": delivery.delivery_id,
            "delivery_scheduled_ns": delivery.scheduled_ns,
            "transcript_sent_to_server": False,
        })
        return delivery

    async def _complete_fixture(self, fixture_key: str, scenario_tag: str) -> ResponseRecord:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.args.response_timeout_s
        delivery = self._assign_fixture(fixture_key, scenario_tag)
        early_attempts: list[ResponseRecord] = []

        def remaining_budget(label: str) -> float:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                raise EvidenceError(
                    f"fixture {fixture_key} response {label} timed out after the delivery deadline"
                )
            return remaining

        record = await self._next_speaking_response(
            remaining_budget("selection"), after_ns=delivery.scheduled_ns
        )
        attempt_index = 0
        while True:
            attempt_index += 1
            record.delivery_id = delivery.delivery_id
            record.delivery_started_ns = delivery.scheduled_ns
            record.delivery_first_sent_ns = delivery.first_sent_ns
            record.delivery_attempt_index = attempt_index
            record.delivery_attempt_status = "pending"
            record.created_event["source_delivery_id"] = delivery.delivery_id
            record.created_event["delivery_attempt_index"] = attempt_index
            await self._await_response_terminal(
                record, want_done=True, timeout_s=remaining_budget("completion")
            )

            if record.done_event is not None:
                await self._wait_playback_drain(record, self.args.playback_timeout_s)
                await self._await_response_finalizer(record)
                if record.action != "SPEAK":
                    raise EvidenceError(
                        f"fixture {fixture_key} produced action {record.action!r}, expected real Control6 SPEAK"
                    )
                self._validate_normal_done(record)
                record.scenario_tag = scenario_tag
                record.delivery_attempt_status = "completed"
                self.log.write("harness", "scenario_response_assigned", {
                    "response_id": record.response_id,
                    "scenario_tag": scenario_tag,
                    "fixture": fixture_key,
                    "delivery_id": delivery.delivery_id,
                    "delivery_started_ns": delivery.scheduled_ns,
                    "delivery_first_sent_ns": delivery.first_sent_ns,
                    "delivery_attempt_index": attempt_index,
                    "action": record.action,
                })
                for early_record in early_attempts:
                    early_record.recovered_by_response_id = record.response_id
                    self.log.write("harness", "delivery_attempt_recovered", {
                        "response_id": early_record.response_id,
                        "recovered_by_response_id": record.response_id,
                        "delivery_id": delivery.delivery_id,
                        "scenario_tag": scenario_tag,
                    })
                self._clear_active_response_if_current(record)
                for early_record in early_attempts:
                    await self._await_response_finalizer(early_record)
                return record

            if record.cancelled_event is None:
                raise EvidenceError(
                    f"response {record.response_id} returned without a completion or cancellation"
                )
            cancel_reason = str(record.cancelled_event.get("reason", "")).lower()
            prompt_speech_active = bool(
                delivery.first_vocal_sent_ns is not None
                and delivery.frames_sent <= delivery.fixture.vocal_end_frame
            )
            if cancel_reason != "yield" or not prompt_speech_active:
                raise EvidenceError(
                    f"response {record.response_id} cancelled unexpectedly: "
                    f"reason={cancel_reason or 'missing'}, "
                    f"prompt_speech_active={prompt_speech_active}"
                )

            record.scenario_tag = "long_response_early_interruption_attempt"
            record.delivery_attempt_status = "early_interruption"
            record.delivery_first_sent_ns = delivery.first_sent_ns
            record.prompt_speech_active_at_cancel = prompt_speech_active
            record.audible_overlap_with_prompt_speech = bool(
                record.playout_started_ns is not None
                and record.cancelled_ns is not None
                and record.playout_started_ns <= record.cancelled_ns
                and prompt_speech_active
            )
            self.log.write("harness", "scenario_response_assigned", {
                "response_id": record.response_id,
                "scenario_tag": record.scenario_tag,
                "fixture": fixture_key,
                "delivery_id": delivery.delivery_id,
                "delivery_started_ns": delivery.scheduled_ns,
                "delivery_first_sent_ns": delivery.first_sent_ns,
                "delivery_attempt_index": attempt_index,
                "action": record.action,
            })
            self.log.write("harness", "delivery_early_interruption_attempt", {
                "response_id": record.response_id,
                "scenario_tag": record.scenario_tag,
                "delivery_id": delivery.delivery_id,
                "delivery_started_ns": delivery.scheduled_ns,
                "delivery_first_sent_ns": delivery.first_sent_ns,
                "delivery_attempt_index": attempt_index,
                "cancel_reason": cancel_reason,
                "cancelled_ns": record.cancelled_ns,
                "prompt_speech_active": prompt_speech_active,
                "prompt_frames_sent": delivery.frames_sent,
                "prompt_frames_total": delivery.fixture.frame_count,
                "last_prompt_frame_sent_ns": delivery.last_sent_ns,
                "last_prompt_vocal_frame_sent_ns": delivery.last_vocal_sent_ns,
                "response_playout_started_ns": record.playout_started_ns,
                "audible_overlap_with_prompt_speech": record.audible_overlap_with_prompt_speech,
                "flush_ns": record.flush_ns,
            })
            early_attempts.append(record)
            record = await self._next_speaking_response(
                remaining_budget("retry selection"), after_ns=delivery.scheduled_ns
            )

    @staticmethod
    def _validate_normal_done(record: ResponseRecord) -> None:
        if record.done_event is None:
            raise EvidenceError(f"response {record.response_id} has no response.done")
        status = str(record.done_event.get("status", "completed")).lower()
        if status not in {"completed", "complete", "done", ""}:
            raise EvidenceError(f"response {record.response_id} terminal status is not normal: {status}")
        if record.audio_samples <= 0 or not record.marker_ids:
            raise EvidenceError(f"response {record.response_id} has no marked output PCM")
        if record.marker_ids != record.segment_ids:
            raise EvidenceError(f"response {record.response_id} is missing a segment_done marker")
        if not record.audio_telemetry_complete or record.audio_quality_events <= 0:
            raise EvidenceError(f"response {record.response_id} lacks raw-source audio telemetry")

    async def _run_scenario(self) -> None:
        long_done = await self._complete_fixture("long_request", "long_response_completed")
        if long_done.created_event.get("response_profile") != getattr(
            self.args, "response_profile", "long"
        ):
            raise EvidenceError("long response was not created with the explicitly selected profile")
        if (
            long_done.audio_samples < LONG_RESPONSE_MIN_SAMPLES
            or long_done.vocal_frames < LONG_RESPONSE_MIN_VOCAL_FRAMES
        ):
            raise EvidenceError(
                "completed long response failed length gate: "
                f"samples={long_done.audio_samples} (<{LONG_RESPONSE_MIN_SAMPLES}) or "
                f"vocal_frames={long_done.vocal_frames} (<{LONG_RESPONSE_MIN_VOCAL_FRAMES})"
            )

        if long_done.done_event is None or long_done.done_event.get("finish_reason") not in {"stop", "length"}:
            raise EvidenceError(
                "long response must close with natural EOS or an explicit length finish_reason"
            )
        if any(value is None for value in (
            long_done.first_audio_ns,
            long_done.done_ns,
            long_done.playout_started_ns,
            long_done.playout_completed_ns,
            long_done.final_observation_ns,
        )):
            raise EvidenceError("long response lacks audio, playout, done, or finalizer timestamps")
        window_send_ids = {
            int(item["local_frame_id"])
            for item in self.input_frame_sends
            if isinstance(item.get("local_frame_id"), int)
            and not isinstance(item.get("local_frame_id"), bool)
            and isinstance(item.get("sent_ns"), int)
            and not isinstance(item.get("sent_ns"), bool)
            and int(long_done.playout_started_ns)
            <= int(item["sent_ns"])
            <= int(long_done.playout_completed_ns)
        }
        deadline_ack = int(long_done.playout_completed_ns) + 2_000_000_000
        while time.monotonic_ns() < deadline_ack:
            if (
                window_send_ids.issubset(self.input_ack_ids)
                or self.fatal_error
                or self.stop_event.is_set()
            ):
                break
            await asyncio.sleep(0.02)
        input_ack_cutoff_ns = min(time.monotonic_ns(), deadline_ack)
        long_done.long_response_input_window = {}
        long_done.long_response_input_window = _long_response_input_window_metrics(
            pcm_samples=long_done.audio_samples,
            first_audio_ns=int(long_done.first_audio_ns),
            done_ns=long_done.done_ns,
            playout_started_ns=int(long_done.playout_started_ns),
            playout_completed_ns=int(long_done.playout_completed_ns),
            final_observation_ns=int(long_done.final_observation_ns),
            input_ack_cutoff_ns=int(input_ack_cutoff_ns),
            input_frame_sends=self.input_frame_sends,
            input_frame_acks=self.input_frame_acks,
            fifo_peak_samples=self.playback_queue_peak_samples,
            metrics_sink=long_done.long_response_input_window,
        )

        # The user follow-up is the next vocal fixture after the long answer.
        memory_followup = await self._complete_fixture("memory_followup", "memory_followup_resume")

        interrupt_delivery = self._assign_fixture("long_request", "long_response_interrupted")
        interrupted = await self._next_speaking_response(self.args.response_timeout_s)
        interrupted.scenario_tag = "long_response_interrupted"
        interrupted.created_event["source_delivery_id"] = interrupt_delivery.delivery_id
        self.log.write("harness", "scenario_response_assigned", {
            "response_id": interrupted.response_id,
            "scenario_tag": "long_response_interrupted",
            "fixture": "long_request",
            "delivery_id": interrupt_delivery.delivery_id,
            "action": interrupted.action,
        })
        if interrupted.action != "SPEAK":
            raise EvidenceError(
                f"dedicated interruption response had action {interrupted.action!r}, expected SPEAK"
            )
        await self._wait_signal(
            interrupted.playout_onset_signal,
            timeout_s=self.args.playback_timeout_s,
            label=f"response {interrupted.response_id} actual FIFO playout onset",
        )
        assert interrupted.playout_onset_ns is not None
        onset_ns = int(interrupted.playout_onset_ns)
        target_ns = onset_ns + int(self.args.interrupt_after_playout_s * 1e9)
        while time.monotonic_ns() < target_ns:
            if self.fatal_error:
                raise EvidenceError(self.fatal_error)
            await asyncio.sleep(min(0.02, max(0.001, (target_ns - time.monotonic_ns()) / 1e9)))
        interrupt_delivery = self._assign_fixture("interruption", "barge_in")
        await self._await_response_terminal(
            interrupted, want_done=False, timeout_s=self.args.cancellation_timeout_s
        )
        onset_frame_ns = interrupt_delivery.first_vocal_sent_ns
        if onset_frame_ns is None:
            raise EvidenceError("interruption fixture vocal onset frame was never sent")
        assert interrupted.cancelled_ns is not None and interrupted.flush_ns is not None
        cancel_ms = (interrupted.cancelled_ns - onset_frame_ns) / 1e6
        flush_ms = (interrupted.flush_ns - onset_frame_ns) / 1e6
        interrupted.created_event["barge_in_metrics"] = {
            "long_response_playout_onset_ns": onset_ns,
            "interruption_first_frame_sent_ns": interrupt_delivery.first_sent_ns,
            "interruption_vocal_onset_frame_sent_ns": onset_frame_ns,
            "response_cancelled_received_ns": interrupted.cancelled_ns,
            "client_fifo_flush_ns": interrupted.flush_ns,
            "cancel_latency_from_vocal_frame_ms": cancel_ms,
            "flush_latency_from_vocal_frame_ms": flush_ms,
            "cancel_budget_ms": self.args.cancellation_budget_ms,
            "within_budget": max(cancel_ms, flush_ms) <= self.args.cancellation_budget_ms,
        }
        self.log.write("harness", "barge_in_metrics", {
            "response_id": interrupted.response_id,
            **interrupted.created_event["barge_in_metrics"],
        })
        if max(cancel_ms, flush_ms) > self.args.cancellation_budget_ms:
            raise EvidenceError(
                f"YIELD cancellation/flush exceeded {self.args.cancellation_budget_ms:.0f}ms "
                f"from the real interruption frame ({cancel_ms:.1f}/{flush_ms:.1f}ms)"
            )
        if str((interrupted.cancelled_event or {}).get("reason", "")).lower() != "yield":
            raise EvidenceError("interrupted long response did not terminate with reason='yield'")
        await self._await_response_finalizer(interrupted)

        closure = await self._complete_fixture("closure", "closure")
        if memory_followup.done_event is None or closure.done_event is None:
            raise EvidenceError("same-connection memory/resume/closure responses did not complete")

    def _read_runtime_telemetry(self) -> dict[str, Any] | None:
        telemetry_path = self.args.runtime_telemetry
        if not telemetry_path:
            self.evidence_gaps.append("runtime KV/VRAM telemetry JSON was not supplied")
            return None
        path = Path(telemetry_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            self.evidence_gaps.append("runtime KV/VRAM telemetry JSON is missing or empty")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            self.evidence_gaps.append("runtime KV/VRAM telemetry JSON is invalid")
            return None
        if not isinstance(payload, dict):
            self.evidence_gaps.append("runtime telemetry root is not an object")
            return None
        return {"path": str(path), "sha256": _sha256_bytes(path.read_bytes()), **payload}

    def _read_server_log(self) -> dict[str, Any] | None:
        if not self.args.server_log:
            self.evidence_gaps.append("local server log was not supplied")
            return None
        path = Path(self.args.server_log).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            self.evidence_gaps.append("local server log is missing or empty")
            return None
        raw = path.read_bytes()
        content = raw.decode("utf-8", errors="replace")
        capture_path_value = getattr(self.args, "server_log_window", None)
        capture: dict[str, Any] | None = None
        capture_errors: list[str] = []
        if not capture_path_value:
            capture_errors.append("server log capture-window metadata was not supplied")
        else:
            capture_path = Path(capture_path_value).expanduser().resolve()
            if not capture_path.is_file() or capture_path.stat().st_size <= 0:
                capture_errors.append("server log capture-window metadata is missing or empty")
            else:
                try:
                    capture = json.loads(capture_path.read_text(encoding="utf-8-sig"))
                except (OSError, json.JSONDecodeError):
                    capture_errors.append("server log capture-window metadata is invalid JSON")
                    capture = None
                if not isinstance(capture, dict):
                    capture_errors.append("server log capture-window metadata is not an object")
                    capture = None

        session_id = str((self.session_created or {}).get("session_id", ""))
        response_ids = set(self.responses)
        window_valid = False
        query_bounds: dict[str, str] = {}
        if capture is not None:
            if capture.get("schema") != "duplex_server_log_window_v1":
                capture_errors.append("server log window schema is not supported")
            if capture.get("session_id") != session_id or not session_id:
                capture_errors.append("server log window session_id does not match this WebSocket")
            since_value = _parse_utc_datetime(capture.get("since_utc"))
            until_value = _parse_utc_datetime(capture.get("until_utc"))
            query_bounds = {
                "since_utc": str(capture.get("since_utc", "")),
                "until_utc": str(capture.get("until_utc", "")),
            }
            if since_value is None or until_value is None or until_value < since_value:
                capture_errors.append("server log UTC since/until bounds are invalid")
            elif self.run_started_at_utc is None or self.run_finished_at_utc is None:
                capture_errors.append("simulator run UTC bounds are unavailable")
            else:
                # The collector may round the query to the nearest second, but
                # it must not pull an unbounded historical log into this run.
                slack = timedelta(seconds=1)
                if (
                    since_value > self.run_started_at_utc
                    or until_value < self.run_finished_at_utc
                    or since_value < self.run_started_at_utc - slack
                    or until_value > self.run_finished_at_utc + slack
                ):
                    capture_errors.append("server log capture bounds do not isolate this simulator run")
            window_valid = not capture_errors

        if capture is not None and window_valid:
            since_value = _parse_utc_datetime(capture.get("since_utc"))
            until_value = _parse_utc_datetime(capture.get("until_utc"))
            scoped_content, timestamped_records, out_of_window_records = _scope_server_log_content(
                content,
                since=since_value,
                until=until_value,
                run_started=self.run_started_at_utc,
                run_finished=self.run_finished_at_utc,
            )
            if out_of_window_records:
                capture_errors.append(
                    f"server log contains {out_of_window_records} timestamped record(s) outside the run window"
                )
                window_valid = False
        else:
            scoped_content = ""
            timestamped_records = 0
            out_of_window_records = 0

        # A window-scoped collector may emit CUDA lines or multi-line Python
        # tracebacks without a session ID on every line. Their identity comes
        # from this run's UTC bounds; the raw window must still contain a
        # concrete session/response marker from this WebSocket run.
        identity_content = scoped_content if timestamped_records else content
        identity_markers = [item for item in response_ids if item and item in identity_content]
        if session_id and session_id in identity_content:
            identity_markers.append(session_id)
        session_prefix = session_id[:8] if len(session_id) >= 8 else ""
        if session_prefix and session_prefix in identity_content:
            identity_markers.append(session_prefix)
        correlated = window_valid and bool(identity_markers)
        if not correlated:
            capture_errors.append("server log cannot be tied to this run's bounded process/session window")
        if not correlated:
            scoped_content = ""
        has_kv = bool(re.search(
            r"exclusive_kv_cursor|cache_position|\bkv(?:_cache)?\b", scoped_content, re.I
        ))
        has_cuda_memory = bool(re.search(
            r"memory_allocated|memory_reserved|peak_memory|cuda_peak|\bvram\b|nvidia-smi",
            scoped_content,
            re.I,
        ))
        exception_markers = [
            line.strip() for line in scoped_content.splitlines()
            if re.search(
                r"Traceback \(most recent call last\)|\b(?:RuntimeError|AssertionError|ValueError|TypeError|Exception)\b|"
                r"\b(?:ERROR|CRITICAL)\b|CUDA error|device-side assert|out of memory",
                line,
                re.I,
            )
        ]
        if exception_markers:
            self.evidence_gaps.append(
                f"run-scoped local server log contains {len(exception_markers)} exception/error marker(s)"
            )
        self.evidence_gaps.extend(capture_errors)
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "contains_kv_telemetry": has_kv,
            "contains_cuda_memory_telemetry": has_cuda_memory,
            "exception_markers": exception_markers,
            "correlated_to_this_run": correlated,
            "capture_window_valid": window_valid,
            "source": None if capture is None else capture.get("source"),
            "query_bounds_utc": query_bounds,
            "identity_markers": sorted(set(identity_markers)),
            "capture_errors": capture_errors,
            "timestamped_records": timestamped_records,
            "out_of_window_records": out_of_window_records,
        }

    def _assess_telemetry(self, runtime: dict[str, Any] | None) -> dict[str, Any]:
        if not self.runtime_kv_evidence:
            self.evidence_gaps.append("no runtime KV cursor/rollback evidence arrived over the WebSocket")
        if not self.runtime_audio_telemetry_complete:
            self.evidence_gaps.append("one or more audio packets lacked raw-source quality telemetry")
        if runtime is None:
            return {"status": "INCOMPLETE"}
        expected_session_id = str((self.session_created or {}).get("session_id", ""))
        if not expected_session_id or runtime.get("session_id") != expected_session_id:
            self.evidence_gaps.append("runtime sidecar is missing this WebSocket session_id")
            return {"status": "INCOMPLETE", "session_id": runtime.get("session_id")}
        allocator = runtime.get("cuda_allocator")
        allocator_values = (
            allocator.get("allocated_bytes"),
            allocator.get("reserved_bytes"),
            allocator.get("peak_allocated_bytes"),
            allocator.get("peak_reserved_bytes"),
            allocator.get("sampled_at_monotonic_ns"),
        ) if isinstance(allocator, Mapping) else ()
        allocator_valid = (
            isinstance(allocator, Mapping)
            and allocator.get("status") in {"ok", "available", "measured"}
            and isinstance(allocator.get("device"), str)
            and allocator["device"].lower().startswith("cuda")
            and all(
                not isinstance(value, bool)
                and isinstance(value, int)
                and value >= 0
                for value in allocator_values
            )
            and allocator["sampled_at_monotonic_ns"] > 0
            and allocator["reserved_bytes"] >= allocator["allocated_bytes"]
            and allocator["peak_allocated_bytes"] >= allocator["allocated_bytes"]
            and allocator["peak_reserved_bytes"] >= allocator["reserved_bytes"]
        )
        window = runtime.get("run_window_utc")
        if not isinstance(window, Mapping):
            window = runtime.get("capture_window")
        window_matches = (
            isinstance(window, Mapping)
            and _parse_utc_datetime(window.get("started_at", window.get("since_utc"))) is not None
            and _parse_utc_datetime(window.get("finished_at", window.get("until_utc"))) is not None
            and self.run_started_at_utc is not None
            and self.run_finished_at_utc is not None
            and abs((_parse_utc_datetime(window.get("started_at", window.get("since_utc"))) - self.run_started_at_utc).total_seconds()) <= 1.0
            and abs((_parse_utc_datetime(window.get("finished_at", window.get("until_utc"))) - self.run_finished_at_utc).total_seconds()) <= 1.0
        )
        if not allocator_valid:
            self.evidence_gaps.append(
                "runtime sidecar lacks measured CUDA device allocator counters correlated to this run"
            )
        if not window_matches:
            self.evidence_gaps.append("runtime allocator sidecar run window does not match the simulator run")
        vram = runtime.get("vram")
        kv = runtime.get("kv")
        def finite_nonnegative(value: Any) -> bool:
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0
            )

        baseline = vram.get("post_warmup_baseline") if isinstance(vram, Mapping) else None
        active = vram.get("active") if isinstance(vram, Mapping) else None
        peak = vram.get("peak") if isinstance(vram, Mapping) else None
        turns = vram.get("turns") if isinstance(vram, Mapping) else None
        top_level_measured = all(
            isinstance(group, Mapping)
            and all(finite_nonnegative(group.get(metric)) for metric in ("allocated_mib", "reserved_mib"))
            for group in (baseline, active, peak)
        )
        turn_values: list[dict[str, Any]] = []
        expected_response_ids = {
            record.scenario_tag: record.response_id
            for record in self.responses.values()
            if record.scenario_tag in (
                "long_response_completed",
                "memory_followup_resume",
                "long_response_interrupted",
                "closure",
            )
        }
        if isinstance(turns, list):
            for row in turns:
                post_drain = row.get("post_drain") if isinstance(row, Mapping) else None
                tag = row.get("scenario_tag") if isinstance(row, Mapping) else None
                response_id = row.get("response_id") if isinstance(row, Mapping) else None
                if (
                    not isinstance(tag, str)
                    or not isinstance(response_id, str)
                    or expected_response_ids.get(tag) != response_id
                    or not isinstance(post_drain, Mapping)
                    or not all(finite_nonnegative(post_drain.get(metric)) for metric in ("allocated_mib", "reserved_mib"))
                ):
                    continue
                turn_values.append({
                    "scenario_tag": tag,
                    "response_id": response_id,
                    "allocated_mib": float(post_drain["allocated_mib"]),
                    "reserved_mib": float(post_drain["reserved_mib"]),
                })
        expected_tag_order = tuple(expected_response_ids)
        expected_tags = set(expected_tag_order)
        observed_tags = {row["scenario_tag"] for row in turn_values}
        if (
            not allocator_valid
            or not window_matches
            or not top_level_measured
            or not expected_tags.issubset(observed_tags)
            or any(sum(row["scenario_tag"] == tag for row in turn_values) != 1 for tag in expected_tag_order)
        ):
            self.evidence_gaps.append(
                "runtime VRAM telemetry lacks measured allocated/reserved baseline, active, peak, or per-turn post-drain values"
            )
            return {"status": "INCOMPLETE", "vram": vram, "kv": kv}
        if not isinstance(kv, Mapping) or not kv:
            self.evidence_gaps.append("runtime sidecar lacks KV telemetry")
        peak_budget = self.args.vram_peak_over_baseline_budget_mib
        drain_budget = self.args.vram_post_drain_growth_budget_mib
        reserved_peak_budget = self.args.vram_reserved_peak_growth_budget_mib
        reserved_drain_budget = self.args.vram_reserved_post_drain_growth_budget_mib
        if any(value is None for value in (
            peak_budget, drain_budget, reserved_peak_budget, reserved_drain_budget
        )):
            self.evidence_gaps.append("explicit allocated/reserved VRAM budgets were not supplied")
            return {"status": "INCOMPLETE", "vram": dict(vram), "kv": kv}
        if any(
            not finite_nonnegative(value)
            for value in (peak_budget, drain_budget, reserved_peak_budget, reserved_drain_budget)
        ):
            self.evidence_gaps.append("allocated/reserved VRAM budgets must be finite and non-negative")
            return {"status": "INCOMPLETE", "vram": dict(vram), "kv": kv}
        allocated_peak_growth = float(peak["allocated_mib"]) - float(baseline["allocated_mib"])
        reserved_peak_growth = float(peak["reserved_mib"]) - float(baseline["reserved_mib"])
        allocated_drain_growth = max(
            row["allocated_mib"] - float(baseline["allocated_mib"])
            for row in turn_values if row["scenario_tag"] in expected_tags
        )
        reserved_drain_growth = max(
            row["reserved_mib"] - float(baseline["reserved_mib"])
            for row in turn_values if row["scenario_tag"] in expected_tags
        )
        within_budget = (
            allocated_peak_growth <= peak_budget
            and allocated_drain_growth <= drain_budget
            and reserved_peak_growth <= reserved_peak_budget
            and reserved_drain_growth <= reserved_drain_budget
        )
        ordered_drain = [
            next(row for row in turn_values if row["scenario_tag"] == tag)
            for tag in expected_tag_order
        ]
        allocation_deltas = [row["allocated_mib"] - float(baseline["allocated_mib"]) for row in ordered_drain]
        reserved_deltas = [row["reserved_mib"] - float(baseline["reserved_mib"]) for row in ordered_drain]
        return {
            "status": "PASS" if within_budget and isinstance(kv, Mapping) and bool(kv) else "FAIL",
            "vram": dict(vram),
            "kv": _json_safe(kv),
            "allocated_peak_growth_over_post_warmup_mib": allocated_peak_growth,
            "reserved_peak_growth_over_post_warmup_mib": reserved_peak_growth,
            "max_allocated_post_drain_growth_over_post_warmup_mib": allocated_drain_growth,
            "max_reserved_post_drain_growth_over_post_warmup_mib": reserved_drain_growth,
            "allocated_peak_growth_budget_mib": peak_budget,
            "allocated_post_drain_growth_budget_mib": drain_budget,
            "reserved_peak_growth_budget_mib": reserved_peak_budget,
            "reserved_post_drain_growth_budget_mib": reserved_drain_budget,
            "post_drain_allocated_delta_by_turn_mib": allocation_deltas,
            "post_drain_reserved_delta_by_turn_mib": reserved_deltas,
            "trend_interpretation": (
                f"observed per-turn trend only; {len(ordered_drain)} samples do not certify absence of a leak"
            ),
            "within_budget": within_budget,
        }

    def _build_report(self) -> dict[str, Any]:
        normal_done = [
            response for response in self.responses.values()
            if response.done_event is not None and response.cancelled_event is None
        ]
        cancelled = [response for response in self.responses.values() if response.cancelled_event is not None]
        delivery_attempts = [
            response for response in self.responses.values()
            if response.delivery_id is not None
        ]
        early_interruption_attempts = [
            response for response in delivery_attempts
            if response.delivery_attempt_status == "early_interruption"
        ]
        long_complete = [
            response for response in normal_done
            if response.scenario_tag == "long_response_completed"
            and response.audio_samples >= LONG_RESPONSE_MIN_SAMPLES
            and response.vocal_frames >= LONG_RESPONSE_MIN_VOCAL_FRAMES
        ]
        if len(normal_done) < 3:
            self.scenario_errors.append(f"only {len(normal_done)} normally completed responses; need at least 3")
        if not long_complete:
            self.scenario_errors.append("no normal response met the long PCM/vocal-frame gate")
        if self.playback_queue_peak_samples > BROWSER_PLAYBACK_MAX_SAMPLES:
            self.scenario_errors.append(
                f"playback FIFO peak exceeded {BROWSER_PLAYBACK_MAX_SAMPLES} samples"
            )
        if len(cancelled) < 1:
            self.scenario_errors.append("no response was cancelled on the same connection")
        if self.input_dropped_frames:
            self.scenario_errors.append(f"input frame drops detected: {self.input_dropped_frames}")
        if self.ack_rejected_count:
            self.scenario_errors.append(f"client-consumed PCM ACK rejection count: {self.ack_rejected_count}")
        if self.input_inflight_high_water >= MAX_INPUT_INFLIGHT:
            self.scenario_errors.append("input in-flight reached 16")
        if self.ws_errors:
            self.scenario_errors.extend(f"WebSocket/runtime error: {item}" for item in self.ws_errors)
        if not self.runtime_kv_evidence:
            self.evidence_gaps.append("runtime KV telemetry missing")
        if self.session_created and not self._voice_from_created(self.session_created):
            self.evidence_gaps.append("session.created voice missing")
        if self.session_created and self.session_updated and (
            self._voice_from_created(self.session_created) != self.session_updated.get("voice")
        ):
            self.scenario_errors.append("session.updated voice differs from session.created voice")

        runtime = self._read_runtime_telemetry()
        server_log = self._read_server_log()
        telemetry_assessment = self._assess_telemetry(runtime)
        if self.evidence_gaps:
            self.outcome = "INCOMPLETE"
        elif self.scenario_errors or telemetry_assessment.get("status") != "PASS":
            self.outcome = "FAIL"
        else:
            self.outcome = "PASS"
        return {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_window_utc": {
                "started_at": None if self.run_started_at_utc is None else self.run_started_at_utc.isoformat(),
                "finished_at": None if self.run_finished_at_utc is None else self.run_finished_at_utc.isoformat(),
            },
            "acceptance_parameters": {
                "mode": self.args.mode,
                "response_profile": getattr(self.args, "response_profile", "long"),
                "noise_rms": float(self.args.noise_rms),
                "seed": int(self.args.seed),
                "pacing_tolerance_ms": float(self.args.pacing_tolerance_ms),
                "max_input_gap_ms": float(self.args.max_input_gap_ms),
                "input_inflight_limit": MAX_INPUT_INFLIGHT,
                "playback_buffer_seconds": self.playback_buffer_seconds,
                "playback_buffer_max_samples": self.playback_capacity_samples,
                "playback_buffer_contract": "server/static/index.html MAX_PLAYBACK_AHEAD_SECONDS",
                "interrupt_after_playout_s": float(self.args.interrupt_after_playout_s),
                "cancellation_budget_ms": float(self.args.cancellation_budget_ms),
                "long_min_audio_samples": LONG_RESPONSE_MIN_SAMPLES,
                "long_min_vocal_frames": LONG_RESPONSE_MIN_VOCAL_FRAMES,
                "vram_peak_over_baseline_budget_mib": self.args.vram_peak_over_baseline_budget_mib,
                "vram_post_drain_growth_budget_mib": self.args.vram_post_drain_growth_budget_mib,
                "vram_reserved_peak_growth_budget_mib": self.args.vram_reserved_peak_growth_budget_mib,
                "vram_reserved_post_drain_growth_budget_mib": self.args.vram_reserved_post_drain_growth_budget_mib,
            },
            "overall_status": self.outcome,
            "connection": {
                "mode": self.args.mode,
                "acceptance_interruption_phrase": "Aspetta fermati un momento",
                "url": self.args.url,
                "tls": self.tls_evidence,
                "session_id": (self.session_created or {}).get("session_id"),
                "turn_policy_created": (self.session_created or {}).get("turn_policy"),
                "turn_policy_updated": (self.session_updated or {}).get("turn_policy"),
                "voice_created": self._voice_from_created(self.session_created or {}),
                "voice_updated": (self.session_updated or {}).get("voice"),
                "voice_conditioning_mode_created": (self.session_created or {}).get("voice_conditioning_mode"),
                "voice_conditioning_mode_updated": (self.session_updated or {}).get("voice_conditioning_mode"),
                "voice_override_sent": False,
                "response_profile_updated": (self.session_updated or {}).get("response_profile"),
                "transcript_injection": False,
                "response_create_events_sent": 0,
            },
            "error_categories": {
                key: list(value) for key, value in self.error_categories.items()
            },
            "fixture_manifest": self.input_manifest,
            "fixtures": {key: fixture.metrics for key, fixture in self.fixtures.items()},
            "scenario": {
                "normal_response_count": len(normal_done),
                "cancelled_response_count": len(cancelled),
                "long_completed_response_count": len(long_complete),
                "required_normal_responses": 3,
                "long_min_audio_samples": LONG_RESPONSE_MIN_SAMPLES,
                "long_min_vocal_frames": LONG_RESPONSE_MIN_VOCAL_FRAMES,
                "long_response_input_window": next((
                    response.long_response_input_window
                    for response in self.responses.values()
                    if response.scenario_tag == "long_response_completed"
                ), None),
                "delivery_attempts": [
                    {
                        "response_id": response.response_id,
                        "delivery_id": response.delivery_id,
                        "delivery_started_ns": response.delivery_started_ns,
                        "delivery_first_sent_ns": response.delivery_first_sent_ns,
                        "attempt_index": response.delivery_attempt_index,
                        "status": response.delivery_attempt_status,
                        "scenario_tag": response.scenario_tag,
                        "action": response.action,
                        "cancel_reason": (response.cancelled_event or {}).get("reason"),
                        "prompt_speech_active_at_cancel": response.prompt_speech_active_at_cancel,
                        "audible_overlap_with_prompt_speech": response.audible_overlap_with_prompt_speech,
                        "recovered_by_response_id": response.recovered_by_response_id,
                    }
                    for response in delivery_attempts
                ],
                "early_interruption_attempt_count": len(early_interruption_attempts),
                "long_response_recovered_after_early_interruption": bool(
                    early_interruption_attempts
                    and any(response.scenario_tag == "long_response_completed" for response in delivery_attempts)
                    and all(response.recovered_by_response_id for response in early_interruption_attempts)
                ),
                "required_response_order": [
                    "long_response_completed",
                    "memory_followup_resume",
                    "long_response_interrupted",
                    "closure",
                ],
                "long_responses": [self._response_report(item) for item in long_complete],
                "responses": [self._response_report(item) for item in self.responses.values()],
                "errors": self.scenario_errors,
                "evidence_gaps": self.evidence_gaps,
            },
            "input": {
                "sample_rate": MIC_SAMPLE_RATE,
                "channels": 1,
                "frame_samples": MIC_FRAME_SAMPLES,
                "frame_ms": FRAME_MS,
                "frames_sent": self.input_local_frame_id,
                "frames_acked": len(self.input_ack_ids),
                "frames_inflight_at_end": len(self.input_sent),
                "frames_dropped": self.input_dropped_frames,
                "inflight_high_water": self.input_inflight_high_water,
                "inflight_fail_threshold": MAX_INPUT_INFLIGHT,
                "server_capacity": self.session_created.get("input_audio_buffer_capacity_frames") if self.session_created else None,
                "ack_latency_p95_ms": float(np.percentile(self.input_ack_latency_ms, 95)) if self.input_ack_latency_ms else None,
                "pacing": self.pacing.report(),
                "sender_timing": {
                    "frames_waiting_for_backpressure": sum(
                        bool(item.get("waited_for_backpressure"))
                        for item in self.input_frame_sends
                    ),
                    "total_backpressure_wait_ms": sum(
                        int(item.get("backpressure_wait_ns", 0))
                        for item in self.input_frame_sends
                    ) / 1e6,
                    "max_backpressure_wait_ms": max((
                        int(item.get("backpressure_wait_ns", 0)) / 1e6
                        for item in self.input_frame_sends
                    ), default=0.0),
                    "max_sender_scheduler_delay_ms": max((
                        int(item.get("sender_scheduler_delay_ns", 0)) / 1e6
                        for item in self.input_frame_sends
                    ), default=0.0),
                    "max_gap_ms": DEFAULT_MAX_GAP_MS,
                    "backpressure_and_scheduler_delay_are_separate": True,
                },
            },
            "playout": {
                "sample_rate": CODEC_SAMPLE_RATE,
                "fifo_max_seconds": self.playback_buffer_seconds,
                "fifo_max_samples": self.playback_capacity_samples,
                "fifo_limit_source": "server/static/index.html MAX_PLAYBACK_AHEAD_SECONDS",
                "block_samples": PLAYOUT_BLOCK_SAMPLES,
                "callback_max_lateness_ms": self.playback_callback_max_lateness_ms,
                "callback_max_gap_ms": max(self.playback_callback_gaps_ms) if self.playback_callback_gaps_ms else None,
                "callback_count": self.playback_callback_count,
                "device_samples_elapsed": self.playback_total_device_samples,
                "queued_pcm_samples_consumed": self.playback_actual_pcm_samples_consumed,
                "effective_device_sample_rate_hz": (
                    self.playback_total_device_samples * 1_000_000_000
                    / max(1, self.playback_callback_previous_ns - self.playback_epoch_ns)
                    if self.playback_epoch_ns is not None and self.playback_callback_previous_ns is not None
                    else None
                ),
                "fifo_peak_pending_samples": self.playback_queue_peak_samples,
                "fifo_peak_pending_seconds": self.playback_queue_peak_samples / CODEC_SAMPLE_RATE,
                "fifo_trace_path": str(self.output_dir / "playback_queue_trace.jsonl"),
                "fifo_trace_events": len(self.playback_queue_trace),
                "onsets": self.playback_onsets,
                "client_consumed_pcm_acks_sent": self.ack_sent_count,
                "client_consumed_pcm_acks_rejected": self.ack_rejected_count,
                "ack_events": self.ack_sent_events,
            },
            "runtime_evidence": {
                "server_log": server_log,
                "telemetry": telemetry_assessment,
                "kv_events": self.runtime_kv_evidence,
                "raw_source_audio_telemetry_complete": self.runtime_audio_telemetry_complete,
                "sources": {
                    "server_log": self.args.server_log,
                    "server_log_window": self.args.server_log_window,
                    "runtime_telemetry": self.args.runtime_telemetry,
                },
            },
            "errors": self.ws_errors,
            "event_log": {"path": str(self.output_dir / "events.jsonl"), "rows": self.log.count},
        }

    @staticmethod
    def _response_report(record: ResponseRecord) -> dict[str, Any]:
        runtime_metrics = (record.done_event or {}).get("runtime_metrics")
        response_completion = (
            runtime_metrics.get("response_completion")
            if isinstance(runtime_metrics, Mapping) else None
        )
        return {
            "response_id": record.response_id,
            "scenario_tag": record.scenario_tag,
            "delivery_id": record.delivery_id,
            "delivery_started_ns": record.delivery_started_ns,
            "delivery_first_sent_ns": record.delivery_first_sent_ns,
            "delivery_attempt_index": record.delivery_attempt_index,
            "delivery_attempt_status": record.delivery_attempt_status,
            "prompt_speech_active_at_cancel": record.prompt_speech_active_at_cancel,
            "audible_overlap_with_prompt_speech": record.audible_overlap_with_prompt_speech,
            "recovered_by_response_id": record.recovered_by_response_id,
            "action": record.action,
            "response_profile": record.created_event.get("response_profile"),
            "text": "".join(record.text_parts),
            "done_status": (record.done_event or {}).get("status"),
            "finish_reason": (record.done_event or {}).get("finish_reason"),
            "cancel_reason": (record.cancelled_event or {}).get("reason"),
            "audio_samples": record.audio_samples,
            "audio_seconds": record.audio_samples / CODEC_SAMPLE_RATE,
            "codec_frames": record.codec_frames,
            "vocal_frames_rms_ge_0_015": record.vocal_frames,
            "segment_ids": sorted(record.segment_ids),
            "segment_done_ids": sorted(record.marker_ids),
            "client_consumed_ack_ids": sorted(record.ack_sent_ids),
            "first_audio_ns": record.first_audio_ns,
            "pcm_duration_seconds": record.audio_samples / CODEC_SAMPLE_RATE,
            "production_window_seconds": (
                (record.done_ns - record.first_audio_ns) / 1e9
                if record.done_ns is not None and record.first_audio_ns is not None else None
            ),
            "playout_started_ns": record.playout_started_ns,
            "playout_completed_ns": record.playout_completed_ns,
            "playout_window_seconds": (
                (record.playout_completed_ns - record.playout_started_ns) / 1e9
                if record.playout_completed_ns is not None and record.playout_started_ns is not None
                else None
            ),
            "pcm_samples_consumed": record.pcm_consumed_samples,
            "playout_onset_ns": record.playout_onset_ns,
            "done_ns": record.done_ns,
            "final_observation_ns": record.final_observation_ns,
            "cancelled_ns": record.cancelled_ns,
            "fifo_flush_ns": record.flush_ns,
            "barge_in_metrics": record.created_event.get("barge_in_metrics"),
            "audio_telemetry_complete": record.audio_telemetry_complete,
            "audio_quality_packet_count": record.audio_quality_events,
            "finalizer_event": record.finalizer_event,
            "response_completion": response_completion,
            "long_response_input_window": record.long_response_input_window,
        }

    def _write_wav(self, path: Path, pcm: np.ndarray) -> None:
        samples = np.asarray(pcm, dtype=np.float64).reshape(-1)
        if not np.isfinite(samples).all():
            raise EvidenceError(f"refusing to write non-finite WAV: {path.name}")
        scaled = np.rint(samples * 32768.0)
        rail_count = int(np.count_nonzero((scaled < -32768) | (scaled > 32767)))
        if rail_count:
            raise EvidenceError(f"WAV output would clip before quantization: {path.name}")
        raw = scaled.clip(-32768, 32767).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(CODEC_SAMPLE_RATE if "playout" in path.name or "response_" in path.name else MIC_SAMPLE_RATE)
            target.writeframes(raw)

    def _write_outputs(self, report: dict[str, Any]) -> None:
        self.log.close()
        with (self.output_dir / "playback_queue_trace.jsonl").open("w", encoding="utf-8", newline="\n") as trace_file:
            for trace_row in self.playback_queue_trace:
                trace_file.write(json.dumps(_json_safe(trace_row), ensure_ascii=False, separators=(",", ":")) + "\n")
        if self.input_wav_bytes:
            with wave.open(str(self.output_dir / "input_stream.wav"), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(MIC_SAMPLE_RATE)
                target.writeframes(bytes(self.input_wav_bytes))
        if self.session_playout_parts:
            self._write_wav(
                self.output_dir / "session_playout.wav",
                np.concatenate(self.session_playout_parts),
            )
        for record in self.responses.values():
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", record.response_id)
            if record.received_pcm:
                self._write_wav(
                    self.output_dir / f"response_{safe_id}_received.wav",
                    np.concatenate(record.received_pcm),
                )
            if record.played_pcm:
                self._write_wav(
                    self.output_dir / f"response_{safe_id}_playout.wav",
                    np.concatenate(record.played_pcm),
                )
        (self.output_dir / "fixture_manifest.json").write_text(
            json.dumps(_json_safe(self.input_manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "report.json").write_text(
            json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    async def run(self) -> dict[str, Any]:
        self.run_started_at_utc = datetime.now(timezone.utc)
        self.log.write("harness", "run_started", {
            "started_at_utc": self.run_started_at_utc.isoformat(),
        })
        for key, fixture in self.fixtures.items():
            if not fixture.spec.transcript_verified:
                self.evidence_gaps.append(f"fixture {key} ASR transcript is not marked verified")
        if self.args.mode != "acceptance":
            self.evidence_gaps.append(
                "diagnostic mode cannot certify the accepted interruption phrase or E2E acceptance"
            )
        if self.args.mode == "acceptance":
            interrupt_text = re.sub(
                r"\W+", " ", self.fixtures["interruption"].spec.transcript.casefold()
            ).strip()
            if "aspetta fermati un momento" not in interrupt_text:
                self.evidence_gaps.append(
                    "acceptance mode requires an ASR-verified 'Aspetta, fermati un momento' fixture"
                )
        try:
            tls_options = self._websocket_tls_options()
            async with websockets.connect(
                self.args.url,
                proxy=None,
                max_size=32 * 1024 * 1024,
                max_queue=8,
                write_limit=32 * 1024,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=2,
                **tls_options,
            ) as socket:
                self.socket = socket
                await self._wait_ready()
                self.tasks = [
                    asyncio.create_task(self._task_guard("input sender", self._sender_loop())),
                    asyncio.create_task(self._task_guard("WebSocket receiver", self._receiver_loop())),
                    asyncio.create_task(self._task_guard("playback ACK sender", self._ack_sender_loop())),
                    asyncio.create_task(self._task_guard("24 kHz playout", self._playout_loop())),
                ]
                try:
                    await self._run_scenario()
                    await self._pause_and_drain_input()
                    if self.playback.pending_samples != 0:
                        raise EvidenceError(
                            f"playback FIFO still held {self.playback.pending_samples} samples at scenario end"
                        )
                    await asyncio.sleep(1.5)
                finally:
                    self.stop_event.set()
                    for task in self.tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*self.tasks, return_exceptions=True)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                interruption = f"{type(exc).__name__}: harness interrupted"
                if self.fatal_error is None:
                    self.fatal_error = interruption
                    self.ws_errors.append(interruption)
                    self.fatal_event.set()
                    self.stop_event.set()
                    self.log.write("harness", "fatal_error", {
                        "message": interruption,
                        "category": "harness_interruption",
                    })
                self.error_categories["harness_interruption"].append(interruption)
            else:
                self.set_fatal(exc)
        if self.fatal_error is None:
            # Counts and terminal events are snapshotted only after all async
            # producer/receiver/playout tasks have stopped and been gathered.
            self.log.write("harness", "scenario_completed", {
                "responses": len(self.responses),
                "input_frames_sent": self.input_local_frame_id,
                "input_frames_acked": len(self.input_ack_ids),
                "playback_pending_samples": self.playback.pending_samples,
                "async_tasks_stopped": all(task.done() for task in self.tasks),
            })
        self.run_finished_at_utc = datetime.now(timezone.utc)
        self.log.write("harness", "run_finished", {
            "finished_at_utc": self.run_finished_at_utc.isoformat(),
            "fatal_error": self.fatal_error,
        })
        report = self._build_report()
        self._write_outputs(report)
        return report


def _load_saved_event_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return [], [f"raw event log cannot be read: {exc}"]
    if not lines:
        return [], ["raw event log is empty"]
    for line_no, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"raw event log line {line_no} is invalid JSON")
            continue
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("utc"), str)
            or not isinstance(row.get("type"), str)
            or not isinstance(row.get("payload"), Mapping)
        ):
            errors.append(f"raw event log line {line_no} is missing its event envelope")
            continue
        if _parse_utc_datetime(row.get("utc")) is None:
            errors.append(f"raw event log line {line_no} has an invalid UTC timestamp")
        rows.append(row)
    return rows, errors


def _extract_kv_evidence_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = {
        "client_consumed_pcm", "unified_native_rollback", "unified_native_finalizer",
        "confirmed_exclusive_kv_cursor", "spoken_prefix_exclusive_cursor",
    }
    found_rows: list[dict[str, Any]] = []
    for row in rows:
        event = row.get("payload")
        if not isinstance(event, Mapping):
            continue
        metrics = event.get("runtime_metrics")
        if not isinstance(metrics, Mapping):
            continue
        found: dict[str, Any] = {}

        def visit(node: Any) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    if key in keys and value is not None:
                        found[str(key)] = value
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(metrics)
        if found:
            found_rows.append({
                "event_type": event.get("type"),
                "response_id": event.get("response_id"),
                "values": found,
            })
    return found_rows


def _read_response_wav_metrics(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise EvidenceError(f"response playout evidence is missing: {path.name}")
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != CODEC_SAMPLE_RATE
            or source.getcomptype() != "NONE"
        ):
            raise EvidenceError(f"response playout evidence has the wrong PCM format: {path.name}")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    pcm = samples.astype(np.float32) / 32768.0
    frame_count = pcm.size // CODEC_FRAME_SAMPLES
    vocal_frames = sum(
        _rms(pcm[index * CODEC_FRAME_SAMPLES:(index + 1) * CODEC_FRAME_SAMPLES]) >= INPUT_RMS_MIN
        for index in range(frame_count)
    )
    return int(samples.size), int(vocal_frames)


def _reconcile_saved_run(output_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute acceptance facts from the saved event stream and PCM artifacts."""
    gaps: list[str] = []
    errors: list[str] = []
    event_path = output_dir / "events.jsonl"
    rows, parse_errors = _load_saved_event_rows(event_path)
    gaps.extend(parse_errors)
    event_manifest = report.get("event_log")
    if not isinstance(event_manifest, Mapping) or event_manifest.get("rows") != len(rows):
        gaps.append("preliminary report event count does not match the saved raw event log")
    if not isinstance(event_manifest, Mapping) or Path(str(event_manifest.get("path", ""))).name != "events.jsonl":
        gaps.append("preliminary report does not identify the saved raw event log")

    run_window = report.get("run_window_utc")
    run_started = _parse_utc_datetime(run_window.get("started_at")) if isinstance(run_window, Mapping) else None
    run_finished = _parse_utc_datetime(run_window.get("finished_at")) if isinstance(run_window, Mapping) else None
    if run_started is None or run_finished is None or run_finished < run_started:
        gaps.append("preliminary report has no valid run UTC bounds")
    acceptance_parameters = report.get("acceptance_parameters")
    acceptance_parameters = acceptance_parameters if isinstance(acceptance_parameters, Mapping) else {}
    try:
        reported_buffer_seconds = float(acceptance_parameters["playback_buffer_seconds"])
        reported_buffer_samples = int(acceptance_parameters["playback_buffer_max_samples"])
        expected_buffer_samples = _playback_capacity_samples(
            str(acceptance_parameters.get("mode", "")), reported_buffer_seconds
        )
        if reported_buffer_samples != expected_buffer_samples:
            gaps.append("reported playback FIFO sample capacity does not match its duration")
        if acceptance_parameters.get("mode") == "acceptance" and (
            reported_buffer_seconds != BROWSER_PLAYBACK_MAX_SECONDS
            or reported_buffer_samples != BROWSER_PLAYBACK_MAX_SAMPLES
        ):
            gaps.append("acceptance run did not use the browser-matched four-second playback FIFO")
        if acceptance_parameters.get("mode") == "diagnostic":
            gaps.append("diagnostic-mode runs cannot certify acceptance")
    except (KeyError, TypeError, ValueError, EvidenceError):
        gaps.append("preliminary report lacks a valid bounded playback FIFO configuration")
    event_types = [str(row.get("type", "")) for row in rows]
    starts = [row for row in rows if row.get("type") == "run_started"]
    finishes = [row for row in rows if row.get("type") == "run_finished"]
    if len(starts) != 1 or len(finishes) != 1:
        gaps.append("raw event log must contain one run_started and one run_finished marker")
    elif run_started is not None and run_finished is not None:
        start_payload = starts[0].get("payload", {})
        finish_payload = finishes[0].get("payload", {})
        if (
            _parse_utc_datetime(start_payload.get("started_at_utc")) != run_started
            or _parse_utc_datetime(finish_payload.get("finished_at_utc")) != run_finished
        ):
            gaps.append("report run bounds do not match the raw run markers")
        for row in rows:
            row_time = _parse_utc_datetime(row.get("utc"))
            if row_time is not None and not (
                run_started - timedelta(seconds=1) <= row_time <= run_finished + timedelta(seconds=1)
            ):
                gaps.append("raw event log contains records outside the report run window")
                break
    scenario_completed_rows = [row for row in rows if row.get("type") == "scenario_completed"]
    if len(scenario_completed_rows) != 1:
        errors.append("raw event log does not contain a completed scenario marker")
    elif scenario_completed_rows[0].get("payload", {}).get("async_tasks_stopped") is not True:
        errors.append("scenario counters were captured before asynchronous tasks stopped")
    if any(row.get("type") == "fatal_error" for row in rows):
        errors.append("raw event log records a simulator fatal error")

    session_created_rows = [row for row in rows if row.get("type") == "session.created"]
    session_updated_rows = [row for row in rows if row.get("type") == "session.updated"]
    if len(session_created_rows) != 1 or len(session_updated_rows) != 1:
        gaps.append("raw event log must contain one session.created and one session.updated")
    session_created = (
        session_created_rows[0].get("payload", {}) if len(session_created_rows) == 1 else {}
    )
    session_updated = (
        session_updated_rows[0].get("payload", {}) if len(session_updated_rows) == 1 else {}
    )
    connection = report.get("connection")
    connection = connection if isinstance(connection, Mapping) else {}
    session_id = str(session_created.get("session_id", ""))
    if not session_id or connection.get("session_id") != session_id:
        gaps.append("report session identity does not match raw session.created")
    if session_created.get("turn_policy") != TURN_POLICY or session_updated.get("turn_policy") != TURN_POLICY:
        errors.append("raw session events do not confirm gemma_native_control6")
    if not session_created.get("voice") or session_updated.get("voice") != session_created.get("voice"):
        gaps.append("raw session events do not confirm a stable effective voice")
    acceptance = report.get("acceptance_parameters")
    acceptance = acceptance if isinstance(acceptance, Mapping) else {}
    response_profile = acceptance.get("response_profile", "long")
    if session_updated.get("response_profile") != response_profile:
        errors.append("raw session.updated did not confirm the explicitly requested response profile")
    transport_rows = [row for row in rows if row.get("type") == "transport_verified"]
    tls = connection.get("tls") if isinstance(connection.get("tls"), Mapping) else {}
    if len(transport_rows) != 1:
        gaps.append("raw event log has no unique verified transport marker")
    else:
        transport = transport_rows[0].get("payload", {})
        if (
            tls.get("scheme") != "wss"
            or tls.get("peer_certificate_verified") is not True
            or bool(tls.get("insecure_loopback_opt_in"))
            or transport != tls
        ):
            errors.append("report TLS evidence does not match the verified WSS event")

    saved_fixture_manifest = output_dir / "fixture_manifest.json"
    try:
        fixture_copy = json.loads(saved_fixture_manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        fixture_copy = None
    if fixture_copy != report.get("fixture_manifest"):
        gaps.append("saved fixture manifest does not match the preliminary report")
    fixture_value = report.get("fixture_manifest")
    fixtures = fixture_value.get("fixtures") if isinstance(fixture_value, Mapping) else None
    interruption_fixture = fixtures.get("interruption") if isinstance(fixtures, Mapping) else None
    transcript = interruption_fixture.get("transcript") if isinstance(interruption_fixture, Mapping) else None
    if (
        not isinstance(interruption_fixture, Mapping)
        or interruption_fixture.get("transcript_verified") is not True
        or not isinstance(transcript, str)
        or "aspetta fermati un momento" not in re.sub(r"\W+", " ", transcript.casefold()).strip()
    ):
        gaps.append("saved fixture manifest lacks the ASR-verified interruption phrase")

    created_rows = [row for row in rows if row.get("type") == "response.created"]
    created_by_id: dict[str, Mapping[str, Any]] = {}
    for row in created_rows:
        event = row.get("payload", {})
        response_id = event.get("response_id") if isinstance(event, Mapping) else None
        if not isinstance(response_id, str) or not response_id or response_id in created_by_id:
            gaps.append("raw response.created events contain a missing or reused response_id")
            continue
        created_by_id[response_id] = event
    created_ns_by_id = {
        str(row["payload"]["response_id"]): row.get("monotonic_ns")
        for row in created_rows
        if isinstance(row.get("payload"), Mapping)
        and isinstance(row["payload"].get("response_id"), str)
    }
    assignment_rows = [row for row in rows if row.get("type") == "scenario_response_assigned"]
    assignments: dict[str, Mapping[str, Any]] = {}
    for row in assignment_rows:
        payload = row.get("payload", {})
        response_id = payload.get("response_id") if isinstance(payload, Mapping) else None
        if not isinstance(response_id, str) or response_id in assignments:
            gaps.append("scenario response assignments are missing or duplicated")
            continue
        assignments[response_id] = payload
    report_scenario = report.get("scenario")
    report_responses = report_scenario.get("responses") if isinstance(report_scenario, Mapping) else None
    if not isinstance(report_responses, list):
        gaps.append("preliminary report has no response metrics")
        report_responses = []
    response_reports = {
        str(item.get("response_id")): item
        for item in report_responses
        if isinstance(item, Mapping) and isinstance(item.get("response_id"), str)
    }
    speaking_created_ids = {
        rid for rid, ev in created_by_id.items()
        if str(ev.get("effective_action", ev.get("action", ""))).upper() == "SPEAK"
    }
    if set(created_by_id) != set(response_reports) or speaking_created_ids != set(assignments):
        gaps.append("response IDs in report, raw events, and scenario assignments do not match")

    done_by_id: dict[str, Mapping[str, Any]] = {}
    cancelled_by_id: dict[str, Mapping[str, Any]] = {}
    audio_by_id: dict[str, list[Mapping[str, Any]]] = {}
    first_audio_ns_by_id: dict[str, int] = {}
    playout_started_ns_by_id: dict[str, int] = {}
    playout_completed_ns_by_id: dict[str, int] = {}
    finalizer_ns_by_id: dict[str, int] = {}
    markers_by_id: dict[str, dict[str, Mapping[str, Any]]] = {}
    acks_by_id: dict[str, dict[str, Mapping[str, Any]]] = {}
    finalizers_by_id: dict[str, Mapping[str, Any]] = {}
    text_by_id: dict[str, list[str]] = {}
    ack_sequences: list[int] = []
    quality_complete = True
    for row in rows:
        event = row.get("payload", {})
        if not isinstance(event, Mapping):
            continue
        response_id = event.get("response_id")
        if row.get("type") == "response.done" and isinstance(response_id, str):
            done_by_id[response_id] = event
        elif row.get("type") == "response.cancelled" and isinstance(response_id, str):
            cancelled_by_id[response_id] = event
        elif row.get("type") == "response.audio.delta" and isinstance(response_id, str):
            audio_by_id.setdefault(response_id, []).append(event)
            event_ns = row.get("monotonic_ns")
            if isinstance(event_ns, int) and not isinstance(event_ns, bool):
                first_audio_ns_by_id.setdefault(response_id, int(event_ns))
            quality = event.get("raw_source_audio_quality")
            samples = event.get("sample_count")
            if (
                not isinstance(quality, Mapping)
                or quality.get("finite") is not True
                or isinstance(quality.get("rms"), bool)
                or not isinstance(quality.get("rms"), (int, float))
                or not math.isfinite(float(quality.get("rms")))
                or isinstance(quality.get("peak"), bool)
                or not isinstance(quality.get("peak"), (int, float))
                or not math.isfinite(float(quality.get("peak")))
                or quality.get("clipping_rail_samples") != 0
                or quality.get("sample_count") != samples
                or not isinstance(samples, int)
                or isinstance(samples, bool)
                or samples <= 0
            ):
                quality_complete = False
        elif row.get("type") == "response.audio.segment_done" and isinstance(response_id, str):
            segment_id = event.get("speaker_segment_id")
            if isinstance(segment_id, str):
                markers_by_id.setdefault(response_id, {})[segment_id] = event
        elif row.get("type") == "client_consumed_pcm" and isinstance(response_id, str):
            segment_id = event.get("speaker_segment_id")
            sequence = event.get("sequence")
            if not isinstance(segment_id, str) or event.get("consumption_semantics") != "logical_client_device_buffer":
                gaps.append("raw client ACK is missing canonical segment identity/semantics")
            elif isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                gaps.append("raw client ACK sequence is invalid")
            else:
                ack_sequences.append(sequence)
                acks_by_id.setdefault(response_id, {})[segment_id] = event
        elif row.get("type") == "client_consumed_pcm_ack_discarded":
            errors.append("client discarded a playback ACK")
        elif row.get("type") == "response.unified_native_finalized" and isinstance(response_id, str):
            finalizers_by_id[response_id] = event
            event_ns = row.get("monotonic_ns")
            if isinstance(event_ns, int) and not isinstance(event_ns, bool):
                finalizer_ns_by_id[response_id] = int(event_ns)
        elif row.get("type") == "playout_started" and isinstance(response_id, str):
            event_ns = event.get("playout_started_ns", row.get("monotonic_ns"))
            if isinstance(event_ns, int) and not isinstance(event_ns, bool):
                playout_started_ns_by_id[response_id] = int(event_ns)
        elif row.get("type") == "playout_completed" and isinstance(response_id, str):
            event_ns = event.get("playout_completed_ns", row.get("monotonic_ns"))
            if isinstance(event_ns, int) and not isinstance(event_ns, bool):
                playout_completed_ns_by_id[response_id] = int(event_ns)
        elif row.get("type") == "response.output_text.delta" and isinstance(response_id, str):
            delta = event.get("delta")
            if isinstance(delta, str):
                text_by_id.setdefault(response_id, []).append(delta)
        elif row.get("type") == "error":
            errors.append("raw WebSocket event log contains a server error")

    done_ns_by_id = {
        str(row["payload"]["response_id"]): row.get("monotonic_ns")
        for row in rows
        if row.get("type") == "response.done"
        and isinstance(row.get("payload"), Mapping)
        and isinstance(row["payload"].get("response_id"), str)
    }
    if ack_sequences != sorted(set(ack_sequences)):
        gaps.append("raw client ACK sequence is not strictly increasing across the connection")
    report_by_id = response_reports
    assigned_tags = {str(value.get("scenario_tag")) for value in assignments.values()}
    expected_tags = {
        "long_response_completed", "long_response_interrupted",
        "memory_followup_resume", "closure",
    }
    allowed_attempt_tags = {"long_response_early_interruption_attempt"}
    if not expected_tags.issubset(assigned_tags) or not assigned_tags.issubset(
        expected_tags | allowed_attempt_tags
    ):
        errors.append("raw scenario response tags do not include the required four dialogue turns")
    expected_response_order = [
        "long_response_completed",
        "memory_followup_resume",
        "long_response_interrupted",
        "closure",
    ]
    observed_response_order = [
        str(value.get("scenario_tag"))
        for value in assignments.values()
        if str(value.get("scenario_tag")) in expected_tags
    ]
    if observed_response_order != expected_response_order:
        errors.append("raw dialogue turns do not place the memory follow-up immediately after the long response")

    response_facts: dict[str, dict[str, Any]] = {}
    for response_id, created in created_by_id.items():
        report_response = report_by_id.get(response_id)
        assignment = assignments.get(response_id)
        if not isinstance(report_response, Mapping) or not isinstance(assignment, Mapping):
            continue
        action = str(created.get("effective_action", created.get("action", ""))).upper()
        if action != "SPEAK" or assignment.get("action") != action or report_response.get("action") != action:
            errors.append(f"response {response_id} did not preserve the SPEAK action identity")
        samples = sum(int(event.get("sample_count", 0)) for event in audio_by_id.get(response_id, ()))
        segment_ids = {
            str(event.get("speaker_segment_id"))
            for event in audio_by_id.get(response_id, ())
            if isinstance(event.get("speaker_segment_id"), str)
        }
        marker_ids = set(markers_by_id.get(response_id, {}))
        ack_ids = set(acks_by_id.get(response_id, {}))
        try:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", response_id)
            wav_samples, vocal_frames = _read_response_wav_metrics(
                output_dir / f"response_{safe_id}_received.wav"
            ) if samples else (0, 0)
        except (EvidenceError, OSError, wave.Error) as exc:
            gaps.append(f"response {response_id} raw PCM evidence is invalid: {exc}")
            wav_samples, vocal_frames = -1, -1
        done_event = done_by_id.get(response_id)
        cancelled_event = cancelled_by_id.get(response_id)
        done_status = done_event.get("status") if done_event else None
        cancel_reason = cancelled_event.get("reason") if cancelled_event else None
        finalizer = finalizers_by_id.get(response_id)
        finalizer_status = finalizer.get("status") if finalizer else None
        if (
            report_response.get("audio_samples") != samples
            or wav_samples != samples
            or report_response.get("vocal_frames_rms_ge_0_015") != vocal_frames
            or set(report_response.get("segment_ids", ())) != segment_ids
            or set(report_response.get("segment_done_ids", ())) != marker_ids
            or set(report_response.get("client_consumed_ack_ids", ())) != ack_ids
            or report_response.get("done_status") != done_status
            or report_response.get("finish_reason") != (done_event.get("finish_reason") if done_event else None)
            or report_response.get("cancel_reason") != cancel_reason
            or report_response.get("text") != "".join(text_by_id.get(response_id, ()))
        ):
            gaps.append(f"response {response_id} report metrics do not match raw events/PCM")
        if done_event is not None and not cancelled_event and (
            not segment_ids or marker_ids != segment_ids or ack_ids != marker_ids
        ):
            errors.append(f"normally completed response {response_id} did not drain every marked segment ACK")
        if finalizer_status != "completed":
            errors.append(f"response {response_id} lacks a completed playback finalizer")
        response_facts[response_id] = {
            "response_id": response_id,
            "scenario_tag": assignment.get("scenario_tag"),
            "action": action,
            "sample_count": samples,
            "vocal_frames": vocal_frames,
            "segment_ids": segment_ids,
            "marker_ids": marker_ids,
            "ack_ids": ack_ids,
            "done": done_event is not None,
            "cancelled": cancelled_event is not None,
            "cancel_reason": cancel_reason,
            "done_status": done_status,
        }

    normal = [fact for fact in response_facts.values() if fact["done"] and not fact["cancelled"]]
    cancelled = [fact for fact in response_facts.values() if fact["cancelled"]]
    long_done = [
        fact for fact in normal
        if fact["scenario_tag"] == "long_response_completed"
        and fact["sample_count"] >= LONG_RESPONSE_MIN_SAMPLES
        and fact["vocal_frames"] >= LONG_RESPONSE_MIN_VOCAL_FRAMES
    ]
    if len(normal) < 3 or not cancelled or not long_done:
        errors.append("raw response events fail the 3-completed/1-cancelled/long-response gates")
    for tag in ("memory_followup_resume", "closure"):
        if not any(fact["scenario_tag"] == tag and fact["done"] and not fact["cancelled"] for fact in response_facts.values()):
            errors.append(f"same-connection {tag} response did not complete")

    input_rows = [row for row in rows if row.get("type") == "input_frame_sent"]
    input_acks = [row for row in rows if row.get("type") == "input_frame_ack_received"]
    sent_payloads = [row.get("payload", {}) for row in input_rows]
    ack_payloads = [row.get("payload", {}) for row in input_acks]
    sent_ids = [payload.get("local_frame_id") for payload in sent_payloads]
    ack_ids = [payload.get("local_frame_id") for payload in ack_payloads]
    if (
        not sent_payloads
        or sent_ids != list(range(len(sent_payloads)))
        or sorted(ack_ids) != list(range(len(sent_payloads)))
        or len(ack_ids) != len(set(ack_ids))
    ):
        errors.append("raw input frame sends/ACKs are incomplete, duplicated, or reordered")
    report_input = report.get("input") if isinstance(report.get("input"), Mapping) else {}
    if (
        report_input.get("frames_sent") != len(sent_payloads)
        or report_input.get("frames_acked") != len(ack_payloads)
        or report_input.get("frames_inflight_at_end") != 0
    ):
        gaps.append("input counters in the preliminary report do not match raw input events")
    sender_timing = report_input.get("sender_timing")
    expected_sender_timing = {
        "frames_waiting_for_backpressure": sum(
            bool(payload.get("waited_for_backpressure")) for payload in sent_payloads
        ),
        "total_backpressure_wait_ms": sum(
            int(payload.get("backpressure_wait_ns", 0))
            for payload in sent_payloads
            if isinstance(payload.get("backpressure_wait_ns", 0), int)
            and not isinstance(payload.get("backpressure_wait_ns", 0), bool)
        ) / 1e6,
        "max_backpressure_wait_ms": max((
            int(payload.get("backpressure_wait_ns", 0)) / 1e6
            for payload in sent_payloads
            if isinstance(payload.get("backpressure_wait_ns", 0), int)
            and not isinstance(payload.get("backpressure_wait_ns", 0), bool)
        ), default=0.0),
        "max_sender_scheduler_delay_ms": max((
            int(payload.get("sender_scheduler_delay_ns", 0)) / 1e6
            for payload in sent_payloads
            if isinstance(payload.get("sender_scheduler_delay_ns", 0), int)
            and not isinstance(payload.get("sender_scheduler_delay_ns", 0), bool)
        ), default=0.0),
        "max_gap_ms": DEFAULT_MAX_GAP_MS,
        "backpressure_and_scheduler_delay_are_separate": True,
    }
    if (
        any(
            "backpressure_wait_ns" not in payload
            or "sender_scheduler_delay_ns" not in payload
            for payload in sent_payloads
        )
        or sender_timing != expected_sender_timing
    ):
        gaps.append("sender scheduling delay and ACK backpressure metrics do not match raw frames")
    if len(scenario_completed_rows) == 1:
        marker = scenario_completed_rows[0].get("payload", {})
        if (
            marker.get("input_frames_sent") != len(sent_payloads)
            or marker.get("input_frames_acked") != len(ack_payloads)
            or marker.get("playback_pending_samples") != 0
        ):
            gaps.append("scenario completion counters do not match stopped-task input/playout state")

    long_response_ids = [
        response_id for response_id, assignment in assignments.items()
        if assignment.get("scenario_tag") == "long_response_completed"
    ]
    if len(long_response_ids) != 1:
        errors.append("raw scenario must identify exactly one completed long response")
    else:
        long_response_id = long_response_ids[0]
        long_done_event = done_by_id.get(long_response_id)
        if long_done_event is None or long_done_event.get("finish_reason") not in {"stop", "length"}:
            errors.append("long response.done lacks natural stop or explicit length finish_reason")

        fifo_trace_path = output_dir / "playback_queue_trace.jsonl"
        fifo_trace_rows: list[Mapping[str, Any]] = []
        try:
            for line in fifo_trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                trace_row = json.loads(line)
                if isinstance(trace_row, Mapping):
                    fifo_trace_rows.append(trace_row)
        except (OSError, json.JSONDecodeError) as exc:
            gaps.append(f"saved playback FIFO trace is missing or invalid: {exc}")
        trace_peak = max((
            int(trace_row.get(key, 0))
            for trace_row in fifo_trace_rows
            for key in ("pending_samples", "pending_before_samples", "pending_after_samples")
            if isinstance(trace_row.get(key), int) and not isinstance(trace_row.get(key), bool)
        ), default=0)
        report_playout = report.get("playout") if isinstance(report.get("playout"), Mapping) else {}
        reported_fifo_peak = report_playout.get("fifo_peak_pending_samples")
        if (
            not isinstance(reported_fifo_peak, int)
            or isinstance(reported_fifo_peak, bool)
            or reported_fifo_peak != trace_peak
        ):
            gaps.append("reported playback FIFO peak does not match the saved queue trace")
        if trace_peak > BROWSER_PLAYBACK_MAX_SAMPLES:
            errors.append(
                f"raw playback FIFO peak {trace_peak} exceeds {BROWSER_PLAYBACK_MAX_SAMPLES} samples"
            )

        done_ns = done_ns_by_id.get(long_response_id)
        first_audio_ns = first_audio_ns_by_id.get(long_response_id)
        playout_started_ns = playout_started_ns_by_id.get(long_response_id)
        playout_completed_ns = playout_completed_ns_by_id.get(long_response_id)
        final_observation_ns = finalizer_ns_by_id.get(long_response_id)
        pcm_samples = sum(
            int(event.get("sample_count", 0))
            for event in audio_by_id.get(long_response_id, ())
        )
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (
            first_audio_ns, done_ns, playout_started_ns,
            playout_completed_ns, final_observation_ns,
        )):
            scenario_report = (
                report.get("scenario")
                if isinstance(report.get("scenario"), Mapping)
                else {}
            )
            reported_long_window = (
                scenario_report.get("long_response_input_window")
                if isinstance(scenario_report, Mapping)
                else {}
            )
            saved_cutoff = (
                (reported_long_window or {}).get("input_ack_cutoff_ns")
                if isinstance(reported_long_window, Mapping)
                else None
            )
            max_ack_cutoff_ns = int(playout_completed_ns) + 2_000_000_000
            if (
                isinstance(saved_cutoff, int)
                and not isinstance(saved_cutoff, bool)
                and saved_cutoff > max_ack_cutoff_ns
            ):
                errors.append(
                    "reported long-response input ACK cutoff exceeds the two-second grace window"
                )
            recomputed_cutoff_ns = (
                int(saved_cutoff)
                if isinstance(saved_cutoff, int) and not isinstance(saved_cutoff, bool)
                else max_ack_cutoff_ns
            )
            fifo_window_peak = max((
                int(trace_row.get(key, 0))
                for trace_row in fifo_trace_rows
                if isinstance(trace_row.get("monotonic_ns"), int)
                and playout_started_ns <= trace_row["monotonic_ns"] <= playout_completed_ns
                for key in ("pending_samples", "pending_before_samples", "pending_after_samples")
                if isinstance(trace_row.get(key), int) and not isinstance(trace_row.get(key), bool)
            ), default=0)
            try:
                recomputed_long_window = _long_response_input_window_metrics(
                    pcm_samples=pcm_samples,
                    first_audio_ns=int(first_audio_ns),
                    done_ns=done_ns,
                    playout_started_ns=int(playout_started_ns),
                    playout_completed_ns=int(playout_completed_ns),
                    final_observation_ns=int(final_observation_ns),
                    input_ack_cutoff_ns=recomputed_cutoff_ns,
                    input_frame_sends=[
                        {
                            "local_frame_id": payload.get("local_frame_id"),
                            "sent_ns": payload.get("sent_ns"),
                            "sender_scheduler_delay_ns": payload.get("sender_scheduler_delay_ns"),
                            "backpressure_wait_ns": payload.get("backpressure_wait_ns"),
                        }
                        for payload in sent_payloads
                    ],
                    input_frame_acks=[
                        {
                            "local_frame_id": payload.get("local_frame_id"),
                            "received_ns": row.get("monotonic_ns"),
                        }
                        for row, payload in zip(input_acks, ack_payloads)
                    ],
                    fifo_peak_samples=fifo_window_peak,
                )
            except EvidenceError as exc:
                errors.append(str(exc))
            else:
                comparison_long_window = recomputed_long_window
                if (
                    isinstance(reported_long_window, Mapping)
                    and "input_ack_cutoff_ns" not in reported_long_window
                ):
                    comparison_long_window = {
                        key: value
                        for key, value in recomputed_long_window.items()
                        if key != "input_ack_cutoff_ns"
                    }
                if reported_long_window != comparison_long_window:
                    gaps.append("long-response input-window metrics do not match raw sends, ACKs, and FIFO trace")
                long_report = response_reports.get(long_response_id)
                expected_timing = {
                    "first_audio_ns": int(first_audio_ns),
                    "done_ns": int(done_ns),
                    "playout_started_ns": int(playout_started_ns),
                    "playout_completed_ns": int(playout_completed_ns),
                    "final_observation_ns": int(final_observation_ns),
                    "pcm_duration_seconds": pcm_samples / CODEC_SAMPLE_RATE,
                    "production_window_seconds": (int(done_ns) - int(first_audio_ns)) / 1e9,
                    "playout_window_seconds": (
                        int(playout_completed_ns) - int(playout_started_ns)
                    ) / 1e9,
                }
                if not isinstance(long_report, Mapping) or any(
                    long_report.get(key) != value for key, value in expected_timing.items()
                ):
                    gaps.append("long-response PCM, production, and playout windows do not match raw timestamps")
        else:
            gaps.append(
                "raw long response lacks monotonic audio, playout, done, or finalizer timestamps"
            )

    inflight_high_water = max(
        (int(payload.get("inflight_after_reservation", 0)) for payload in sent_payloads),
        default=0,
    )
    if inflight_high_water >= MAX_INPUT_INFLIGHT:
        errors.append("raw input in-flight reached the fail-closed threshold of 16")
    pacing_config = report.get("acceptance_parameters", {})
    try:
        configured_gap_ms = float(pacing_config["max_input_gap_ms"])
        if configured_gap_ms > DEFAULT_MAX_GAP_MS:
            raise ValueError(
                f"max input gap {configured_gap_ms:g}ms exceeds the fixed {DEFAULT_MAX_GAP_MS:g}ms limit"
            )
        pacing = PacingMonitor(
            tolerance_ms=float(pacing_config["pacing_tolerance_ms"]),
            max_gap_ms=configured_gap_ms,
        )
        for payload in sent_payloads:
            sent_ns = int(payload["sent_ns"])
            backpressure_wait_ns = int(payload.get("backpressure_wait_ns", 0))
            if payload.get("waited_for_backpressure"):
                pacing.deadline_ns = sent_ns
            pacing.record_send(sent_ns, backpressure_wait_ns=backpressure_wait_ns)
    except (KeyError, TypeError, ValueError, EvidenceError) as exc:
        errors.append(f"raw input pacing failed revalidation: {exc}")
        pacing = None
    for payload in sent_payloads:
        rms = payload.get("frame_rms_before_quantization")
        rails = payload.get("prequantization_pcm16_rail_samples")
        if (
            isinstance(rms, bool) or not isinstance(rms, (int, float))
            or not math.isfinite(float(rms))
            or isinstance(rails, bool) or not isinstance(rails, int) or rails != 0
        ):
            errors.append("raw input RMS/clipping telemetry failed revalidation")
            break
        if payload.get("fixture") == "idle_noise_floor" and not NOISE_RMS_MIN <= float(rms) <= NOISE_RMS_MAX:
            errors.append("raw idle input noise RMS is outside the required floor range")
            break
    if isinstance(report_input.get("inflight_high_water"), int) and report_input.get("inflight_high_water") != inflight_high_water:
        gaps.append("input in-flight high-water report does not match raw sends")

    ack_rejections = [row for row in rows if row.get("type") == "client_consumed_pcm_ack_discarded"]
    if ack_rejections:
        errors.append("raw event log contains rejected playback ACKs")
    if any(fact["cancelled"] and fact["cancel_reason"] != "yield" for fact in cancelled):
        errors.append("cancelled response did not use the native YIELD action")
    barge_fixtures = [
        row.get("payload", {}) for row in rows
        if row.get("type") == "fixture_scheduled"
        and isinstance(row.get("payload"), Mapping)
        and row["payload"].get("scenario_tag") == "barge_in"
    ]
    interrupted = next((
        fact for fact in response_facts.values()
        if fact["scenario_tag"] == "long_response_interrupted"
    ), None)
    if len(barge_fixtures) != 1 or interrupted is None:
        errors.append("raw event log lacks the dedicated interruption fixture/response")
    else:
        delivery_id = barge_fixtures[0].get("delivery_id")
        vocal_inputs = [
            payload for payload in sent_payloads
            if payload.get("delivery_id") == delivery_id
            and isinstance(payload.get("frame_rms_before_quantization"), (int, float))
            and float(payload["frame_rms_before_quantization"]) >= INPUT_RMS_MIN
        ]
        onset_rows = [
            row.get("payload", {}) for row in rows
            if row.get("type") == "playout_onset"
            and isinstance(row.get("payload"), Mapping)
            and row["payload"].get("response_id") == interrupted["response_id"]
        ]
        cancel_rows = [
            row.get("payload", {}) for row in rows
            if row.get("type") == "response_cancel_observed"
            and isinstance(row.get("payload"), Mapping)
            and row["payload"].get("response_id") == interrupted["response_id"]
        ]
        if not vocal_inputs or len(onset_rows) != 1 or len(cancel_rows) != 1:
            errors.append("raw interruption onset/cancellation timing evidence is incomplete")
        else:
            first_vocal_ns = min(int(item["sent_ns"]) for item in vocal_inputs)
            playout_onset_ns = int(onset_rows[0].get("playout_onset_ns", -1))
            cancel_ns = int(cancel_rows[0].get("cancelled_ns", -1))
            flush_ns = int(cancel_rows[0].get("flush_ns", -1))
            cancel_budget = float(pacing_config.get("cancellation_budget_ms", 250.0))
            onset_delay_ns = first_vocal_ns - playout_onset_ns
            if (
                playout_onset_ns < 0
                or onset_delay_ns < int(float(pacing_config.get("interrupt_after_playout_s", 3.0)) * 1e9) - 1_000_000
                or max(cancel_ns - first_vocal_ns, flush_ns - first_vocal_ns) > cancel_budget * 1e6
                or cancel_rows[0].get("reason") != "yield"
            ):
                errors.append("raw YIELD cancellation/flush missed the actual-onset 250 ms gate")

    client_events = [row.get("payload", {}) for row in rows if row.get("direction") == "client"]
    if any(event.get("type") == "response.create" for event in client_events if isinstance(event, Mapping)):
        errors.append("raw client events contain an injected response.create")
    if any(event.get("type") in {"input_audio_transcription.partial", "input_audio_transcription.completed"} for event in client_events if isinstance(event, Mapping)):
        errors.append("raw client events contain injected transcription text")
    updates = [
        event for event in client_events
        if isinstance(event, Mapping) and event.get("type") == "session.update"
    ]
    update_config = updates[0].get("session", {}) if len(updates) == 1 else {}
    if (
        len(updates) != 1
        or not isinstance(update_config, Mapping)
        or update_config.get("auto_response") is not True
        or update_config.get("turn_policy") != TURN_POLICY
        or update_config.get("input_audio_transcription") is not False
        or update_config.get("response_profile") != response_profile
        or "voice" in update_config
    ):
        errors.append("raw client session update does not match the automatic native Control6 contract")

    recomputed_kv = _extract_kv_evidence_from_rows(rows)
    runtime_evidence = report.get("runtime_evidence")
    runtime_evidence = runtime_evidence if isinstance(runtime_evidence, Mapping) else {}
    if runtime_evidence.get("kv_events") != recomputed_kv:
        gaps.append("reported runtime KV events do not match the raw WebSocket event stream")
    if runtime_evidence.get("raw_source_audio_telemetry_complete") != quality_complete:
        gaps.append("reported raw-source audio telemetry flag does not match packet evidence")
    return {
        "rows": rows,
        "event_path": event_path,
        "run_started_at_utc": run_started,
        "run_finished_at_utc": run_finished,
        "session_created": session_created,
        "session_updated": session_updated,
        "response_facts": response_facts,
        "scenario_errors": errors,
        "evidence_gaps": gaps,
        "runtime_kv_evidence": recomputed_kv,
        "runtime_audio_telemetry_complete": quality_complete,
        "pacing": None if pacing is None else pacing.report(),
    }


def _offline_finalize_report(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    try:
        preliminary = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"preliminary report cannot be read: {exc}") from exc
    if not isinstance(preliminary, dict):
        raise EvidenceError("preliminary report must be a JSON object")
    reconciled = _reconcile_saved_run(output_dir, preliminary)
    final_report = copy.deepcopy(preliminary)
    integrity_errors = list(reconciled["scenario_errors"])
    integrity_gaps = list(reconciled["evidence_gaps"])
    original_scenario = final_report.get("scenario")
    if not isinstance(original_scenario, dict):
        original_scenario = {}
        final_report["scenario"] = original_scenario
    original_errors = original_scenario.get("errors", [])
    if isinstance(original_errors, list):
        for item in original_errors:
            if item not in integrity_errors:
                integrity_errors.append(str(item))
    stored_errors = final_report.get("errors", [])
    if isinstance(stored_errors, list) and stored_errors:
        integrity_errors.append("preliminary run recorded WebSocket/runtime errors")

    sources = final_report.get("runtime_evidence", {}).get("sources", {})
    sources = sources if isinstance(sources, Mapping) else {}
    server_log = args.server_log or sources.get("server_log")
    server_log_window = args.server_log_window or sources.get("server_log_window")
    runtime_telemetry = args.runtime_telemetry or sources.get("runtime_telemetry")
    parameters = final_report.get("acceptance_parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    for name in (
        "vram_peak_over_baseline_budget_mib",
        "vram_post_drain_growth_budget_mib",
        "vram_reserved_peak_growth_budget_mib",
        "vram_reserved_post_drain_growth_budget_mib",
    ):
        supplied = getattr(args, name, None)
        recorded = parameters.get(name)
        if supplied is not None and supplied != recorded:
            integrity_gaps.append(f"offline finalization cannot change the run-time budget {name}")
    evaluator = object.__new__(RealDialogueSimulator)
    evaluator.args = argparse.Namespace(
        server_log=server_log,
        server_log_window=server_log_window,
        runtime_telemetry=runtime_telemetry,
        vram_peak_over_baseline_budget_mib=parameters.get("vram_peak_over_baseline_budget_mib"),
        vram_post_drain_growth_budget_mib=parameters.get("vram_post_drain_growth_budget_mib"),
        vram_reserved_peak_growth_budget_mib=parameters.get("vram_reserved_peak_growth_budget_mib"),
        vram_reserved_post_drain_growth_budget_mib=parameters.get("vram_reserved_post_drain_growth_budget_mib"),
    )
    evaluator.evidence_gaps = integrity_gaps
    evaluator.session_created = reconciled["session_created"]
    evaluator.responses = {
        response_id: SimpleNamespace(response_id=response_id, scenario_tag=fact["scenario_tag"])
        for response_id, fact in reconciled["response_facts"].items()
    }
    evaluator.runtime_kv_evidence = reconciled["runtime_kv_evidence"]
    evaluator.runtime_audio_telemetry_complete = reconciled["runtime_audio_telemetry_complete"]
    evaluator.run_started_at_utc = reconciled["run_started_at_utc"]
    evaluator.run_finished_at_utc = reconciled["run_finished_at_utc"]
    log_evidence = evaluator._read_server_log()
    runtime = evaluator._read_runtime_telemetry()
    telemetry = evaluator._assess_telemetry(runtime)
    all_gaps = list(dict.fromkeys(evaluator.evidence_gaps))
    final_report["runtime_evidence"] = dict(final_report.get("runtime_evidence", {}))
    final_report["runtime_evidence"].update({
        "server_log": log_evidence,
        "telemetry": telemetry,
        "kv_events": reconciled["runtime_kv_evidence"],
        "raw_source_audio_telemetry_complete": reconciled["runtime_audio_telemetry_complete"],
        "sources": {
            "server_log": server_log,
            "server_log_window": server_log_window,
            "runtime_telemetry": runtime_telemetry,
        },
    })
    original_scenario["errors"] = integrity_errors
    original_scenario["evidence_gaps"] = all_gaps
    original_scenario["normal_response_count"] = sum(
        bool(fact["done"] and not fact["cancelled"])
        for fact in reconciled["response_facts"].values()
    )
    original_scenario["cancelled_response_count"] = sum(
        bool(fact["cancelled"]) for fact in reconciled["response_facts"].values()
    )
    original_scenario["long_completed_response_count"] = sum(
        fact["scenario_tag"] == "long_response_completed"
        and fact["done"] and not fact["cancelled"]
        and fact["sample_count"] >= LONG_RESPONSE_MIN_SAMPLES
        and fact["vocal_frames"] >= LONG_RESPONSE_MIN_VOCAL_FRAMES
        for fact in reconciled["response_facts"].values()
    )
    final_report["overall_status"] = (
        "INCOMPLETE" if all_gaps
        else "FAIL" if integrity_errors or telemetry.get("status") == "FAIL"
        else "PASS" if telemetry.get("status") == "PASS"
        else "INCOMPLETE"
    )
    final_report["finalization"] = {
        "mode": "offline_evidence_finalize",
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "preliminary_report": str(report_path),
        "raw_event_log": str(reconciled["event_path"]),
        "raw_event_rows_reconciled": len(reconciled["rows"]),
        "scenario_gates_recomputed": True,
        "external_validators_reused": ["_read_server_log", "_assess_telemetry"],
    }
    final_path = output_dir / "report.finalized.json"
    final_path.write_text(json.dumps(_json_safe(final_report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final_report


def _required_protocol_int(event: Mapping[str, Any], key: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{key} must be a non-negative integer")
    return value


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


_SERVER_LOG_TIMESTAMP_RE = re.compile(
    r"^\s*\[?(?P<stamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)


def _server_log_line_timestamp(line: str) -> datetime | None:
    match = _SERVER_LOG_TIMESTAMP_RE.match(line)
    if match is None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, Mapping):
            return None
        stamp = next((
            payload.get(key) for key in ("timestamp_utc", "timestamp", "time", "asctime")
            if isinstance(payload.get(key), str)
        ), None)
        return _parse_utc_datetime(stamp)
    stamp = match.group("stamp").replace(",", ".")
    # Naive app log timestamps are interpreted as UTC because the capture
    # contract requires a UTC query and this runtime host is configured for UTC.
    if not re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", stamp):
        stamp = stamp.replace(" ", "T", 1) + "+00:00"
    return _parse_utc_datetime(stamp)


def _scope_server_log_content(
    content: str,
    *,
    since: datetime | None,
    until: datetime | None,
    run_started: datetime | None,
    run_finished: datetime | None,
) -> tuple[str, int, int]:
    """Keep timestamped log records inside the captured simulator window.

    Lines without a new timestamp are continuation text for the preceding
    record, so multi-line CUDA diagnostics and stack traces retain their
    process/time context. If the source format has no parseable timestamps at
    all, the verified process-scoped --since/--until envelope is authoritative.
    """
    records: list[tuple[datetime, list[str]]] = []
    current: tuple[datetime, list[str]] | None = None
    prefix_lines: list[str] = []
    for line in content.splitlines():
        timestamp = _server_log_line_timestamp(line)
        if timestamp is not None:
            current = (timestamp, [line])
            records.append(current)
        elif current is not None:
            current[1].append(line)
        else:
            prefix_lines.append(line)

    if not records:
        return content, 0, 0
    included: list[str] = []
    outside = 0
    slack = timedelta(seconds=1)
    for timestamp, lines in records:
        in_query = since is not None and until is not None and since <= timestamp <= until
        in_run = (
            run_started is not None
            and run_finished is not None
            and run_started - slack <= timestamp <= run_finished + slack
        )
        if in_query and in_run:
            included.extend(lines)
        else:
            outside += 1
    return "\n".join(included), len(records), outside


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize-report",
        default=None,
        metavar="OUTPUT_DIR",
        help="offline reanalysis of a saved run after its log and runtime telemetry are collected",
    )
    parser.add_argument(
        "--url",
        default="wss://127.0.0.1:8443/v1/realtime?duplex=1",
        help="Realtime WebSocket URL (default is verified TLS on local port 8443)",
    )
    parser.add_argument(
        "--fixture-manifest",
        default=None,
        help=(
            "manifest with long_request/interruption/memory_followup/closure; each WAV must pair "
            "with offline ASR JSON metadata"
        ),
    )
    parser.add_argument("--mode", choices=("acceptance", "diagnostic"), default="acceptance")
    parser.add_argument(
        "--response-profile",
        choices=("standard", "long"),
        default="long",
        help="explicit server response profile; long requests up to 768 text tokens and 1536 codec frames",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--server-log", default=None, help="local server log captured for this same run")
    parser.add_argument(
        "--server-log-window",
        default=None,
        help="small JSON sidecar with matching session_id and UTC since_utc/until_utc for the run-only log",
    )
    parser.add_argument("--runtime-telemetry", default=None, help="runtime KV/VRAM JSON sidecar")
    parser.add_argument(
        "--ca-cert",
        default=None,
        help="local CA or explicitly trusted self-signed server certificate for WSS verification",
    )
    parser.add_argument(
        "--tls-server-name",
        default=None,
        help="explicit expected TLS certificate hostname/IP when it differs from the loopback URL",
    )
    parser.add_argument(
        "--insecure-loopback",
        action="store_true",
        help="disable TLS verification only for loopback diagnostic runs; acceptance cannot pass",
    )
    parser.add_argument("--vram-peak-over-baseline-budget-mib", type=float, default=None)
    parser.add_argument("--vram-post-drain-growth-budget-mib", type=float, default=None)
    parser.add_argument("--vram-reserved-peak-growth-budget-mib", type=float, default=None)
    parser.add_argument("--vram-reserved-post-drain-growth-budget-mib", type=float, default=None)
    parser.add_argument("--noise-rms", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=407)
    parser.add_argument("--pacing-tolerance-ms", type=float, default=DEFAULT_PACING_TOLERANCE_MS)
    parser.add_argument("--max-input-gap-ms", type=float, default=DEFAULT_MAX_GAP_MS)
    parser.add_argument(
        "--playback-buffer-seconds",
        type=float,
        default=BROWSER_PLAYBACK_MAX_SECONDS,
        help=(
            "simulated PCM FIFO horizon; acceptance is fixed to the browser's 4.0s limit, "
            "diagnostic runs may choose a smaller bound"
        ),
    )
    parser.add_argument("--interrupt-after-playout-s", type=float, default=3.0)
    parser.add_argument("--cancellation-budget-ms", type=float, default=250.0)
    parser.add_argument("--handshake-timeout-s", type=float, default=20.0)
    parser.add_argument("--response-timeout-s", type=float, default=120.0)
    parser.add_argument("--playback-timeout-s", type=float, default=180.0)
    parser.add_argument("--cancellation-timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--instructions",
        default=(
            "Parla in italiano e rispondi naturalmente. Ricorda i fatti gia comunicati "
            "nella conversazione. Per una richiesta di itinerario di piu giorni, dai "
            "una proposta dettagliata giorno per giorno con spostamenti e pause; non "
            "presentare prezzi o disponibilita come confermati."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.finalize_report:
        try:
            report = _offline_finalize_report(Path(args.finalize_report).expanduser().resolve(), args)
        except Exception as exc:
            parser.exit(2, f"offline report finalization failed: {type(exc).__name__}: {exc}\n")
        output_dir = str(Path(args.finalize_report).expanduser().resolve())
        print(json.dumps({
            "overall_status": report.get("overall_status"),
            "output_dir": output_dir,
            "final_report": str(Path(output_dir) / "report.finalized.json"),
            "normal_response_count": report.get("scenario", {}).get("normal_response_count"),
            "cancelled_response_count": report.get("scenario", {}).get("cancelled_response_count"),
            "evidence_gaps": report.get("scenario", {}).get("evidence_gaps"),
            "errors": report.get("scenario", {}).get("errors"),
        }, ensure_ascii=False, indent=2))
        return 0 if report.get("overall_status") == "PASS" else 2
    if not args.fixture_manifest:
        parser.error("--fixture-manifest is required unless --finalize-report is used")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = str(REPO_ROOT / "artifacts" / "real_dialogue_simulator" / stamp)
    try:
        report = asyncio.run(RealDialogueSimulator(args).run())
    except Exception as exc:
        parser.exit(2, f"real dialogue simulator could not start: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "overall_status": report.get("overall_status"),
        "output_dir": args.output_dir,
        "normal_response_count": report.get("scenario", {}).get("normal_response_count"),
        "cancelled_response_count": report.get("scenario", {}).get("cancelled_response_count"),
        "evidence_gaps": report.get("scenario", {}).get("evidence_gaps"),
        "errors": report.get("scenario", {}).get("errors"),
    }, ensure_ascii=False, indent=2))
    return 0 if report.get("overall_status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
