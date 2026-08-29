#!/usr/bin/env python3
"""Independent verifier for P0 reports.

The verifier treats the JSONL trace as the source of truth: request counts,
health counters, timings, RTFs, frame/sample totals and CUDA peaks are all
recomputed here rather than trusted from the runner summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def pstats(events: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(e[key]) for e in events if e.get(key) is not None and math.isfinite(float(e[key]))]
    return {
        "p50": round(float(np.percentile(values, 50)), 3) if values else None,
        "p95": round(float(np.percentile(values, 95)), 3) if values else None,
    }


def same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return actual is not None and expected is not None and abs(float(actual) - float(expected)) <= 1e-9
    return actual == expected


def verify_metrics(report: dict[str, Any], request_events: list[dict[str, Any]], phase_metadata: dict[str, Any], errors: list[str]) -> None:
    metrics = report.get("metrics", {})
    successful = [e for e in request_events if e.get("success")]
    recomputed: dict[str, Any] = {
        "requests": len(request_events),
        "successful_requests": len(successful),
        "success_rate": len(successful) / len(request_events) if request_events else 0.0,
        "nan_count": sum(int(e.get("nan_count", 0)) for e in request_events),
        "inf_count": sum(int(e.get("inf_count", 0)) for e in request_events),
        "empty_audio_responses": sum(bool(e.get("empty_audio")) for e in request_events),
        "uncaught_exceptions": sum(len(e.get("exceptions", [])) for e in request_events),
        "generated_cb0_frames": sum(int(e.get("generated_cb0_frames", 0)) for e in request_events),
        "generated_pcm_samples": sum(int(e.get("generated_pcm_samples", 0)) for e in request_events),
        "cuda_peak_allocated_bytes": max((int(e.get("cuda_peak_allocated_bytes", 0)) for e in request_events), default=0),
        "cuda_peak_reserved_bytes": max((int(e.get("cuda_peak_reserved_bytes", 0)) for e in request_events), default=0),
        "think_time_ms": pstats(successful, "think_time_ms"),
        "v9_time_ms": pstats(successful, "v9_time_ms"),
        "talker_time_ms": pstats(successful, "talker_time_ms"),
        "ttfa_ms": pstats(successful, "ttfa_ms"),
        "e2e_rtf": pstats(successful, "e2e_rtf"),
        "speaker_rtf": pstats(successful, "speaker_rtf"),
    }
    for key, expected in recomputed.items():
        if key not in metrics:
            fail(errors, f"metrica obbligatoria mancante: {key}")
        elif not same(metrics[key], expected):
            fail(errors, f"metrica {key} non coincide col raw trace: {metrics[key]!r} != {expected!r}")
    if metrics.get("cpu_contract") != phase_metadata.get("cpu_contract"):
        fail(errors, "cpu_contract non coincide col record phase_metadata del raw trace")


def verify_one(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        fail(errors, "schema_version non supportata")
    if report.get("phase") != "P0":
        fail(errors, "verify_report supporta attualmente solo P0")
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = report_path.parent / raw
    events: list[dict[str, Any]] = []
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        for line_no, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object")
                events.append(event)
            except Exception as exc:
                fail(errors, f"raw trace line {line_no} invalida: {exc}")
    manifest = Path(report.get("input_manifest", ""))
    if not manifest.is_absolute():
        manifest = report_path.parent / manifest
    if not manifest.is_file():
        fail(errors, f"manifest mancante: {manifest}")
    elif sha256(manifest) != report.get("input_manifest_sha256"):
        fail(errors, "hash manifest non coincide")

    request_events = [e for e in events if e.get("event") == "p0_request"]
    phase_metadata = next((e for e in events if e.get("event") == "phase_metadata"), {})
    if len(request_events) != len(json.loads(manifest.read_text(encoding="utf-8")).get("inputs", [])) if manifest.is_file() else False:
        fail(errors, "numero richieste nel raw trace diverso dal manifest")
    if events:
        verify_metrics(report, request_events, phase_metadata, errors)

    metrics = report.get("metrics", {})
    thresholds = report.get("thresholds", {})
    checks = {
        "cpu_contract_exit_code": metrics.get("cpu_contract", {}).get("exit_code"),
        "success_rate": metrics.get("success_rate"),
        "nan_count": metrics.get("nan_count"),
        "inf_count": metrics.get("inf_count"),
        "empty_audio_responses": metrics.get("empty_audio_responses"),
        "uncaught_exceptions": metrics.get("uncaught_exceptions"),
    }
    for key, actual in checks.items():
        expected = thresholds.get(key)
        if actual is None or expected is None:
            fail(errors, f"threshold non verificabile: {key}")
        elif key == "success_rate":
            if float(actual) < float(expected):
                fail(errors, f"threshold fallita: {key}={actual} < {expected}")
        elif actual != expected:
            fail(errors, f"threshold fallita: {key}={actual} != {expected}")

    independently_passed = not errors
    if report.get("passed") is not independently_passed:
        fail(errors, f"passed non coincide col risultato indipendente: {report.get('passed')} != {independently_passed}")
    return report, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--compare", type=Path, default=None, help="secondo report per il gate di riproducibilità")
    args = ap.parse_args()
    report, errors = verify_one(args.report)
    if args.compare is not None:
        other, other_errors = verify_one(args.compare)
        errors.extend(f"compare: {e}" for e in other_errors)
        for key in ("git_sha", "git_dirty", "input_manifest_sha256", "gpu_uuid",
                    "torch_version", "cuda_version", "transformers_version"):
            if report.get(key) != other.get(key):
                fail(errors, f"riproducibilità fallita: {key} differente")
        if report.get("phase") != other.get("phase") or report.get("phase") != "P0":
            fail(errors, "i report di confronto non sono entrambi P0")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
