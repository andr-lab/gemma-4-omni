#!/usr/bin/env python3
"""Independent verifier for a phase report and its raw trace."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--compare", type=Path, default=None, help="second report per il gate di riproducibilità")
    args = ap.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    if report.get("phase") != "P0":
        fail(errors, "verify_report supporta attualmente solo P0")

    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = args.report.parent / raw
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
        events: list[dict[str, Any]] = []
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        events = []
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
        manifest = args.report.parent / manifest
    if not manifest.is_file():
        fail(errors, f"manifest mancante: {manifest}")
    elif sha256(manifest) != report.get("input_manifest_sha256"):
        fail(errors, "hash manifest non coincide")

    metrics = report.get("metrics", {})
    required = ["requests", "successful_requests", "success_rate", "nan_count", "inf_count",
                "empty_audio_responses", "uncaught_exceptions", "think_time_ms", "v9_time_ms",
                "talker_time_ms", "ttfa_ms", "speaker_rtf", "generated_cb0_frames",
                "generated_pcm_samples", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"]
    for key in required:
        if key not in metrics:
            fail(errors, f"metrica obbligatoria mancante: {key}")
    if events:
        success = [e for e in events if e.get("success")]
        recomputed = {
            "requests": len(events),
            "successful_requests": len(success),
            "success_rate": len(success) / len(events),
            "nan_count": sum(int(e.get("nan_count", 0)) for e in events),
            "inf_count": sum(int(e.get("inf_count", 0)) for e in events),
            "empty_audio_responses": sum(bool(e.get("empty_audio")) for e in events),
            "uncaught_exceptions": sum(len(e.get("exceptions", [])) for e in events),
        }
        for key, value in recomputed.items():
            actual = metrics.get(key)
            if key == "success_rate":
                if actual is None or abs(float(actual) - value) > 1e-12:
                    fail(errors, f"metrica {key} non ricalcolata dal raw trace")
            elif actual != value:
                fail(errors, f"metrica {key} non ricalcolata dal raw trace: {actual} != {value}")

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
    if report.get("passed") is not True:
        fail(errors, "report dichiara passed=false")
    if args.compare is not None:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        for key in ("git_sha", "git_dirty", "input_manifest_sha256"):
            if report.get(key) != other.get(key):
                fail(errors, f"riproducibilità fallita: {key} differente")
        if other.get("phase") != "P0":
            fail(errors, "il report di confronto non è P0")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
