#!/usr/bin/env python3
"""Independent Whisper check for paired eager/compiled Qwen end-of-SAY WAVs.

Uses the installed SYSTRAN/faster-whisper API and a local, pinned CTranslate2
snapshot. The runner never downloads weights; it fails if the requested local
snapshot is incomplete. API reference: https://github.com/SYSTRAN/faster-whisper
and https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any

from dialogue_eval.evaluator import calculate_wer


_IT_DIGITS = {
    "zero": "0", "uno": "1", "una": "1", "due": "2", "tre": "3",
    "quattro": "4", "cinque": "5", "sei": "6", "sette": "7",
    "otto": "8", "nove": "9",
}


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Character error rate with the project's Unicode word-character normalization."""
    ref = re.findall(r"\w", str(reference).casefold(), flags=re.UNICODE)
    hyp = re.findall(r"\w", str(hypothesis).casefold(), flags=re.UNICODE)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, start=1):
        current = [i]
        for j, right in enumerate(hyp, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return round(min(1.0, previous[-1] / len(ref)), 4)


def normalize_number_tokens(text: str, expected_digits: str) -> str:
    """Canonicalize only an exact numeric or spoken digit sequence for WER.

    This covers outputs such as ``4829``, ``48 29`` and ``quattro otto due
    nove``. It deliberately does not guess Italian cardinal-number morphology.
    """
    expected = str(expected_digits).strip()
    if not expected.isdigit():
        raise ValueError("expected_digits must contain only decimal digits")
    tokens = re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE)
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isdigit() or token in _IT_DIGITS:
            end = index
            digit_run = ""
            while end < len(tokens) and (tokens[end].isdigit() or tokens[end] in _IT_DIGITS):
                current = tokens[end]
                digit_run += current if current.isdigit() else _IT_DIGITS[current]
                end += 1
            if digit_run == expected:
                normalized.append(f"number{expected}")
            else:
                normalized.extend(tokens[index:end])
            index = end
            continue
        normalized.append(token)
        index += 1
    return " ".join(normalized)


def transcript_metrics(reference: str, hypothesis: str, expected_digits: str) -> dict[str, Any]:
    reference_numbered = normalize_number_tokens(reference, expected_digits)
    hypothesis_numbered = normalize_number_tokens(hypothesis, expected_digits)
    expected_number_token = f"number{expected_digits}"
    return {
        "raw_wer": calculate_wer(reference, hypothesis),
        "raw_cer": calculate_cer(reference, hypothesis),
        "number_normalized_wer": calculate_wer(reference_numbered, hypothesis_numbered),
        "number_normalized_cer": calculate_cer(reference_numbered, hypothesis_numbered),
        "expected_order_number_present": expected_number_token in hypothesis_numbered.split(),
        "number_normalization_scope": "exact digits, adjacent numeric groups, or consecutive Italian single-digit words only",
    }


def read_wav_metadata(path: str | Path) -> dict[str, Any]:
    wav_path = Path(path).expanduser().resolve(strict=True)
    if not wav_path.is_file():
        raise ValueError(f"WAV is not a regular file: {wav_path}")
    with wave.open(str(wav_path), "rb") as handle:
        frames = int(handle.getnframes())
        sample_rate = int(handle.getframerate())
        metadata = {
            "path": str(wav_path),
            "bytes": int(wav_path.stat().st_size),
            "channels": int(handle.getnchannels()),
            "sample_width_bytes": int(handle.getsampwidth()),
            "sample_rate_hz": sample_rate,
            "frames": frames,
            "duration_s": round(frames / sample_rate, 6) if sample_rate > 0 else None,
            "compression": str(handle.getcomptype()),
        }
    if frames <= 0 or sample_rate <= 0:
        raise ValueError(f"WAV has no samples or invalid sample rate: {wav_path}")
    return metadata


def validate_model_snapshot(path: str | Path) -> Path:
    snapshot = Path(path).expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError(f"Whisper snapshot is not a directory: {snapshot}")
    missing = [name for name in ("config.json", "model.bin") if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Whisper snapshot is incomplete; missing {missing}: {snapshot}")
    return snapshot


def make_report(*, reference_text: str, expected_digits: str, model: dict[str, Any],
                sample_results: list[dict[str, Any]], compiled_requested: bool) -> dict[str, Any]:
    eager = next((row for row in sample_results if row.get("role") == "eager"), None)
    compiled = next((row for row in sample_results if row.get("role") == "compiled"), None)
    paired = bool(eager and compiled)
    return {
        "schema": "qwen_eager_compiled_whisper_asr_v1",
        "status": "pair_complete" if paired else "eager_only_compiled_unavailable",
        "scope": "independent offline ASR on actual Qwen speech WAVs; not a human speech-quality rating",
        "backend": "faster-whisper",
        "model": model,
        "transcription": {
            "language": "it",
            "task": "transcribe",
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": True,
        },
        "reference_text": reference_text,
        "expected_order_number": str(expected_digits),
        "samples": sample_results,
        "comparison": {
            "status": "paired" if paired else ("compiled_wav_not_supplied" if not compiled_requested else "compiled_transcription_missing"),
            "same_reference_text": paired,
            "transcript_exact_match": (eager.get("transcript") == compiled.get("transcript")) if paired else None,
            "both_asr_transcripts_contain_expected_number": (
                bool(eager.get("metrics", {}).get("expected_order_number_present"))
                and bool(compiled.get("metrics", {}).get("expected_order_number_present"))
            ) if paired else None,
        },
        "limitations": [
            "WER/CER measure this single TTS phrase with this ASR model, not natural conversation quality.",
            "Number-normalized metrics canonicalize digits, adjacent numeric groups, and consecutive single-digit Italian words; they do not infer full Italian cardinal-number morphology.",
            "The Qwen compile microbenchmark uses fixed code embeddings; this report only compares actual WAVs supplied to this runner.",
        ],
    }


def _atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    extra_pkgs = Path(args.extra_pkgs).expanduser()
    if extra_pkgs.is_dir() and str(extra_pkgs) not in sys.path:
        sys.path.insert(0, str(extra_pkgs))
    model_path = validate_model_snapshot(args.model_path)
    wav_inputs = [("eager", args.eager_wav)]
    if args.compiled_wav:
        wav_inputs.append(("compiled", args.compiled_wav))
    for _, wav_path in wav_inputs:
        read_wav_metadata(wav_path)

    from faster_whisper import WhisperModel

    package_version = importlib.metadata.version("faster-whisper")
    import ctranslate2

    ctranslate2_version = importlib.metadata.version("ctranslate2")
    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(str(model_path), device=args.device, compute_type=compute_type)
    model_metadata = {
        "model_id": "Systran/faster-whisper-small",
        "revision": args.model_revision,
        "snapshot_path": str(model_path),
        "model_bin_bytes": int((model_path / "model.bin").stat().st_size),
        "package_version": package_version,
        "ctranslate2_version": ctranslate2_version,
        "device": args.device,
        "compute_type": compute_type,
        "official_api": "https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py",
        "model_source": "https://huggingface.co/Systran/faster-whisper-small/tree/" + args.model_revision,
    }
    results: list[dict[str, Any]] = []
    report_path = Path(args.output)
    for role, wav_path in wav_inputs:
        metadata = read_wav_metadata(wav_path)
        started = time.perf_counter()
        segments, info = model.transcribe(
            str(Path(wav_path).expanduser().resolve()),
            language="it",
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        result = {
            "role": role,
            "audio": metadata,
            "transcript": transcript,
            "language_detected": getattr(info, "language", None),
            "transcription_seconds": round(time.perf_counter() - started, 3),
            "metrics": transcript_metrics(args.reference_text, transcript, args.expected_digits),
        }
        results.append(result)
        _atomic_write(report_path, make_report(
            reference_text=args.reference_text,
            expected_digits=args.expected_digits,
            model=model_metadata,
            sample_results=results,
            compiled_requested=bool(args.compiled_wav),
        ))
    return make_report(
        reference_text=args.reference_text,
        expected_digits=args.expected_digits,
        model=model_metadata,
        sample_results=results,
        compiled_requested=bool(args.compiled_wav),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-wav", required=True)
    parser.add_argument("--compiled-wav")
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--expected-digits", default="4829")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default="536b0662742c02347bc0e980a01041f333bce120")
    parser.add_argument("--extra-pkgs", default="/workspace/.extra_pkgs")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] in {"pair_complete", "eager_only_compiled_unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
