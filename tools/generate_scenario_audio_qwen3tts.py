#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and independently ASR-check the canonical scenario input WAVs.

Run inside the RTX 3090 Docker image. Qwen generation is delegated to the
project's isolated `/opt/venv_phase1` (qwen-tts / Transformers 4.57.3); ASR runs
in the main image. Both model snapshots must already be cached locally.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Optional

import numpy as np
import soundfile as sf
import torch

WORKSPACE = Path("/workspace")
sys.path.insert(0, str(WORKSPACE))
DEFAULT_OUTPUT_DIR = WORKSPACE / "artifacts" / "scenario_audio_qwen3tts"
MODEL_CACHE = WORKSPACE / ".hf_cache"
BASE_MODEL_GLOB = str(MODEL_CACHE / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "snapshots" / "*")
PHASE1_PYTHON = Path("/opt/venv_phase1/bin/python")
PHASE1_WORKER = WORKSPACE / "tools" / "generate_scenario_audio_qwen3tts_phase1_worker.py"
WHISPER_MODEL_PATTERNS = (
    str(MODEL_CACHE / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "snapshots" / "*"),
    "/home/administrator/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*",
    str(MODEL_CACHE / "models--Systran--faster-whisper-small" / "snapshots" / "*"),
    str(MODEL_CACHE / "hub" / "models--Systran--faster-whisper-small" / "snapshots" / "*"),
)
SAMPLE_RATE_INPUT = 16_000
ASR_WER_LIMIT = 0.15
ITALIAN_DIGIT_WORDS = {
    "0": "zero",
    "1": "uno",
    "2": "due",
    "3": "tre",
    "4": "quattro",
    "5": "cinque",
    "6": "sei",
    "7": "sette",
    "8": "otto",
    "9": "nove",
}
ITALIAN_NUMBER_WORDS = {word: digit for digit, word in ITALIAN_DIGIT_WORDS.items()}
ITALIAN_NUMBER_WORDS.update({"venti": "20", "trenta": "30"})


def _find_local_snapshot(patterns: Iterable[str], description: str) -> Path:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    raise FileNotFoundError(f"No local {description} snapshot found; network downloads are disabled")


def _language_for(scenario_id: str, step_id: str) -> str:
    if scenario_id == "03_instruction_change" and step_id == "turn_2_user":
        return "English"
    return "Italian"


def _spoken_text_for_tts(expected_text: str, language: str) -> str:
    """Speak long Italian numeric identifiers digit-by-digit, preserving the canonical reference."""
    if language.casefold() not in {"italian", "it"}:
        return expected_text

    def spell_digits(match: re.Match[str]) -> str:
        return ", ".join(ITALIAN_DIGIT_WORDS[digit] for digit in match.group(0))

    return re.sub(r"(?<!\d)\d{3,}(?!\d)", spell_digits, expected_text)


def _numeric_normalized_text(text: str) -> str:
    """Normalize Italian number words and spoken digit runs for a secondary WER."""
    tokens = re.findall(r"\d+|[a-zà-öø-ÿ]+", text.casefold())
    normalized = [ITALIAN_NUMBER_WORDS.get(token, token) for token in tokens]
    collapsed: list[str] = []
    index = 0
    while index < len(normalized):
        if len(normalized[index]) == 1 and normalized[index].isdigit():
            end = index
            digits = []
            while end < len(normalized) and len(normalized[end]) == 1 and normalized[end].isdigit():
                digits.append(normalized[end])
                end += 1
            collapsed.append("".join(digits))
            index = end
        else:
            collapsed.append(normalized[index])
            index += 1
    return " ".join(collapsed)


def _load_asr(asr_model_path: Path) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(str(asr_model_path), device="cuda", compute_type="float16")


def _transcribe(asr_model: Any, path: Path, language: str) -> str:
    language_code = {"italian": "it", "it": "it", "english": "en", "en": "en"}.get(language.casefold())
    segments, _info = asr_model.transcribe(str(path), language=language_code, beam_size=1)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _audio_metrics(pcm: np.ndarray) -> Dict[str, Any]:
    from dialogue_eval.audio_metrics import compute_physical_audio_metrics

    return compute_physical_audio_metrics(pcm, sample_rate=SAMPLE_RATE_INPUT)


def _run_phase1_worker(
    *,
    jobs: list[dict[str, str]],
    model_snapshot: Path,
    reference_audio: Path,
    output_dir: Path,
) -> None:
    if not PHASE1_PYTHON.is_file():
        raise FileNotFoundError(f"Isolated Qwen-TTS Python environment not found: {PHASE1_PYTHON}")
    if not PHASE1_WORKER.is_file():
        raise FileNotFoundError(f"Qwen-TTS Phase 1 worker not found: {PHASE1_WORKER}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen3tts_jobs_") as temp_dir:
        jobs_path = Path(temp_dir) / "jobs.json"
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            str(PHASE1_PYTHON),
            str(PHASE1_WORKER),
            "--model-snapshot",
            str(model_snapshot),
            "--reference-audio",
            str(reference_audio),
            "--jobs-json",
            str(jobs_path),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "256",
        ]
        print("[qwen3tts] Running isolated Phase 1 worker (Transformers 4.57.3)", flush=True)
        subprocess.run(command, cwd=WORKSPACE, env=environment, check=True)

    missing = [job["filename"] for job in jobs if not (output_dir / job["filename"]).is_file()]
    if missing:
        raise RuntimeError(f"Qwen-TTS worker exited without creating expected WAVs: {missing}")


def _inspect_clip(
    *,
    asr: Any,
    out_dir: Path,
    filename: str,
    scenario_id: str,
    step_id: str,
    expected_text: str,
    spoken_text: str,
    language: str,
) -> Dict[str, Any]:
    output_path = out_dir / filename
    saved_pcm, saved_sr = sf.read(str(output_path), dtype="float32")
    if saved_sr != SAMPLE_RATE_INPUT or saved_pcm.ndim != 1:
        raise RuntimeError(f"Saved scenario WAV is not mono 16 kHz: {output_path}")
    if saved_pcm.size == 0 or not np.isfinite(saved_pcm).all():
        raise RuntimeError(f"Saved scenario WAV is empty or non-finite: {output_path}")

    transcript = _transcribe(asr, output_path, language)
    metrics = _audio_metrics(saved_pcm)
    from dialogue_eval.evaluator import calculate_wer, contains_italian_digit_sequence

    expected_codes = re.findall(r"(?<!\d)\d{3,}(?!\d)", expected_text)
    code_checks = {
        code: contains_italian_digit_sequence(transcript, code)
        for code in expected_codes
    }
    expected_marco = bool(re.search(r"\bmarco\b", expected_text, flags=re.IGNORECASE))
    contains_marco = bool(re.search(r"\bmarco\b", transcript, flags=re.IGNORECASE))
    expected_code_4829 = contains_italian_digit_sequence(expected_text, "4829")
    contains_code_4829 = contains_italian_digit_sequence(transcript, "4829")
    wer = calculate_wer(expected_text, transcript)
    numeric_normalized_wer = calculate_wer(
        _numeric_normalized_text(expected_text),
        _numeric_normalized_text(transcript),
    )
    asr_quality_passed = bool(
        numeric_normalized_wer < ASR_WER_LIMIT
        and (not expected_marco or contains_marco)
        and all(code_checks.values())
        and (not expected_code_4829 or contains_code_4829)
        and metrics["finite"]
        and metrics["rms"] > 0.0
        and metrics["clipping_pct"] == 0.0
    )

    return {
        "scenario_id": scenario_id,
        "step_id": step_id,
        "filename": filename,
        "audio_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        # `text` stays the canonical scenario reference for existing consumers.
        "text": expected_text,
        "spoken_text": spoken_text,
        "language": language,
        "sample_rate_hz": int(saved_sr),
        "asr_engine": "faster-whisper",
        "asr_language": language,
        "asr_transcript": transcript,
        "wer": wer,
        "numeric_normalized_wer": numeric_normalized_wer,
        "contains_marco": contains_marco,
        "expected_numeric_codes": code_checks,
        "contains_code_4829": contains_code_4829,
        "asr_quality_passed": asr_quality_passed,
        "audio_metrics": metrics,
    }


def _force_offline_model_loading() -> None:
    """Forbid Hub fallbacks even if the parent environment requested online access."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _path_has_symlink_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return any(component.is_symlink() for component in (absolute, *absolute.parents))


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = Path(os.path.abspath(output_dir))
    if _path_has_symlink_component(output_dir):
        raise ValueError(f"output directory contains a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(output_dir):
        raise ValueError(f"output directory became symlinked during setup: {output_dir}")
    return output_dir


def _snapshot_verified_audio_cache(
    source_dir: Path,
    snapshot_dir: Path,
    expected_hashes: Dict[str, str],
) -> Path:
    """Copy exactly the verified WAV bytes into a run-owned directory."""
    source_root = Path(os.path.abspath(source_dir))
    snapshot_root = Path(os.path.abspath(snapshot_dir))
    if _path_has_symlink_component(source_root):
        raise ValueError(f"cache root contains a symlink: {source_root}")
    if _path_has_symlink_component(snapshot_root):
        raise ValueError(f"snapshot path contains a symlink: {snapshot_root}")
    if (
        source_root == snapshot_root
        or source_root in snapshot_root.parents
        or snapshot_root in source_root.parents
    ):
        raise ValueError("audio snapshot and mutable source cache must not contain each other")
    payloads: Dict[str, bytes] = {}
    for filename, expected_digest in expected_hashes.items():
        if Path(filename).name != filename or "/" in filename or "\\" in filename or not filename.endswith(".wav"):
            raise ValueError(f"unsafe cache snapshot filename: {filename!r}")
        source_path = source_root / filename
        if _path_has_symlink_component(source_path) or not source_path.is_file():
            raise ValueError(f"cache snapshot source is missing or symlinked: {source_path}")
        payload = source_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise RuntimeError(f"cache changed after verification: {filename}")
        payloads[filename] = payload
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        target_path = snapshot_root / filename
        target_path.write_bytes(payload)
        target_path.chmod(0o444)
        if hashlib.sha256(target_path.read_bytes()).hexdigest() != expected_hashes[filename]:
            raise RuntimeError(f"audio snapshot digest mismatch: {filename}")
    return snapshot_root


def _manifest_is_complete(manifest: Dict[str, Any]) -> bool:
    if not isinstance(manifest, dict):
        return False
    clips = manifest.get("clips")
    kickoff = manifest.get("mode_b_kickoff")

    def _record_has_required_shape(record: Any) -> bool:
        return bool(
            isinstance(record, dict)
            and all(
                isinstance(record.get(key), str) and record[key].strip()
                for key in (
                    "filename", "scenario_id", "step_id", "text", "spoken_text", "language",
                    "asr_engine", "asr_language", "asr_transcript", "audio_sha256",
                )
            )
            and re.fullmatch(r"[0-9a-f]{64}", record["audio_sha256"]) is not None
            and record.get("sample_rate_hz") == SAMPLE_RATE_INPUT
            and record.get("asr_quality_passed") is True
            and isinstance(record.get("audio_metrics"), dict)
        )

    return bool(
        manifest.get("schema_version") == 2
        and manifest.get("generator") == "Qwen3-TTS-12Hz-1.7B-Base"
        and isinstance(manifest.get("model_snapshot"), str)
        and bool(manifest["model_snapshot"].strip())
        and isinstance(manifest.get("asr_model_snapshot"), str)
        and bool(manifest["asr_model_snapshot"].strip())
        and manifest.get("asr_engine") == "faster-whisper"
        and manifest.get("sample_rate_hz") == SAMPLE_RATE_INPUT
        and isinstance(manifest.get("mode_b_kickoff_text"), str)
        and bool(manifest["mode_b_kickoff_text"].strip())
        and manifest.get("status") == "complete"
        and manifest.get("pretest_passed") is True
        and manifest.get("cache_complete") is True
        and manifest.get("all_audio_asr_passed") is True
        and isinstance(clips, list)
        and len(clips) == 11
        and all(_record_has_required_shape(clip) for clip in clips)
        and len({clip["filename"] for clip in clips}) == len(clips)
        and _record_has_required_shape(kickoff)
        and kickoff.get("filename") == "mode_b_kickoff.wav"
        and kickoff.get("scenario_id") == "mode_b"
        and kickoff.get("step_id") == "kickoff"
        and kickoff["filename"] not in {clip["filename"] for clip in clips}
    )


def _validate_cache_clip(
    clip: Dict[str, Any],
    *,
    expected_text: str,
    audio_path: Path,
    independent_transcript: Optional[str] = None,
    expected_scenario_id: Optional[str] = None,
    expected_step_id: Optional[str] = None,
) -> list[str]:
    """Revalidate manifest claims against canonical text and the exact WAV bytes."""
    errors: list[str] = []
    if clip.get("text") != expected_text:
        errors.append("canonical text mismatch")

    language = clip.get("language")
    if not isinstance(language, str) or not language.strip():
        errors.append("missing language")
        language = ""
    if clip.get("spoken_text") != _spoken_text_for_tts(expected_text, language):
        errors.append("spoken_text is inconsistent with the canonical text/language")
    if clip.get("filename") != audio_path.name:
        errors.append("filename mismatch")
    if expected_scenario_id is not None and clip.get("scenario_id") != expected_scenario_id:
        errors.append("scenario_id mismatch")
    if expected_step_id is not None and clip.get("step_id") != expected_step_id:
        errors.append("step_id mismatch")
    if clip.get("asr_engine") != "faster-whisper":
        errors.append("independent ASR engine mismatch")
    if clip.get("asr_language") != language:
        errors.append("ASR language metadata mismatch")
    if clip.get("sample_rate_hz") != SAMPLE_RATE_INPUT:
        errors.append("manifest sample rate is not 16000 Hz")
    transcript = clip.get("asr_transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        errors.append("ASR transcript is missing")
        transcript = ""

    from dialogue_eval.evaluator import calculate_wer, contains_italian_digit_sequence

    try:
        raw_wer = float(clip.get("wer"))
    except (TypeError, ValueError):
        raw_wer = math.nan
    try:
        numeric_wer = float(clip.get("numeric_normalized_wer"))
    except (TypeError, ValueError):
        numeric_wer = math.nan
    if not math.isfinite(raw_wer) or raw_wer < 0.0:
        errors.append("raw WER is not finite and non-negative")
    if not math.isfinite(numeric_wer) or numeric_wer < 0.0:
        errors.append("numeric_normalized_wer is not finite and non-negative")
    else:
        if numeric_wer >= ASR_WER_LIMIT:
            errors.append("numeric-normalized WER exceeds the ASR limit")
        recomputed_numeric_wer = calculate_wer(
            _numeric_normalized_text(expected_text),
            _numeric_normalized_text(transcript),
        )
        if numeric_wer != recomputed_numeric_wer:
            errors.append("numeric-normalized WER does not match the stored transcript")
    if math.isfinite(raw_wer) and raw_wer != calculate_wer(expected_text, transcript):
        errors.append("raw WER does not match the stored transcript")

    expected_marco = bool(re.search(r"\bmarco\b", expected_text, flags=re.IGNORECASE))
    transcript_has_marco = bool(re.search(r"\bmarco\b", transcript, flags=re.IGNORECASE))
    if clip.get("contains_marco") is not transcript_has_marco:
        errors.append("contains_marco metadata does not match the ASR transcript")
    if expected_marco and not transcript_has_marco:
        errors.append("ASR transcript is missing Marco")

    expected_codes = re.findall(r"(?<!\d)\d{3,}(?!\d)", expected_text)
    expected_code_checks = {
        code: contains_italian_digit_sequence(transcript, code)
        for code in expected_codes
    }
    if clip.get("expected_numeric_codes") != expected_code_checks:
        errors.append("numeric-code metadata does not match the ASR transcript")
    if not all(expected_code_checks.values()):
        errors.append("ASR transcript is missing one or more expected numeric codes")
    contains_4829 = contains_italian_digit_sequence(transcript, "4829")
    if clip.get("contains_code_4829") is not contains_4829:
        errors.append("contains_code_4829 metadata does not match the ASR transcript")
    if contains_italian_digit_sequence(expected_text, "4829") and not contains_4829:
        errors.append("ASR transcript is missing code 4829")

    if independent_transcript is not None:
        if not isinstance(independent_transcript, str) or not independent_transcript.strip():
            errors.append("fresh independent ASR returned an empty transcript for the WAV")
        else:
            independent_marco = bool(re.search(r"\bmarco\b", independent_transcript, flags=re.IGNORECASE))
            independent_codes = {
                code: contains_italian_digit_sequence(independent_transcript, code)
                for code in expected_codes
            }
            independent_raw_wer = calculate_wer(expected_text, independent_transcript)
            independent_wer = calculate_wer(
                _numeric_normalized_text(expected_text),
                _numeric_normalized_text(independent_transcript),
            )
            if not math.isfinite(independent_raw_wer) or independent_raw_wer < 0.0:
                errors.append("fresh independent ASR raw WER is not finite and non-negative")
            if not math.isfinite(independent_wer) or independent_wer >= ASR_WER_LIMIT:
                errors.append("fresh independent ASR numeric-normalized WER exceeds the ASR limit")
            if expected_marco and not independent_marco:
                errors.append("fresh independent ASR transcript is missing Marco")
            if not all(independent_codes.values()):
                errors.append("fresh independent ASR transcript is missing an expected numeric code")
            if (
                contains_italian_digit_sequence(expected_text, "4829")
                and not contains_italian_digit_sequence(independent_transcript, "4829")
            ):
                errors.append("fresh independent ASR transcript is missing code 4829")

    metrics = clip.get("audio_metrics")
    if not isinstance(metrics, dict) or metrics.get("finite") is not True:
        errors.append("audio metrics are absent or marked non-finite")
        metrics = {}
    try:
        manifest_rms = float(metrics.get("rms"))
    except (TypeError, ValueError):
        manifest_rms = math.nan
    try:
        manifest_clipping = float(metrics.get("clipping_pct"))
    except (TypeError, ValueError):
        manifest_clipping = math.nan
    if not math.isfinite(manifest_rms) or manifest_rms <= 0.0:
        errors.append("manifest RMS is not finite and positive")
    if not math.isfinite(manifest_clipping) or manifest_clipping != 0.0:
        errors.append("manifest audio clipping is nonzero or invalid")

    audio_finite = False
    audio_rms = math.nan
    audio_clipping = math.nan
    if not audio_path.is_file():
        errors.append("WAV file is missing")
    elif _path_has_symlink_component(audio_path) or audio_path.resolve().parent != audio_path.parent.resolve():
        errors.append("WAV path is a symlink or escapes the cache directory")
    else:
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if not isinstance(clip.get("audio_sha256"), str) or digest != clip["audio_sha256"]:
            errors.append("WAV SHA-256 does not match the manifest")
        try:
            pcm, sample_rate = sf.read(str(audio_path), dtype="float32")
            if sample_rate != SAMPLE_RATE_INPUT or clip.get("sample_rate_hz") != sample_rate:
                errors.append("WAV sample rate is not 16000 Hz")
            if pcm.ndim != 1 or pcm.size == 0:
                errors.append("WAV is not non-empty mono audio")
            audio_finite = bool(np.isfinite(pcm).all())
            if not audio_finite:
                errors.append("WAV contains non-finite samples")
            if audio_finite and pcm.size:
                actual_metrics = _audio_metrics(pcm)
                if set(metrics) != set(actual_metrics):
                    errors.append("audio_metrics fields do not match the recomputed metric schema")
                for metric_name, actual_value in actual_metrics.items():
                    manifest_value = metrics.get(metric_name)
                    if isinstance(actual_value, bool):
                        if manifest_value is not actual_value:
                            errors.append(f"audio_metrics.{metric_name} does not match the WAV")
                    elif isinstance(actual_value, int):
                        if type(manifest_value) is not int or manifest_value != actual_value:
                            errors.append(f"audio_metrics.{metric_name} does not match the WAV")
                    else:
                        try:
                            numeric_manifest_value = float(manifest_value)
                        except (TypeError, ValueError):
                            numeric_manifest_value = math.nan
                        if (
                            not math.isfinite(numeric_manifest_value)
                            or not math.isclose(numeric_manifest_value, float(actual_value), rel_tol=0.0, abs_tol=1e-6)
                        ):
                            errors.append(f"audio_metrics.{metric_name} does not match the WAV")
                audio_rms = float(actual_metrics.get("rms", math.nan))
                audio_clipping = float(actual_metrics.get("clipping_pct", math.nan))
                if not math.isfinite(audio_rms) or audio_rms <= 0.0:
                    errors.append("WAV RMS is not finite and positive")
                if not math.isfinite(audio_clipping) or audio_clipping != 0.0:
                    errors.append("WAV clipping is nonzero or invalid")
        except Exception as exc:
            errors.append(f"WAV validation failed: {type(exc).__name__}: {exc}")

    expected_asr_passed = bool(
        math.isfinite(numeric_wer)
        and numeric_wer < ASR_WER_LIMIT
        and (not expected_marco or transcript_has_marco)
        and all(expected_code_checks.values())
        and (not contains_italian_digit_sequence(expected_text, "4829") or contains_4829)
        and audio_finite
        and math.isfinite(audio_rms)
        and audio_rms > 0.0
        and math.isfinite(audio_clipping)
        and audio_clipping == 0.0
    )
    if clip.get("asr_quality_passed") is not expected_asr_passed:
        errors.append("asr_quality_passed metadata does not match recomputed gates")
    if not expected_asr_passed:
        errors.append("ASR/audio quality gate failed")
    return errors


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_targets() -> list[tuple[str, str, str, str]]:
    from dialogue_eval.scenarios import ALL_SCENARIOS

    targets = [
        (scenario.scenario_id, step.step_id, step.expected_text, _language_for(scenario.scenario_id, step.step_id))
        for scenario in ALL_SCENARIOS
        for step in scenario.steps
        if step.speaker == "user"
    ]
    if len(targets) != 11 or targets[0][2] != (
        "Buongiorno, mi chiamo Marco e il mio codice ordine è 4829. Ricordalo per favore."
    ):
        raise RuntimeError("The canonical name/number smoke-test utterance changed; review its ASR gate before proceeding")
    return targets


def _pretest_passed(clip: Dict[str, Any]) -> bool:
    """Require exact facts/audio quality while allowing ASR digit-token segmentation."""
    return bool(
        float(clip.get("numeric_normalized_wer", 1.0)) < ASR_WER_LIMIT
        and clip.get("contains_marco") is True
        and clip.get("contains_code_4829") is True
        and clip.get("asr_quality_passed") is True
    )


def _make_job(filename: str, expected_text: str, language: str) -> dict[str, str]:
    return {
        "filename": filename,
        "text": _spoken_text_for_tts(expected_text, language),
        "language": language,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Generate and ASR-check only the first canonical user utterance")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-audio", type=Path, default=None)
    parser.add_argument("--asr-model", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir = _prepare_output_dir(args.output_dir)
    manifest_path = args.output_dir / "scenario_audio_manifest.json"
    # Invalidate any previous successful cache before model discovery or generation
    # can fail; the proof runner must never accept stale evidence after a new attempt.
    _save_manifest(
        manifest_path,
        {
            "schema_version": 2,
            "generator": "Qwen3-TTS-12Hz-1.7B-Base",
            "pretest_passed": False,
            "cache_complete": False,
            "all_audio_asr_passed": False,
            "status": "generation_in_progress",
            "clips": [],
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS scenario generation requires CUDA")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_snapshot = _find_local_snapshot((BASE_MODEL_GLOB,), "Qwen3-TTS 12Hz 1.7B Base")
    reference_candidates = [
        args.reference_audio,
        WORKSPACE / "samples_v9" / "sample_1_native_qwen_reference.wav",
        WORKSPACE / "samples_kernel_final" / "kernel_bench_00.wav",
    ]
    reference_audio = next((path for path in reference_candidates if path is not None and path.is_file()), None)
    if reference_audio is None:
        raise FileNotFoundError("No local Qwen voice-cloning reference WAV was found")

    asr_snapshot = args.asr_model or _find_local_snapshot(WHISPER_MODEL_PATTERNS, "faster-whisper")
    print(f"[qwen3tts] Loading local independent ASR: {asr_snapshot}", flush=True)
    asr = _load_asr(asr_snapshot)

    from dialogue_eval.scenarios import ALL_SCENARIOS, MODE_B_KICKOFF_TEXT

    targets = _build_targets()
    first_scenario, first_step, first_text, first_language = targets[0]
    first_filename = f"{first_scenario}__{first_step}.wav"
    first_spoken_text = _spoken_text_for_tts(first_text, first_language)
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "generator": "Qwen3-TTS-12Hz-1.7B-Base",
        "qwen_runtime": "qwen-tts 0.1.1 / Transformers 4.57.3 in /opt/venv_phase1",
        "model_snapshot": str(model_snapshot),
        "reference_audio": str(reference_audio),
        "speaker_conditioning": "x_vector_only_mode=True",
        "generation_mode": "non_streaming_mode=True",
        "max_new_tokens": 256,
        "number_pronunciation": "Italian numeric identifiers with 3+ digits are spoken digit-by-digit; canonical text is unchanged",
        "asr_wer_normalization": "Raw WER is preserved; a separately reported WER maps Italian number words and contiguous spoken digit tokens to numeric tokens",
        "asr_engine": "faster-whisper",
        "asr_model_snapshot": str(asr_snapshot),
        "sample_rate_hz": SAMPLE_RATE_INPUT,
        "mode_b_kickoff_text": MODE_B_KICKOFF_TEXT,
        "status": "generation_in_progress",
        "clips": [],
        "cache_complete": False,
        "all_audio_asr_passed": False,
    }
    _save_manifest(manifest_path, manifest)

    _run_phase1_worker(
        jobs=[_make_job(first_filename, first_text, first_language)],
        model_snapshot=model_snapshot,
        reference_audio=reference_audio,
        output_dir=args.output_dir,
    )
    first_clip = _inspect_clip(
        asr=asr,
        out_dir=args.output_dir,
        filename=first_filename,
        scenario_id=first_scenario,
        step_id=first_step,
        expected_text=first_text,
        spoken_text=first_spoken_text,
        language=first_language,
    )
    manifest["clips"].append(first_clip)
    manifest["pretest_passed"] = _pretest_passed(first_clip)
    _save_manifest(manifest_path, manifest)
    print(json.dumps({"pretest": first_clip, "pretest_passed": manifest["pretest_passed"]}, ensure_ascii=False), flush=True)

    if not manifest["pretest_passed"]:
        manifest["status"] = "pretest_failed"
        _save_manifest(manifest_path, manifest)
        print("[qwen3tts] PRETEST FAILED: stopping before generating the remaining cache", file=sys.stderr)
        return 2
    if args.smoke:
        manifest["status"] = "smoke_passed_cache_incomplete"
        _save_manifest(manifest_path, manifest)
        print(f"[qwen3tts] SMOKE PASS; cache contains only {first_clip['filename']}", flush=True)
        return 0

    remaining_jobs: list[dict[str, str]] = []
    remaining_targets: list[tuple[str, str, str, str, str, str]] = []
    for scenario in ALL_SCENARIOS:
        for step in scenario.steps:
            if step.speaker != "user" or (scenario.scenario_id == first_scenario and step.step_id == first_step):
                continue
            language = _language_for(scenario.scenario_id, step.step_id)
            filename = f"{scenario.scenario_id}__{step.step_id}.wav"
            remaining_jobs.append(_make_job(filename, step.expected_text, language))
            remaining_targets.append((scenario.scenario_id, step.step_id, filename, step.expected_text, _spoken_text_for_tts(step.expected_text, language), language))

    kickoff_filename = "mode_b_kickoff.wav"
    remaining_jobs.append(_make_job(kickoff_filename, MODE_B_KICKOFF_TEXT, "Italian"))
    remaining_targets.append(("mode_b", "kickoff", kickoff_filename, MODE_B_KICKOFF_TEXT, MODE_B_KICKOFF_TEXT, "Italian"))
    _run_phase1_worker(
        jobs=remaining_jobs,
        model_snapshot=model_snapshot,
        reference_audio=reference_audio,
        output_dir=args.output_dir,
    )

    for scenario_id, step_id, filename, expected_text, spoken_text, language in remaining_targets:
        clip = _inspect_clip(
            asr=asr,
            out_dir=args.output_dir,
            filename=filename,
            scenario_id=scenario_id,
            step_id=step_id,
            expected_text=expected_text,
            spoken_text=spoken_text,
            language=language,
        )
        if scenario_id == "mode_b":
            manifest["mode_b_kickoff"] = clip
        else:
            manifest["clips"].append(clip)
        _save_manifest(manifest_path, manifest)
        print(json.dumps({"clip": clip}, ensure_ascii=False), flush=True)

    expected_clip_count = sum(step.speaker == "user" for scenario in ALL_SCENARIOS for step in scenario.steps)
    manifest["cache_complete"] = bool(len(manifest["clips"]) == expected_clip_count and "mode_b_kickoff" in manifest)
    all_clips = manifest["clips"] + [manifest["mode_b_kickoff"]]
    manifest["all_audio_asr_passed"] = bool(manifest["cache_complete"] and all(clip["asr_quality_passed"] for clip in all_clips))
    _save_manifest(manifest_path, manifest)

    if not manifest["cache_complete"]:
        raise RuntimeError("Scenario audio cache is missing one or more canonical user inputs")
    if not manifest["all_audio_asr_passed"]:
        manifest["status"] = "asr_quality_failed"
        _save_manifest(manifest_path, manifest)
        failed = [clip["filename"] for clip in all_clips if not clip["asr_quality_passed"]]
        print(f"[qwen3tts] ASR QUALITY GATE FAILED for: {failed}", file=sys.stderr)
        return 3

    manifest["status"] = "complete"
    _save_manifest(manifest_path, manifest)
    print(f"[qwen3tts] CACHE COMPLETE: {len(manifest['clips'])} Mode A utterances + Mode B kickoff", flush=True)
    print(f"[qwen3tts] Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
