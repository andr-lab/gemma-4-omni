#!/usr/bin/env python3
"""Generate scenario WAVs in the isolated Qwen/Transformers Phase 1 environment.

Run with `/opt/venv_phase1/bin/python` inside the RTX 3090 container. This keeps
Qwen-TTS (Transformers 4.57.3) separate from the main serving image's
Transformers 5.x installation.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
from pathlib import Path
import secrets
import sys
import types
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def _install_audio_import_shims() -> None:
    try:
        import torchaudio  # noqa: F401
    except Exception:
        module = types.ModuleType("torchaudio")
        module.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        sys.modules["torchaudio"] = module
        sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")
        sys.modules["torchaudio.compliance"] = types.ModuleType("torchaudio.compliance")
        sys.modules["torchaudio.compliance.kaldi"] = types.ModuleType("torchaudio.compliance.kaldi")
    sys.modules["sox"] = types.ModuleType("sox")


def _validate_jobs(jobs: Any) -> list[dict[str, str]]:
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs JSON must be a non-empty list")
    validated = []
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        filename, text, language = job.get("filename"), job.get("text"), job.get("language")
        if not all(isinstance(value, str) and value.strip() for value in (filename, text, language)):
            raise ValueError("each job requires non-empty filename, text, and language")
        if (
            Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or not filename.endswith(".wav")
        ):
            raise ValueError(f"unsafe or invalid output filename: {filename!r}")
        if filename in seen:
            raise ValueError(f"duplicate output filename: {filename}")
        seen.add(filename)
        validated.append({"filename": filename, "text": text, "language": language})
    return validated


def _load_tts(snapshot: Path) -> Any:
    _install_audio_import_shims()
    from qwen_tts import Qwen3TTSModel

    print(f"[qwen3tts-phase1] Loading local model {snapshot}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        str(snapshot),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    speech_tokenizer = getattr(tts.model, "speech_tokenizer", None)
    if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
        speech_tokenizer.model = speech_tokenizer.model.to("cuda:0")
        speech_tokenizer.device = torch.device("cuda:0")
    return tts


def _safe_output_dir(output_dir: Path) -> Path:
    root = Path(os.path.abspath(output_dir))
    if any(component.is_symlink() for component in (root, *root.parents)):
        raise ValueError(f"output directory contains a symlink: {root}")
    return root


def _open_safe_output_dir_fd(output_dir: Path) -> int | None:
    root = _safe_output_dir(output_dir)
    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.rename not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
    ):
        return None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(root.anchor, flags)
    try:
        for component in root.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except OSError as exc:
        os.close(directory_fd)
        raise ValueError(f"output directory contains an unsafe path component: {root}") from exc


def _safe_output_path(output_dir: Path, filename: str) -> Path:
    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.endswith(".wav")
    ):
        raise ValueError(f"unsafe or invalid output filename: {filename!r}")
    root = _safe_output_dir(output_dir)
    output_path = root / filename
    if output_path.is_symlink():
        raise ValueError(f"refusing symlink output path: {output_path}")
    try:
        output_path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path escapes output directory: {output_path}") from exc
    return output_path


def _write_pcm_atomically(
    output_path: Path,
    pcm: np.ndarray,
    *,
    output_dir_fd: int | None = None,
) -> None:
    temporary_name = ""
    temporary_path: Path | None = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(10):
        candidate = f".{output_path.stem}_{secrets.token_hex(12)}.tmp.wav"
        try:
            if output_dir_fd is not None:
                file_descriptor = os.open(candidate, flags, 0o600, dir_fd=output_dir_fd)
            else:
                temporary_path = output_path.parent / candidate
                file_descriptor = os.open(temporary_path, flags, 0o600)
            temporary_name = candidate
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"could not allocate a unique temporary WAV for {output_path.name}")

    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            sf.write(temporary_file, pcm, 16_000, format="WAV", subtype="PCM_16")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if output_dir_fd is not None:
            os.rename(
                temporary_name,
                output_path.name,
                src_dir_fd=output_dir_fd,
                dst_dir_fd=output_dir_fd,
            )
        else:
            os.replace(temporary_path, output_path)
    finally:
        try:
            if output_dir_fd is not None:
                os.unlink(temporary_name, dir_fd=output_dir_fd)
            elif temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def _to_mono_16khz_pcm(waveform: Any, source_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError(f"Invalid source sample rate: {source_rate}")
    pcm = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise RuntimeError("Qwen3-TTS returned empty or non-finite audio")
    if source_rate != 16_000:
        pcm = resample_poly(pcm, up=16_000, down=source_rate).astype(np.float32)
        if not np.isfinite(pcm).all():
            raise RuntimeError("Qwen3-TTS audio contains non-finite values after resampling")
    peak = float(np.max(np.abs(pcm)))
    if peak > 0.95:
        pcm = (pcm * (0.95 / peak)).astype(np.float32)
    return pcm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS Phase 1 worker requires CUDA")
    if args.max_new_tokens < 8:
        raise ValueError("max-new-tokens must be >= 8")
    if not args.model_snapshot.is_dir() or not args.reference_audio.is_file() or not args.jobs_json.is_file():
        raise FileNotFoundError("model snapshot, reference audio, and jobs JSON must all exist locally")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    jobs = _validate_jobs(json.loads(args.jobs_json.read_text(encoding="utf-8")))
    output_dir = _safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_fd = _open_safe_output_dir_fd(output_dir)
    try:
        tts = _load_tts(args.model_snapshot)
        return _generate_jobs(tts, jobs, args, output_dir, output_dir_fd)
    finally:
        if output_dir_fd is not None:
            os.close(output_dir_fd)


def _generate_jobs(tts: Any, jobs: list[dict[str, str]], args: Any, output_dir: Path, output_dir_fd: int | None) -> int:
    for job in jobs:
        print(f"[qwen3tts-phase1] Generating {job['filename']}: {job['text']}", flush=True)
        wavs, sample_rate = tts.generate_voice_clone(
            text=job["text"],
            language=job["language"],
            ref_audio=str(args.reference_audio),
            x_vector_only_mode=True,
            non_streaming_mode=True,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            max_new_tokens=args.max_new_tokens,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
        )
        if len(wavs) != 1:
            raise RuntimeError(f"Expected one WAV for {job['filename']}, received {len(wavs)}")
        pcm = _to_mono_16khz_pcm(wavs[0], int(sample_rate))
        output_path = _safe_output_path(output_dir, job["filename"])
        _write_pcm_atomically(output_path, pcm, output_dir_fd=output_dir_fd)
        print(
            json.dumps(
                {"filename": job["filename"], "sample_rate_hz": 16_000, "duration_s": len(pcm) / 16_000},
                ensure_ascii=False,
            ),
            flush=True,
        )
        del wavs, pcm
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
