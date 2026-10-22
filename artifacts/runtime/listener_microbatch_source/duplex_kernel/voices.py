"""Voice conditioning for the standalone Qwen3-TTS talker."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from duplex_kernel.config import WORKSPACE


def _root() -> Path:
    return Path(
        os.environ.get(
            "GEMMA4_OMNI_CUSTOM_VOICE_DIR",
            os.path.join(WORKSPACE, "custom_voices"),
        )
    )


def load_voice_profiles() -> dict[str, dict[str, Any]]:
    path = _root() / "custom_voice_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("voices", data)
    if not isinstance(raw, dict):
        return {}
    return {
        str(name).strip().lower(): {
            "name": str(info.get("name") or name),
            "file": str(info.get("file") or f"{str(name).lower()}.safetensors"),
            "mode": str(info.get("mode") or "xvec").lower(),
            "ref_text": info.get("ref_text"),
        }
        for name, info in raw.items() if isinstance(info, dict) and str(name).strip()
    }


def _decode_wav_b64(value: str) -> list[float] | None:
    if len(value) > 8_000_000:
        raise ValueError("ref_audio supera il limite di 6 MB base64")
    try:
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(base64.b64decode(value)), dtype="float32")
        wav = np.asarray(wav).mean(axis=1) if np.asarray(wav).ndim > 1 else np.asarray(wav)
        if sr != 24000:
            n = max(1, round(len(wav) * 24000 / sr))
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav)
        return wav.astype(np.float32).tolist()
    except (ValueError, OSError, base64.binascii.Error, RuntimeError):
        return None


def resolve_voice_conditioning(
    voice: str | None = None,
    voice_style: str | None = None,
    ref_audio_b64: str | None = None,
    ref_text: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a profile into the strict conditioning schema consumed by P7."""
    if voice:
        entry = load_voice_profiles().get(str(voice).strip().lower())
        if entry is None:
            raise ValueError(f"profilo voce non trovato: {str(voice).strip()}")
        if entry["mode"] not in {"xvec", "x_vector_only"}:
            raise ValueError(
                "P7 supporta esclusivamente voice conditioning x_vector_only"
            )
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors e PyTorch sono necessari per caricare la voce") from exc
        root = _root().resolve()
        path = (root / entry["file"]).resolve()
        if root not in path.parents:
            raise ValueError("percorso voce non autorizzato")
        try:
            tensors = load_file(str(path))
        except (OSError, ValueError) as exc:
            raise ValueError(f"impossibile caricare il profilo voce {str(voice).strip()}") from exc
        speaker = tensors.get("speaker_embedding")
        if not isinstance(speaker, torch.Tensor) or speaker.numel() == 0:
            raise ValueError("speaker_embedding mancante o non valido nel profilo voce")
        speaker = speaker.detach().float().reshape(-1).cpu().contiguous()
        if not bool(torch.isfinite(speaker).all()):
            raise ValueError("speaker_embedding contiene valori non finiti")
        return {
            "mode": "x_vector_only",
            "speaker_embedding": speaker,
        }
    if voice_style and str(voice_style).strip():
        raise ValueError("P7 non supporta voice_style; selezionare un profilo x_vector_only")
    if ref_audio_b64:
        raise ValueError("P7 richiede un profilo x_vector_only; ref_audio non è supportato")
    return None


__all__ = ["load_voice_profiles", "resolve_voice_conditioning"]
