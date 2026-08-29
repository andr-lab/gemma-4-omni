"""Voice conditioning for the standalone Qwen3-TTS talker."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _root() -> Path:
    return Path(os.environ.get("GEMMA4_OMNI_CUSTOM_VOICE_DIR", "/workspace/custom_voices"))


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
    """Return the native Qwen3-TTS additional_information payload."""
    if voice:
        entry = load_voice_profiles().get(str(voice).strip().lower())
        if entry:
            try:
                from safetensors.torch import load_file
                root = _root().resolve()
                path = (root / entry["file"]).resolve()
                if root not in path.parents:
                    raise ValueError("percorso voce non autorizzato")
                tensors = load_file(str(path))
            except (ImportError, OSError, ValueError):
                tensors = {}
            speaker = tensors.get("speaker_embedding")
            if speaker is not None:
                prompt: dict[str, Any] = {
                    "ref_spk_embedding": speaker.reshape(-1).detach().cpu().tolist(),
                    "icl_mode": entry["mode"] == "icl",
                }
                if entry["mode"] == "icl":
                    code = tensors.get("ref_code")
                    prompt["ref_code"] = code.tolist() if code is not None else None
                out = {"task_type": ["Base"], "voice_clone_prompt": prompt,
                       "x_vector_only_mode": [entry["mode"] != "icl"]}
                if entry.get("ref_text"):
                    out["ref_text"] = [entry["ref_text"]]
                return out
            return {"task_type": ["CustomVoice"], "speaker": [str(voice)]}
    if voice_style and str(voice_style).strip():
        return {"task_type": ["VoiceDesign"], "instruct": [str(voice_style).strip()]}
    if ref_audio_b64:
        samples = _decode_wav_b64(str(ref_audio_b64).strip())
        if samples:
            out = {"task_type": ["Base"], "ref_audio": [[samples, 24000]],
                   "x_vector_only_mode": [not bool(ref_text)]}
            if ref_text:
                out["ref_text"] = [str(ref_text).strip()]
            return out
        raise ValueError("ref_audio non è un WAV base64 valido")
    return None


__all__ = ["load_voice_profiles", "resolve_voice_conditioning"]
