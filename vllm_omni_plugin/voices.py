# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/voices.py — Profili voce di default per il talker Qwen3-TTS.

Formato ufficiale vLLM-Omni (SDD: vllm_omni/utils/speaker_cache.py):
- custom_voice_dir/custom_voice_manifest.json:
    {"model_type": "qwen3_tts", "voices": {"gemma": {"name": "Gemma",
      "file": "gemma.safetensors", "mode": "xvec"|"icl", "ref_text": "..."}}}
- custom_voice_dir/<file>.safetensors: tensore `speaker_embedding` (dim
  speaker_encoder_config.enc_dim = 2048); `ref_code` obbligatorio se mode=icl.

Il report tecnico Qwen3-TTS (arXiv:2601.15621) conferma: la variante Base
supporta sia il voice cloning a 3 secondi (task Base con ref_audio) sia
l'instruction following (task VoiceDesign con instruct). Le varianti
specializzate VoiceDesign/CustomVoice non servono: usiamo il Base con i
profili registrati (x-vector) per le voci di default.

Nota builder (prompt_embeds_builder.py ~1225): lo speaker embedding va dentro
``voice_clone_prompt.ref_spk_embedding`` (non come campo top-level), e
l'``x_vector_only_mode=True`` salta l'estrazione da ref_audio.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_VOICES_DIR = "/workspace/custom_voices"
_MANIFEST = "custom_voice_manifest.json"


def _voices_dir() -> str | None:
    d = os.environ.get("GEMMA4_OMNI_CUSTOM_VOICE_DIR") or DEFAULT_VOICES_DIR
    return d if os.path.isdir(d) else None


def load_voice_profiles() -> dict[str, dict[str, Any]]:
    """Carica i profili voce (manifest + metadati) dalla custom_voice_dir."""
    root = _voices_dir()
    if root is None:
        return {}
    manifest_path = Path(root) / _MANIFEST
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    voices: dict[str, dict[str, Any]] = {}
    raw_voices = manifest.get("voices", {})
    if not isinstance(raw_voices, dict):
        return voices
    for key, info in raw_voices.items():
        if not isinstance(info, dict):
            continue
        name = str(info.get("name") or key).strip()
        if not name:
            continue
        entry = {
            "name": name,
            "name_lower": name.lower(),
            "file": str(info.get("file") or f"{name.lower()}.safetensors"),
            "mode": str(info.get("mode") or "xvec").lower(),
            "ref_text": info.get("ref_text"),
        }
        voices[name.lower()] = entry
    return voices


def _load_profile_tensors(file_name: str) -> dict[str, Any] | None:
    root = _voices_dir()
    if root is None:
        return None
    path = Path(root) / file_name
    if not path.exists():
        return None
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None
    try:
        return load_file(str(path))
    except Exception:
        return None


def _decode_wav_b64(b64: str) -> list[float] | None:
    """Decodifica un WAV base64 in lista di campioni float32 (-1..1)."""
    try:
        import soundfile as sf

        raw = base64.b64decode(b64)
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 24000:
            # Resample lineare (i ref di produzione sono 24kHz).
            n = int(len(wav) * 24000 / sr)
            x = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav)
            wav = x
        return wav.astype(np.float32).tolist()
    except Exception:
        return None


def resolve_voice_conditioning(
    voice: str | None,
    voice_style: str | None,
    ref_audio_b64: str | None,
    ref_text: str | None,
) -> dict[str, Any] | None:
    """Risolve il conditioning voce del talker in additional_information.

    Priorità (allineata al report Qwen3-TTS + adapter nativo):
      1. voice = profilo registrato  → Base + voice_clone_prompt (xvec/icl)
      2. voice_style (instructions)  → VoiceDesign + instruct
      3. ref_audio                  → Base + ref_audio [[samples, sr]] (cloning 3s)
      4. nessuno                    → None (il talker usa il default VoiceDesign)
    """
    if voice:
        profiles = load_voice_profiles()
        entry = profiles.get(str(voice).strip().lower())
        if entry is not None:
            tensors = _load_profile_tensors(entry["file"]) or {}
            spk = tensors.get("speaker_embedding")
            if spk is not None:
                import torch

                emb = spk.reshape(-1).detach().cpu().tolist()
                prompt: dict[str, Any] = {
                    "ref_spk_embedding": emb,
                    "icl_mode": entry["mode"] == "icl",
                }
                if entry["mode"] == "icl":
                    ref_code = tensors.get("ref_code")
                    prompt["ref_code"] = ref_code.tolist() if ref_code is not None else None
                    return {
                        "task_type": ["Base"],
                        "voice_clone_prompt": prompt,
                        "x_vector_only_mode": [False],
                        "ref_text": [entry.get("ref_text") or ""],
                    }
                return {
                    "task_type": ["Base"],
                    "voice_clone_prompt": prompt,
                    "x_vector_only_mode": [True],
                }
            # Profilo senza tensori: fallback al nome (CustomVoice spk_id).
            return {"task_type": ["CustomVoice"], "speaker": [voice]}

    if voice_style and str(voice_style).strip():
        return {"task_type": ["VoiceDesign"], "instruct": [str(voice_style).strip()]}

    if ref_audio_b64 and str(ref_audio_b64).strip():
        samples = _decode_wav_b64(str(ref_audio_b64).strip())
        if samples is not None and len(samples) > 0:
            info: dict[str, Any] = {
                "task_type": ["Base"],
                "ref_audio": [[samples, 24000]],
            }
            if ref_text and str(ref_text).strip():
                info["ref_text"] = [str(ref_text).strip()]
                info["x_vector_only_mode"] = [False]
            else:
                info["x_vector_only_mode"] = [True]
            return info
        return {"task_type": ["Base"], "ref_audio": [str(ref_audio_b64).strip()]}

    return None
