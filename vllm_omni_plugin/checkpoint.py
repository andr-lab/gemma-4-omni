# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/checkpoint.py — Risolutori di percorso per gli stadi TTS.

Il ``--model`` della pipeline è il thinker Gemma 4; gli stadi Talker e Code2Wav
devono puntare al repo Qwen3-TTS (modello + subfolder speech_tokenizer).
Firma ufficiale del resolver (vllm_omni): ``resolver(model, revision, task_type) -> str``.
"""

from __future__ import annotations

import glob
import os


def _find_repo(pattern: str, env_var: str, what: str) -> str:
    env_path = os.environ.get(env_var)
    if env_path and os.path.isdir(env_path):
        return env_path
    snaps = sorted(glob.glob(pattern))
    if not snaps:
        raise FileNotFoundError(
            f"Repository {what} non trovato (pattern={pattern}). "
            f"Impostare {env_var}."
        )
    return snaps[-1]


def resolve_qwen3_tts_root(model: str, revision: str | None, task_type: str | None) -> str:
    """Risolve il repo radice Qwen3-TTS (Talker + subfolder speech_tokenizer)."""
    return _find_repo(
        "/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*",
        "GEMMA4_OMNI_QWEN3_TTS_REPO",
        "Qwen3-TTS-12Hz-1.7B-Base",
    )


def resolve_qwen3_tts_speech_tokenizer(model: str, revision: str | None, task_type: str | None) -> str:
    """Risolve il subfolder speech_tokenizer (Code2Wav)."""
    root = resolve_qwen3_tts_root(model, revision, task_type)
    sub = os.path.join(root, "speech_tokenizer")
    if not os.path.isdir(sub):
        # Fallback: repo standalone Qwen3-TTS-Tokenizer-12Hz.
        tok = _find_repo(
            "/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*",
            "GEMMA4_OMNI_QWEN3_TTS_TOKENIZER_REPO",
            "Qwen3-TTS-Tokenizer-12Hz",
        )
        return tok
    return sub
