"""Shared Gemma ASR prompt formatting for training and realtime inference."""
from __future__ import annotations

from typing import Any


def asr_language_name(language: Any) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    if value in {"it", "ita", "italian"} or value.startswith("it-") or "italian" in value:
        return "Italian"
    if value in {"cn", "zh", "chinese", "mandarin"} or value.startswith(("zh-", "cn-")) or "chinese" in value:
        return "Chinese"
    if value in {"en", "eng", "english"} or value.startswith("en-") or "english" in value:
        return "English"
    return str(language).strip() if language else "the original language"


def asr_prompt_content(language: Any, audio_token: str) -> str:
    """Return the exact training user text, including its marker separator."""
    token = str(audio_token)
    if not token:
        raise ValueError("audio_token non può essere vuoto")
    return (
        f"Transcribe the speech faithfully in {asr_language_name(language)}; "
        f"preserve the original wording. {token}"
    )
