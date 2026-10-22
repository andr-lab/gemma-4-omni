"""Shared prompt and causal audio-window contract for direct speech responses."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


TICK_MS = 80
DEFAULT_MAX_AUDIO_MS = 30_000


def _language_instruction(language: Any) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    if value in {"it", "ita", "italian"} or value.startswith("it-") or "italian" in value:
        return "Rispondi in modo naturale e pertinente in italiano, usando la cronologia e l'audio dell'utente."
    if value in {"en", "eng", "english"} or value.startswith("en-") or "english" in value:
        return "Reply naturally and relevantly in English, using the conversation history and the user's audio."
    if value in {"cn", "zh", "chinese", "mandarin"} or value.startswith(("zh-", "cn-")) or "chinese" in value:
        return "请结合对话历史和用户音频，用中文自然、贴切地回答。"
    return "Rispondi in modo naturale e pertinente nella lingua della conversazione, usando la cronologia e l'audio dell'utente."


def build_audio_response_messages(
    *,
    language: Any,
    history_messages: Sequence[Mapping[str, Any]],
    audio_token: str,
) -> list[dict[str, str]]:
    """Build the identical multimodal chat prefix for training and serving.

    ``history_messages`` must contain only previously completed turns. The
    current user utterance is represented by the final audio marker and must
    not also appear as transcript text.
    """
    if not isinstance(audio_token, str) or not audio_token.strip():
        raise ValueError("audio_token Gemma mancante")
    messages: list[dict[str, str]] = [{"role": "system", "content": _language_instruction(language)}]
    for index, item in enumerate(history_messages):
        if not isinstance(item, Mapping):
            raise TypeError(f"history message {index} must be a mapping")
        role = str(item.get("role", ""))
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError(f"history message {index} must be a completed user/assistant text turn")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": audio_token})
    return messages


def build_audio_response_protocol_v2_messages(
    *,
    history_messages: Sequence[Mapping[str, Any]],
    audio_token: str,
    session_instructions: str | None = None,
) -> list[dict[str, str]]:
    """Build the generic v2 protocol prefix shared by training and runtime.

    The current user is represented by audio only. The protocol-specific
    system instruction is owned by ``audio_response_protocol`` so it cannot
    drift between prompt and decoder parsing.
    """
    if not isinstance(audio_token, str) or not audio_token.strip():
        raise ValueError("audio_token Gemma mancante")
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM

    system = str(AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM)
    if not system.strip():
        raise ValueError("system prompt del protocollo audio risposta v2 vuoto")
    instructions = str(session_instructions or "").strip()
    system_content = (
        f"Contesto e istruzioni della conversazione:\n{instructions}\n\n{system}"
        if instructions else system
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for index, item in enumerate(history_messages):
        if not isinstance(item, Mapping):
            raise TypeError(f"history message {index} must be a mapping")
        role = str(item.get("role", ""))
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError(f"history message {index} must be a completed user/assistant text turn")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": audio_token})
    return messages


def build_controller_guided_say_v1_messages(
    *,
    language: Any,
    history_messages: Sequence[Mapping[str, Any]],
    audio_token: str,
    session_instructions: str | None,
    action: str,
) -> list[dict[str, str]]:
    """Build audio context for a response whose action is already controller-selected."""
    from duplex_kernel.audio_response_protocol import (
        CONTROLLER_BACKCHANNEL_MAX_WORDS,
        controller_guided_response_action,
    )

    selected_action = controller_guided_response_action(action)
    messages = build_audio_response_messages(
        language=language,
        history_messages=history_messages,
        audio_token=audio_token,
    )
    instructions = str(session_instructions or "").strip()
    if selected_action == "BACKCHANNEL":
        action_instruction = (
            "Il controller ha già selezionato BACKCHANNEL. Mantieni fissa questa azione: "
            "non riclassificare o cambiare l'azione. Genera soltanto il testo da pronunciare, "
            f"con un breve cenno di ascolto di 1–{CONTROLLER_BACKCHANNEL_MAX_WORDS} parole."
        )
    else:
        action_instruction = (
            "Il controller ha già selezionato SPEAK. Mantieni fissa questa azione: "
            "non riclassificare o cambiare l'azione. Genera soltanto il testo da pronunciare "
            "come risposta naturale e pertinente."
        )
    system_parts = [
        part for part in (instructions, _language_instruction(language), action_instruction) if part
    ]
    messages[0]["content"] = "\n\n".join(system_parts)
    return messages


def causal_audio_window(
    *,
    action: str,
    target_start_ms: float,
    current_user_start_ms: float,
    current_user_end_ms: float | None,
    current_user_active: bool | None = None,
    allow_active_speak: bool = False,
    allow_active_replan: bool = False,
    max_audio_ms: int = DEFAULT_MAX_AUDIO_MS,
    tick_ms: int = TICK_MS,
) -> dict[str, int | float]:
    """Return an end-exclusive, frame-aligned user-audio window.

    The last audio sample is no later than the 80 ms frame boundary available
    before assistant target onset. BACKCHANNEL requires an active user span;
    SPEAK and REPLAN require the current user span to have completed by that
    boundary, except a controller-authorized active SPEAK explicitly allowed
    by the live duplex scheduler. Long user turns are rejected rather than
    silently truncated.
    """
    normalized_action = str(action).strip().upper()
    values = (target_start_ms, current_user_start_ms)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise ValueError("tempi audio risposta devono essere numeri finiti")
    if current_user_active is not None and not isinstance(current_user_active, bool):
        raise ValueError("current_user_active deve essere un booleano esplicito")
    if not isinstance(allow_active_speak, bool):
        raise ValueError("allow_active_speak deve essere un booleano")
    if not isinstance(allow_active_replan, bool):
        raise ValueError("allow_active_replan deve essere un booleano")
    if current_user_end_ms is not None and (
        isinstance(current_user_end_ms, bool)
        or not isinstance(current_user_end_ms, (int, float))
        or not math.isfinite(float(current_user_end_ms))
    ):
        raise ValueError("fine audio utente deve essere un numero finito o None per un backchannel attivo")
    target_start = float(target_start_ms)
    user_start = float(current_user_start_ms)
    user_end = float(current_user_end_ms) if current_user_end_ms is not None else None
    if tick_ms <= 0 or max_audio_ms <= 0:
        raise ValueError("tick e durata massima audio devono essere positivi")
    if not (0 <= user_start and target_start > user_start):
        raise ValueError("intervallo utente corrente/inizio target non valido")
    if user_end is not None and not user_start < user_end:
        raise ValueError("intervallo utente corrente/inizio target non valido")
    if normalized_action not in {"BACKCHANNEL", "SPEAK", "REPLAN"}:
        raise ValueError(f"azione di risposta non supportata: {action!r}")

    cutoff_frame_count = math.floor(target_start / int(tick_ms))
    cutoff_ms = cutoff_frame_count * int(tick_ms)
    if normalized_action == "BACKCHANNEL":
        active = current_user_active if current_user_active is not None else (
            user_end is not None and user_start < target_start < user_end
        )
        if not active:
            raise ValueError("BACKCHANNEL richiede una voce utente attiva all'onset target")
    elif normalized_action == "SPEAK" and allow_active_speak:
        if current_user_active is not True or user_end is not None:
            raise ValueError("SPEAK attivo richiede span utente esplicito ancora aperto")
    elif normalized_action == "REPLAN" and allow_active_replan:
        if current_user_active is not True or user_end is not None:
            raise ValueError("REPLAN attivo richiede uno snapshot causale ancora aperto")
    elif user_end is None or user_end > cutoff_ms or current_user_active is True:
        raise ValueError("turno utente non completo al cutoff causale disponibile")
    if cutoff_ms <= user_start:
        raise ValueError("nessun frame audio utente disponibile prima del target")
    duration_ms = cutoff_ms - user_start
    if duration_ms > int(max_audio_ms):
        raise ValueError("turno utente supera la durata massima audio senza poter essere troncato")
    return {
        "audio_start_ms": user_start,
        "audio_end_ms": cutoff_ms,
        "input_cutoff_ms": cutoff_ms,
        "input_cutoff_frame_count": cutoff_frame_count,
        "input_cutoff_frame_id": cutoff_frame_count - 1,
        "input_audio_duration_ms": duration_ms,
    }


__all__ = [
    "DEFAULT_MAX_AUDIO_MS",
    "TICK_MS",
    "build_audio_response_messages",
    "build_audio_response_protocol_v2_messages",
    "build_controller_guided_say_v1_messages",
    "causal_audio_window",
]
