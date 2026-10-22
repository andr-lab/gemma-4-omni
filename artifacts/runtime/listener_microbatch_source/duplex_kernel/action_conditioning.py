"""Plain-text action instructions shared by training and serving."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_ACTION_INSTRUCTIONS = {
    "BACKCHANNEL": (
        "Current turn action: BACKCHANNEL. The user is still speaking. Generate only a brief "
        "conversational acknowledgment of one to four words in the conversation language. "
        "Do not answer, summarize, or transcribe the unfinished user utterance."
    ),
    "SPEAK": (
        "Current turn action: SPEAK. Generate a new substantive response to the most recent "
        "completed user utterance, in the conversation language."
    ),
    "REPLAN": (
        "Current turn action: REPLAN. Revise the response to incorporate the most recent "
        "completed user correction or changed preference, in the conversation language."
    ),
}


def action_conditioning_instruction(action: str) -> str:
    """Return the canonical ordinary-text system instruction for a spoken action.

    This deliberately adds no vocabulary item or atomic action token. The same
    string is serialized into Stage1 action-conditioned examples and serving's
    ephemeral system message.
    """
    normalized = str(action).strip().upper()
    try:
        return _ACTION_INSTRUCTIONS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"action conditioning supports only {', '.join(_ACTION_INSTRUCTIONS)}; got {action!r}"
        ) from exc


def action_conditioned_system_prompt(base_instruction: str, action: str) -> str:
    """Append the canonical per-turn instruction without mutating session state."""
    base = str(base_instruction or "").strip()
    action_instruction = action_conditioning_instruction(action)
    return f"{base}\n\n{action_instruction}" if base else action_instruction


def action_control_marker_text(action: str) -> str:
    """Return the legacy marker literal for generated-output rejection guards."""
    normalized = str(action).strip().upper()
    action_conditioning_instruction(normalized)  # validate against the closed action set
    return f"[Evento di controllo del kernel: {normalized}]"


def action_control_event(action: str) -> str:
    """Return the canonical empty user-role event payload used for action conditioning.

    The action itself is represented by the ordinary-text system instruction.
    Keep an empty final user turn in train and runtime so the Gemma chat template
    ends at the same causal boundary without teaching the model to echo metadata.
    """
    action_conditioning_instruction(action)  # validate against the closed action set
    return ""


def action_conditioned_messages(
    base_instruction: str,
    action: str,
    history_messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a fresh per-generation prompt with a final empty user event turn.

    Action semantics live only in the ordinary-text system instruction. The
    empty turn is ephemeral, not a transcript, and must not be persisted to
    conversation history. Histories may end in either role, including
    consecutive assistant messages; the empty turn restores the same causal
    user boundary in training and serving.
    """
    messages = [{"role": "system", "content": action_conditioned_system_prompt(base_instruction, action)}]
    for index, item in enumerate(history_messages):
        if not isinstance(item, Mapping):
            raise TypeError(f"history message {index} must be a mapping")
        role = str(item.get("role", ""))
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"history message {index} must have user/assistant role and string content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": action_control_event(action)})
    return messages


__all__ = [
    "action_conditioning_instruction",
    "action_conditioned_system_prompt",
    "action_control_event",
    "action_control_marker_text",
    "action_conditioned_messages",
]
