"""Fail-closed checks for short controller-guided BACKCHANNEL SAY text."""

from __future__ import annotations

import re


_TRAILING_CONNECTORS = frozenset({
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli", "all", "anche",
    "che", "con", "da", "dal", "dalla", "dalle", "dallo", "dei", "del",
    "della", "delle", "dello", "di", "e", "ed", "fra", "in", "ma", "o",
    "od", "per", "tra", "su",
})


class BackchannelAttemptLatch:
    """Allow one BACKCHANNEL attempt for each observed peer speech event.

    The caller scopes a peer event by its speech onset and keeps that key
    through the corresponding endpoint. Response completion and empty SAY
    output do not re-arm the current event.
    """

    def __init__(self) -> None:
        self._event_key: tuple[str, int] | None = None
        self._attempted = False

    def has_attempted(self, event_key: tuple[str, int] | None) -> bool:
        return event_key is not None and event_key == self._event_key and self._attempted

    def mark_attempted(self, event_key: tuple[str, int] | None) -> bool:
        if event_key is None:
            return False
        if event_key != self._event_key:
            self._event_key = event_key
            self._attempted = False
        if self._attempted:
            return False
        self._attempted = True
        return True


def peer_audio_event_key_for_frame(
    current_key: tuple[str, int] | None,
    frame_id: int,
    *,
    new_voice_onset: bool,
) -> tuple[str, int] | None:
    """Start a new BACKCHANNEL attempt scope only at a real speech onset."""
    if new_voice_onset:
        return ("onset", int(frame_id))
    return current_key


def _lexical_words(text: str) -> list[str]:
    # Apostrophes delimit orthographic tokens ("l'anatomia" counts as two).
    return [match.group(0).casefold() for match in re.finditer(r"[^\W_]+", text, flags=re.UNICODE)]


def validate_backchannel_phrase(text: str, *, max_words: int = 4) -> tuple[bool, str, int]:
    """Validate generated text without trimming, truncating, or rewriting it."""
    if not isinstance(text, str):
        return False, "text_not_string", 0
    words = _lexical_words(text)
    if not words:
        return False, "empty_phrase", 0
    if len(words) > max_words:
        return False, "too_many_words", len(words)
    if words[-1] in _TRAILING_CONNECTORS or re.search(r"[,;:…—-]\s*$", text):
        return False, "incomplete_clause", len(words)
    return True, "ok", len(words)


__all__ = [
    "BackchannelAttemptLatch",
    "peer_audio_event_key_for_frame",
    "validate_backchannel_phrase",
]
