"""Small, deterministic helpers for safe Thinker budget termination."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def ends_at_sentence_boundary(text: str) -> bool:
    """Return whether stripped text ends at a sentence terminator."""
    value = str(text).rstrip()
    closers = "\"'”’)]}»"
    value = value.rstrip(closers)
    return bool(value) and value[-1] in ".!?"


def graceful_budget_action(
    text: str,
    emitted_tokens: int,
    requested_tokens: int,
    hard_limit_tokens: int,
) -> str:
    """Select whether to continue, stop on sentence end, or mark a cutoff."""
    if emitted_tokens < requested_tokens:
        return "continue"
    if ends_at_sentence_boundary(text):
        return "sentence_complete"
    if emitted_tokens < hard_limit_tokens:
        return "extend"
    return "partial"


def last_complete_word_prefix(
    token_ids: Sequence[int],
    decode: Callable[[Sequence[int]], str],
) -> tuple[int, str]:
    """Keep the longest token prefix ending at a provable word boundary.

    A boundary is established by a trailing delimiter in the decoded text, a
    sentence terminator, or the whitespace immediately before a potentially
    partial final word. The returned count always indexes the original token
    sequence so callers can trim matching hidden states without retokenizing.
    """
    ids = list(token_ids)
    text = str(decode(ids))
    stripped = text.rstrip()
    if not stripped:
        return 0, ""

    if ends_at_sentence_boundary(stripped) or text != stripped:
        target = stripped
    else:
        boundary = max(
            (index for index, char in enumerate(stripped) if char.isspace()),
            default=-1,
        )
        if boundary < 0:
            return 0, ""
        target = stripped[:boundary].rstrip()
        if not target:
            return 0, ""

    for count in range(len(ids), 0, -1):
        candidate = str(decode(ids[:count])).strip()
        if candidate == target:
            return count, target
    return 0, ""
