"""Wire format for opt-in audio-native responses from Gemma.

The protocol text and target formatter are also used by the data builder. The
incremental parser keeps the private transcript and action separate from the
user-facing ``say`` value and rejects any extra field, key reordering, or text.
"""
from __future__ import annotations

import json
import copy
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


AUDIO_RESPONSE_PROTOCOL_V2 = "heard_action_say_v2"
CONTROLLER_GUIDED_SAY_V1 = "controller_guided_say_v1"
AUDIO_RESPONSE_ACTIONS = ("SPEAK", "BACKCHANNEL", "REPLAN")
AUDIO_RESPONSE_V2_MAX_NEW_TOKENS = 128
AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS = 48
AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS = 8
AUDIO_RESPONSE_V2_MAX_SAY_TOKENS = 48
AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS = 16
AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM = (
    "Ricevi la cronologia della conversazione e l'audio dell'utente corrente. "
    "Restituisci soltanto un oggetto JSON compatto, senza markdown, con esattamente "
    'le chiavi e nell’ordine {"heard":"...","action":"SPEAK|BACKCHANNEL|REPLAN","say":"..."}. '
    '"heard" è la trascrizione dell’intervento utente corrente disponibile: per SPEAK e REPLAN '
    "includi l’intervento corrente completo quando è terminato; per BACKCHANNEL lascia la stringa vuota, "
    "perche' l'audio corrente e' ancora parziale e le parole future non sono disponibili. "
    'Scegli l’azione coerente con l’audio e la cronologia. "say" contiene il testo da pronunciare. '
    'Pronuncia e mostra solo "say": "heard" e "action" sono dati interni. '
    "Non inventare parole che l’audio non supporta."
)

_JSON_WHITESPACE = " \t\r\n"
_START_SEQUENCE = ((True, "{"), (True, '"heard"'), (True, ":"), (True, '"'))
_ACTION_SEQUENCE = ((True, ","), (True, '"action"'), (True, ":"), (True, '"'))
_SAY_SEQUENCE = ((True, ","), (True, '"say"'), (True, ":"), (True, '"'))
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class AudioResponseProtocolError(ValueError):
    """A v2 response violated its closed JSON contract.

    ``reason`` is a safe, stable trace label. It never contains generated
    text, since the transcript and action are private runtime metadata.
    """

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(f"audio response protocol v2 rejected: {self.reason}")


@dataclass(frozen=True)
class AudioResponseV2SayToken:
    """One token overlapping the raw say body.

    char_span uses offsets in the decoded, unescaped say string. A zero-width
    span is possible when this token contributes part of a JSON escape that
    becomes a say character only after a later token arrives.
    """

    token_index: int
    token_id: int
    char_span: tuple[int, int]
    hidden: Any


def _normalize_action(action: Any) -> str:
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    normalized = action.strip().upper()
    if normalized not in AUDIO_RESPONSE_ACTIONS:
        raise ValueError(f"unsupported audio response action: {action!r}")
    return normalized


def _normalize_allowed_actions(actions: Iterable[str] | None) -> tuple[str, ...] | None:
    if actions is None:
        return None
    normalized = tuple(dict.fromkeys(_normalize_action(value) for value in actions))
    if not normalized:
        raise ValueError("allowed_actions must contain at least one action")
    return normalized


def audio_response_actions_for_wake(wake_action: str | None) -> tuple[str, ...]:
    """Keep listener timing separate from the Thinker's learned response action."""
    if wake_action is None:
        return AUDIO_RESPONSE_ACTIONS
    wake = _normalize_action(wake_action)
    if wake in {"SPEAK", "BACKCHANNEL"}:
        return ("SPEAK", "BACKCHANNEL")
    if wake == "REPLAN":
        return ("REPLAN", "SPEAK")
    raise ValueError("wake action is not eligible for direct audio generation")


def audio_response_disposition_for_actions(wake_action: str | None, response_action: str) -> str:
    """Map a validated native action to its existing realtime lifecycle."""
    response = _normalize_action(response_action)
    allowed = audio_response_actions_for_wake(wake_action)
    if response not in allowed:
        raise AudioResponseProtocolError("response_action_not_allowed_for_wake")
    wake = _normalize_action(wake_action) if wake_action is not None else None
    if wake == "BACKCHANNEL" and response == "SPEAK":
        return "defer_until_endpoint"
    if response == "BACKCHANNEL":
        return "backchannel_ack"
    return "archive_completed_user_turn"


def _normalize_backchannel_phrase(value: str, *, preserve_trailing_space: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    trailing_space = bool(normalized and normalized[-1].isspace())
    collapsed = " ".join(normalized.split())
    if preserve_trailing_space and trailing_space and collapsed:
        collapsed += " "
    return collapsed


def normalize_audio_response_protocol(value: Any) -> str | None:
    """Resolve optional bundle metadata while preserving the legacy default."""
    if value is None or value == "":
        return None
    if value not in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}:
        raise ValueError(f"unsupported audio response protocol: {value!r}")
    return str(value)


def controller_guided_response_action(action: Any) -> str:
    """Accept only renderable actions already selected by the 80 ms controller."""
    try:
        normalized = _normalize_action(action)
    except ValueError as exc:
        raise AudioResponseProtocolError("controller_action_not_renderable") from exc
    if normalized not in {"SPEAK", "BACKCHANNEL"}:
        raise AudioResponseProtocolError("controller_action_not_renderable")
    return normalized


CONTROLLER_BACKCHANNEL_MAX_WORDS = 4
_CONTROLLER_BACKCHANNEL_TERMINAL_PUNCTUATION = frozenset(",.!?;…")


def controller_backchannel_candidate_allowed(
    current_text: str,
    candidate_text: str,
    *,
    terminal: bool,
    max_words: int = CONTROLLER_BACKCHANNEL_MAX_WORDS,
) -> bool:
    """Keep generated backchannels within a token-aligned word boundary."""
    if not isinstance(current_text, str) or not isinstance(candidate_text, str):
        return False
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer")
    current_words = len(current_text.split())
    if terminal:
        return 1 <= current_words <= max_words
    proposed = current_text + candidate_text
    if len(proposed.split()) > max_words:
        return False
    punctuation_at = next(
        (index for index, char in enumerate(proposed) if char in _CONTROLLER_BACKCHANNEL_TERMINAL_PUNCTUATION),
        None,
    )
    if punctuation_at is not None and proposed[punctuation_at + 1 :].strip():
        return False
    return True


def controller_backchannel_should_stop(
    text: str,
    *,
    max_words: int = CONTROLLER_BACKCHANNEL_MAX_WORDS,
) -> bool:
    """Stop at generated punctuation or after the fourth complete word."""
    if not isinstance(text, str):
        return False
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer")
    punctuation_at = next(
        (index for index, char in enumerate(text) if char in _CONTROLLER_BACKCHANNEL_TERMINAL_PUNCTUATION),
        None,
    )
    if punctuation_at is not None:
        return (
            1 <= len(text[: punctuation_at + 1].split()) <= max_words
            and not text[punctuation_at + 1 :].strip()
        )
    stripped = text.rstrip()
    return (
        len(stripped.split()) == max_words
        and len(text) > len(stripped)
        and text[len(stripped) :].isspace()
    )


def slice_controller_guided_say_hidden(
    hidden: Any,
    *,
    hidden_prompt_len: int,
    generated_token_ids: Iterable[int],
    excluded_token_ids: Iterable[int] = (),
) -> Any:
    """Return only text-bearing generated-token hidden states, excluding the prompt."""
    import torch

    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise AudioResponseProtocolError("say_hidden_invalid_shape")
    if isinstance(hidden_prompt_len, bool) or not isinstance(hidden_prompt_len, int) or hidden_prompt_len < 0:
        raise AudioResponseProtocolError("say_hidden_prompt_length_invalid")
    token_ids = list(generated_token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
        raise AudioResponseProtocolError("say_hidden_token_ids_invalid")
    if int(hidden.shape[1]) != hidden_prompt_len + len(token_ids):
        raise AudioResponseProtocolError("say_hidden_token_alignment")
    excluded = {int(token_id) for token_id in excluded_token_ids}
    positions = [
        hidden_prompt_len + index
        for index, token_id in enumerate(token_ids)
        if token_id not in excluded
    ]
    if not positions:
        raise AudioResponseProtocolError("say_hidden_empty")
    indices = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    return hidden.index_select(1, indices)


def _validate_semantics(*, heard: str, action: str, say: str | None = None) -> None:
    if action == "BACKCHANNEL":
        if heard:
            raise AudioResponseProtocolError("backchannel_heard_must_be_empty")
    elif not heard.strip():
        raise AudioResponseProtocolError("completed_turn_heard_empty")
    if say is not None and not say.strip():
        raise AudioResponseProtocolError("say_empty")


def format_audio_response_v2_target(*, heard: str, action: str, say: str) -> str:
    """Serialize one canonical compact training target in stable key order."""
    if not isinstance(heard, str) or not isinstance(say, str):
        raise TypeError("heard and say must be strings")
    normalized_action = _normalize_action(action)
    _validate_semantics(heard=heard, action=normalized_action, say=say)
    return json.dumps(
        {"heard": heard, "action": normalized_action, "say": say},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _decode_complete_json_string(raw: str, field: str) -> str:
    try:
        value = json.loads('"' + raw + '"')
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AudioResponseProtocolError(f"invalid_{field}_string") from exc
    if not isinstance(value, str):
        raise AudioResponseProtocolError(f"invalid_{field}_string")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise AudioResponseProtocolError(f"invalid_{field}_unicode_scalar")
    return value


def _json_string_escape_completion_suffix(raw: str) -> str | None:
    """Return a valid JSON-string suffix for a trailing incomplete escape.

    ``None`` means the current raw value is already invalid; ``""`` means
    there is no incomplete escape; a non-empty result finishes only the
    trailing escape and never adds a quote.  The suffix is used with the
    active tokenizer to reserve field-token capacity before accepting an
    escape prefix near its cap.
    """
    def scalar_for_prefix(prefix: str, *, low_surrogate: bool = False) -> int | None:
        if any(char not in "0123456789abcdefABCDEF" for char in prefix):
            return None
        missing = 4 - len(prefix)
        base = int(prefix or "0", 16) << (4 * missing)
        upper = base + (16 ** missing) - 1
        ranges = ((0xDC00, 0xDFFF),) if low_surrogate else (
            (0x0000, 0xD7FF),
            (0xE000, 0x10FFFF),
        )
        for lower, end in ranges:
            candidate = max(base, lower)
            if candidate <= min(upper, end):
                return candidate
        return None

    index = 0
    while index < len(raw):
        if raw[index] != "\\":
            index += 1
            continue
        index += 1
        if index >= len(raw):
            return "\\"
        escaped = raw[index]
        if escaped in _SIMPLE_ESCAPES:
            index += 1
            continue
        if escaped != "u":
            return None

        digit_start = index + 1
        digits = raw[digit_start:digit_start + 4]
        if len(digits) < 4:
            value = scalar_for_prefix(digits)
            if value is not None:
                return f"{value:04X}"[len(digits):]
            missing = 4 - len(digits)
            base = int(digits or "0", 16) << (4 * missing)
            upper = base + (16 ** missing) - 1
            high = max(base, 0xD800)
            if high > min(upper, 0xDBFF):
                return None
            suffix = f"{high:04X}"[len(digits):] + r"\uDC00"
            return suffix
        if any(char not in "0123456789abcdefABCDEF" for char in digits):
            return None

        codepoint = int(digits, 16)
        index = digit_start + 4
        if 0xDC00 <= codepoint <= 0xDFFF:
            return None
        if not 0xD800 <= codepoint <= 0xDBFF:
            continue

        # A high surrogate must be followed by a low-surrogate escape. If its
        # prefix has already started, only complete that exact prefix.
        if index >= len(raw):
            return r"\uDC00"
        if raw[index] != "\\":
            return None
        if index + 1 >= len(raw):
            return "uDC00"
        if raw[index + 1] != "u":
            return None
        low_start = index + 2
        low_digits = raw[low_start:low_start + 4]
        if len(low_digits) < 4:
            low = scalar_for_prefix(low_digits, low_surrogate=True)
            if low is None:
                return None
            return f"{low:04X}"[len(low_digits):]
        if any(char not in "0123456789abcdefABCDEF" for char in low_digits):
            return None
        low = int(low_digits, 16)
        if not 0xDC00 <= low <= 0xDFFF:
            return None
        index = low_start + 4

    return ""


class AudioResponseV2Parser:
    """Incrementally parse the exact ``heard``/``action``/``say`` JSON form.

    ``feed`` returns only newly decoded ``say`` characters. Callers that
    synthesize speech must hold those characters and their matching hidden
    states until :meth:`finish` validates the complete object.
    """

    def __init__(
        self,
        *,
        expected_action: str | None = None,
        allowed_actions: Iterable[str] | None = None,
    ) -> None:
        self.expected_action = _normalize_action(expected_action) if expected_action is not None else None
        self.allowed_actions = _normalize_allowed_actions(allowed_actions)
        if self.expected_action is not None:
            if self.allowed_actions is not None and self.allowed_actions != (self.expected_action,):
                raise ValueError("expected_action and allowed_actions conflict")
            self.allowed_actions = (self.expected_action,)
        self._phase = "heard_prefix"
        self._buffer = ""
        self._cursor = 0
        self._syntax_item_index = 0
        self._literal_index = 0
        self._string_raw: list[str] = []
        self._string_escaped = False
        self._say_drain_offset = 0
        self._safe_say: list[str] = []
        self._feed_say_start: int | None = None
        self._feed_say_end: int | None = None
        self._feed_heard_start: int | None = None
        self._feed_heard_end: int | None = None
        self._feed_action_start: int | None = None
        self._feed_action_end: int | None = None
        self.last_feed_heard_raw_span: tuple[int, int] | None = None
        self.last_feed_action_raw_span: tuple[int, int] | None = None
        self.last_feed_say_raw_span: tuple[int, int] | None = None
        self.heard: str | None = None
        self.action: str | None = None
        self.failure_reason: str | None = None
        self.cancelled = False

    @property
    def _allows_native_backchannel_after_partial_heard(self) -> bool:
        """Whether a wake-scoped native choice may discard a partial HEARD.

        Fixed-action BACKCHANNEL decoding and canonical training targets remain
        strict. Only a multi-action runtime choice may select BACKCHANNEL after
        the model has transcribed audio that was still partial at the wake.
        """
        return (
            self.expected_action is None
            and self.allowed_actions is not None
            and len(self.allowed_actions) > 1
            and "BACKCHANNEL" in self.allowed_actions
        )

    @property
    def pending_say(self) -> str:
        """Decoded say text collected so far; it is still uncommitted."""
        return "".join(self._safe_say)

    @property
    def complete(self) -> bool:
        return self._phase == "done" and self.failure_reason is None and not self.cancelled

    @property
    def phase(self) -> str:
        """Current syntactic field, exposed for narrow decoder constraints."""
        return self._phase

    def _reject(self, reason: str) -> None:
        self.failure_reason = reason
        raise AudioResponseProtocolError(reason)

    def _consume_sequence(
        self,
        sequence: tuple[tuple[bool, str], ...],
        next_phase: str,
        failure_reason: str,
    ) -> bool:
        """Consume JSON punctuation while allowing RFC JSON whitespace between tokens."""
        while self._syntax_item_index < len(sequence):
            allow_whitespace, expected = sequence[self._syntax_item_index]
            if self._literal_index == 0 and allow_whitespace:
                while self._cursor < len(self._buffer) and self._buffer[self._cursor] in _JSON_WHITESPACE:
                    self._cursor += 1
            while self._literal_index < len(expected):
                if self._cursor >= len(self._buffer):
                    return False
                if self._buffer[self._cursor] != expected[self._literal_index]:
                    self._reject(failure_reason)
                self._cursor += 1
                self._literal_index += 1
            self._syntax_item_index += 1
            self._literal_index = 0
        self._syntax_item_index = 0
        self._phase = next_phase
        self._string_raw.clear()
        self._string_escaped = False
        return True

    def _drain_say(self) -> str:
        raw = "".join(self._string_raw)
        index = self._say_drain_offset
        decoded: list[str] = []
        while index < len(raw):
            char = raw[index]
            if char != "\\":
                codepoint = ord(char)
                if codepoint < 0x20:
                    self._reject("invalid_say_control_character")
                if 0xD800 <= codepoint <= 0xDFFF:
                    self._reject("invalid_say_unicode_scalar")
                decoded.append(char)
                index += 1
                continue

            if index + 1 >= len(raw):
                break
            escaped = raw[index + 1]
            if escaped in _SIMPLE_ESCAPES:
                decoded.append(_SIMPLE_ESCAPES[escaped])
                index += 2
                continue
            if escaped != "u":
                self._reject("invalid_say_escape")
            if index + 6 > len(raw):
                break
            digits = raw[index + 2:index + 6]
            if any(char not in "0123456789abcdefABCDEF" for char in digits):
                self._reject("invalid_say_unicode_escape")
            codepoint = int(digits, 16)
            index += 6
            if 0xD800 <= codepoint <= 0xDBFF:
                low_prefix = raw[index:index + 2]
                if not "\\u".startswith(low_prefix):
                    self._reject("invalid_say_unicode_scalar")
                if index + 6 > len(raw):
                    index -= 6
                    break
                low_digits = raw[index + 2:index + 6]
                if any(char not in "0123456789abcdefABCDEF" for char in low_digits):
                    self._reject("invalid_say_unicode_escape")
                low = int(low_digits, 16)
                if not 0xDC00 <= low <= 0xDFFF:
                    self._reject("invalid_say_unicode_scalar")
                decoded.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)))
                index += 6
            elif 0xDC00 <= codepoint <= 0xDFFF:
                self._reject("invalid_say_unicode_scalar")
            else:
                decoded.append(chr(codepoint))

        self._say_drain_offset = index
        delta = "".join(decoded)
        if delta:
            self._safe_say.append(delta)
        return delta

    def _close_string(self, field: str) -> None:
        raw = "".join(self._string_raw)
        try:
            value = _decode_complete_json_string(raw, field)
        except AudioResponseProtocolError as exc:
            self.failure_reason = exc.reason
            raise
        if field == "heard":
            self.heard = value
            self._phase = "action_prefix"
        elif field == "action":
            try:
                self.action = _normalize_action(value)
            except ValueError as exc:
                self._reject("unsupported_action")
            if self.expected_action is not None and self.action != self.expected_action:
                self._reject("expected_action_mismatch")
            if self.allowed_actions is not None and self.action not in self.allowed_actions:
                self._reject("action_not_allowed_for_wake")
            assert self.heard is not None
            if self.action == "BACKCHANNEL" and self._allows_native_backchannel_after_partial_heard:
                # HEARD was generated before the native response action became
                # known. BACKCHANNEL acknowledges an ongoing utterance, so this
                # partial transcript is discarded before it can be published or
                # archived as a completed user turn.
                self.heard = ""
            else:
                try:
                    _validate_semantics(heard=self.heard, action=self.action)
                except AudioResponseProtocolError as exc:
                    self.failure_reason = exc.reason
                    raise
            self._phase = "say_prefix"
        else:
            if self._say_drain_offset != len(raw):
                self._reject("invalid_say_unicode_scalar")
            decoded = "".join(self._safe_say)
            if value != decoded:
                self._reject("invalid_say_string")
            try:
                _validate_semantics(heard=self.heard or "", action=self.action or "", say=value)
            except AudioResponseProtocolError as exc:
                self.failure_reason = exc.reason
                raise
            self._phase = "closing_brace"
        self._string_raw.clear()
        self._string_escaped = False
        self._say_drain_offset = 0

    def feed(self, fragment: str) -> str:
        """Consume a decoded JSON fragment and return newly safe say text."""
        if self.cancelled:
            self._reject("cancelled")
        if self.failure_reason is not None:
            raise AudioResponseProtocolError(self.failure_reason)
        if not isinstance(fragment, str):
            self._reject("fragment_must_be_text")
        self._buffer += fragment
        self._feed_heard_start = None
        self._feed_heard_end = None
        self._feed_action_start = None
        self._feed_action_end = None
        self._feed_say_start = None
        self._feed_say_end = None
        self.last_feed_heard_raw_span = None
        self.last_feed_action_raw_span = None
        self.last_feed_say_raw_span = None
        visible: list[str] = []
        while True:
            if self._phase == "heard_prefix":
                if not self._consume_sequence(_START_SEQUENCE, "heard_value", "expected_heard_key"):
                    break
            elif self._phase == "heard_value":
                if self._cursor >= len(self._buffer):
                    break
                char_index = self._cursor
                char = self._buffer[self._cursor]
                self._cursor += 1
                if char == '"' and not self._string_escaped:
                    self._close_string("heard")
                    continue
                if self._feed_heard_start is None:
                    self._feed_heard_start = char_index
                self._feed_heard_end = char_index + 1
                self._string_raw.append(char)
                if self._string_escaped:
                    self._string_escaped = False
                elif char == "\\":
                    self._string_escaped = True
            elif self._phase == "action_prefix":
                if not self._consume_sequence(_ACTION_SEQUENCE, "action_value", "expected_action_key"):
                    break
            elif self._phase == "action_value":
                if self._cursor >= len(self._buffer):
                    break
                char_index = self._cursor
                char = self._buffer[self._cursor]
                self._cursor += 1
                if char == '"' and not self._string_escaped:
                    self._close_string("action")
                    continue
                if self._feed_action_start is None:
                    self._feed_action_start = char_index
                self._feed_action_end = char_index + 1
                self._string_raw.append(char)
                if self._string_escaped:
                    self._string_escaped = False
                elif char == "\\":
                    self._string_escaped = True
            elif self._phase == "say_prefix":
                if not self._consume_sequence(_SAY_SEQUENCE, "say_value", "expected_say_key"):
                    break
            elif self._phase == "say_value":
                if self._cursor >= len(self._buffer):
                    visible.append(self._drain_say())
                    break
                char_index = self._cursor
                char = self._buffer[self._cursor]
                self._cursor += 1
                if char == '"' and not self._string_escaped:
                    visible.append(self._drain_say())
                    self._close_string("say")
                    continue
                if self._feed_say_start is None:
                    self._feed_say_start = char_index
                self._feed_say_end = char_index + 1
                self._string_raw.append(char)
                if self._string_escaped:
                    self._string_escaped = False
                elif char == "\\":
                    self._string_escaped = True
            elif self._phase == "closing_brace":
                while self._cursor < len(self._buffer) and self._buffer[self._cursor] in _JSON_WHITESPACE:
                    self._cursor += 1
                if self._cursor >= len(self._buffer):
                    break
                if self._buffer[self._cursor] != "}":
                    self._reject("expected_closing_brace")
                self._cursor += 1
                self._phase = "done"
            elif self._phase == "done":
                trailing = self._buffer[self._cursor:]
                if any(char not in " \t\r\n" for char in trailing):
                    self._reject("trailing_content")
                self._cursor = len(self._buffer)
                break
            else:
                self._reject("invalid_parser_state")
        if self._cursor:
            self._buffer = self._buffer[self._cursor:]
            self._cursor = 0
        if self._feed_heard_start is not None and self._feed_heard_end is not None:
            self.last_feed_heard_raw_span = (self._feed_heard_start, self._feed_heard_end)
        if self._feed_action_start is not None and self._feed_action_end is not None:
            self.last_feed_action_raw_span = (self._feed_action_start, self._feed_action_end)
        if self._feed_say_start is not None and self._feed_say_end is not None:
            self.last_feed_say_raw_span = (self._feed_say_start, self._feed_say_end)
        return "".join(piece for piece in visible if piece)

    def finish(self) -> dict[str, str]:
        """Return a fully validated object or raise a safe protocol error."""
        if self.cancelled:
            self._reject("cancelled")
        if self.failure_reason is not None:
            raise AudioResponseProtocolError(self.failure_reason)
        if not self.complete:
            self._reject("incomplete_json")
        assert self.heard is not None and self.action is not None
        return {"heard": self.heard, "action": self.action, "say": self.pending_say}

    def cancel(self) -> None:
        """Discard private, uncommitted fields after a generation fence."""
        self.cancelled = True
        self._phase = "cancelled"
        self._buffer = ""
        self._cursor = 0
        self._syntax_item_index = 0
        self._feed_heard_start = None
        self._feed_heard_end = None
        self._feed_action_start = None
        self._feed_action_end = None
        self.last_feed_heard_raw_span = None
        self.last_feed_action_raw_span = None
        self._string_raw.clear()
        self._safe_say.clear()
        self.heard = None
        self.action = None


class AudioResponseV2Collector:
    """Collect token-aligned Thinker output and retain only ``say`` hidden states.

    Nothing becomes publishable until ``finish`` validates the closing JSON
    delimiter and confirms that tokenwise decoded text matches the final
    Thinker decode. Tokens crossing the say-body boundary retain only the exact
    body character span in their trace.
    """

    def __init__(
        self,
        *,
        expected_action: str | None = None,
        allowed_actions: Iterable[str] | None = None,
        backchannel_say_phrases: tuple[str, ...] | list[str] = (),
    ) -> None:
        self.parser = AudioResponseV2Parser(
            expected_action=expected_action,
            allowed_actions=allowed_actions,
        )
        if not isinstance(backchannel_say_phrases, (tuple, list)) or any(
            not isinstance(phrase, str) or not phrase.strip()
            for phrase in backchannel_say_phrases
        ):
            raise ValueError("backchannel_say_phrases deve contenere solo frasi non vuote")
        raw_phrases_by_normalized: dict[str, str] = {}
        for phrase in backchannel_say_phrases:
            raw_phrase = phrase.strip()
            raw_phrases_by_normalized.setdefault(
                _normalize_backchannel_phrase(raw_phrase), raw_phrase
            )
        self._backchannel_say_phrases = tuple(raw_phrases_by_normalized)
        self._backchannel_say_phrase_texts = tuple(raw_phrases_by_normalized.values())
        self.backchannel_say_lexicon_rejections = 0
        self._response_parts: list[str] = []
        self._say_tokens: list[AudioResponseV2SayToken] = []
        self._private_token_ids: list[int] = []
        self.heard_field_token_count = 0
        self.action_field_token_count = 0
        self.say_field_token_count = 0
        self.heard_token_cap = AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS
        self.action_token_cap = AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS
        self.say_token_cap = AUDIO_RESPONSE_V2_MAX_SAY_TOKENS
        self.heard_truncated = False
        self.say_truncated = False
        self._finished = False
        self._cancelled = False
        self._parsed: dict[str, str] | None = None

    @property
    def backchannel_say_lexicon_active(self) -> bool:
        return (
            bool(self._backchannel_say_phrases)
            and (
                self.parser.action == "BACKCHANNEL"
                or self.parser.expected_action == "BACKCHANNEL"
            )
        )

    @property
    def say_tokens(self) -> tuple[AudioResponseV2SayToken, ...]:
        return tuple(self._say_tokens) if self.publishable else ()

    @property
    def generated_token_count(self) -> int:
        return len(self._private_token_ids)

    def set_token_caps(self, *, heard: int, action: int, say: int) -> None:
        caps = (int(heard), int(action), int(say))
        if any(cap < 0 for cap in caps):
            raise ValueError("audio response field token caps must be non-negative")
        self.heard_token_cap, self.action_token_cap, self.say_token_cap = caps

    @property
    def publishable_text(self) -> str:
        if not self.publishable or self._parsed is None:
            return ""
        return self._parsed["say"]

    @property
    def publishable(self) -> bool:
        return self._finished and not self._cancelled

    @property
    def action_value_pending(self) -> bool:
        """Whether the ordered JSON parser is positioned at the action string."""
        return (
            not self._cancelled
            and self.parser.phase == "action_value"
            and not self.parser._string_raw
        )

    def is_valid_next_text(
        self,
        text: str,
        *,
        terminal: bool,
        max_say_words: int | None = None,
        max_heard_tokens: int | None = None,
        max_action_tokens: int | None = None,
        max_say_tokens: int | None = None,
    ) -> bool:
        """Preview one prospective token without mutating accepted parser state.

        The decoder calls this before it forwards a token into the Thinker KV
        cache. Incomplete but syntactically valid prefixes are accepted;
        terminators are accepted only after the entire JSON object validates.
        """
        if self._cancelled or self._finished or not isinstance(text, str):
            return False
        if not terminal and not text:
            return False
        preview = copy.deepcopy(self.parser)
        heard_body_token = False
        action_body_token = False
        say_body_token = False
        try:
            if text:
                preview.feed(text)
                heard_body_token = preview.last_feed_heard_raw_span is not None
                action_body_token = preview.last_feed_action_raw_span is not None
                say_body_token = preview.last_feed_say_raw_span is not None
            if terminal:
                preview.finish()
        except AudioResponseProtocolError:
            return False
        if (
            max_heard_tokens is not None
            and self.heard_field_token_count + int(heard_body_token) > int(max_heard_tokens)
        ):
            return False
        if (
            max_action_tokens is not None
            and self.action_field_token_count + int(action_body_token) > int(max_action_tokens)
        ):
            return False
        if (
            max_say_tokens is not None
            and self.say_field_token_count + int(say_body_token) > int(max_say_tokens)
        ):
            return False
        if max_say_words is not None and preview.action == "BACKCHANNEL":
            if len(preview.pending_say.split()) > max_say_words:
                return False
        allowed_actions = self.parser.allowed_actions
        if allowed_actions is not None and preview.phase == "action_value":
            # A tokenizer token may cross the closing heard quote and the
            # action-field boundary. Keep that compound token only when its
            # action bytes still form a prefix of the controller-selected
            # enum; waiting for the closing quote lets an incompatible class
            # enter the KV cache and makes the field impossible to recover.
            action_prefix = "".join(preview._string_raw)
            if not any(action.startswith(action_prefix) for action in allowed_actions):
                return False
        if self.backchannel_say_lexicon_active and preview.action == "BACKCHANNEL" and (
            preview.phase in {"say_value", "closing_brace", "done"}
        ):
            candidate = _normalize_backchannel_phrase(
                preview.pending_say,
                preserve_trailing_space=preview.phase == "say_value",
            )
            complete = terminal or preview.phase in {"closing_brace", "done"}
            if complete:
                if candidate not in self._backchannel_say_phrases:
                    self.backchannel_say_lexicon_rejections += 1
                    return False
            elif not any(phrase.startswith(candidate) for phrase in self._backchannel_say_phrases):
                self.backchannel_say_lexicon_rejections += 1
                return False
        return preview.failure_reason is None and not preview.cancelled

    @property
    def backchannel_say_is_complete_phrase(self) -> bool:
        """Whether the current BACKCHANNEL SAY body is one bundled train phrase."""
        if not self.backchannel_say_lexicon_active:
            return False
        candidate = _normalize_backchannel_phrase(self.parser.pending_say)
        return candidate in self._backchannel_say_phrases

    @property
    def backchannel_say_lexicon_phrase_count(self) -> int:
        return len(self._backchannel_say_phrases) if self.backchannel_say_lexicon_active else 0

    def would_exceed_say_word_limit(self, text: str, *, max_say_words: int) -> bool:
        """Check a decoded candidate against the BACKCHANNEL word cap."""
        if self._cancelled or self._finished or not isinstance(text, str) or not text:
            return False
        preview = copy.deepcopy(self.parser)
        try:
            preview.feed(text)
        except AudioResponseProtocolError:
            return False
        return (
            preview.action == "BACKCHANNEL"
            and len(preview.pending_say.split()) > int(max_say_words)
        )

    def feed_token(self, *, token_index: int, token_id: int, text: str, hidden: Any) -> str:
        if self._cancelled:
            raise AudioResponseProtocolError("cancelled")
        if self._finished:
            raise AudioResponseProtocolError("collector_already_finished")
        if not isinstance(text, str):
            raise AudioResponseProtocolError("fragment_must_be_text")
        say_char_start = len(self.parser.pending_say)
        self.parser.feed(text)
        if self.parser.last_feed_heard_raw_span is not None:
            self.heard_field_token_count += 1
        if self.parser.last_feed_action_raw_span is not None:
            self.action_field_token_count += 1
        if self.parser.last_feed_say_raw_span is not None:
            self.say_field_token_count += 1
        if len(self._private_token_ids) < AUDIO_RESPONSE_V2_MAX_NEW_TOKENS:
            self._private_token_ids.append(int(token_id))
        raw_say_span = self.parser.last_feed_say_raw_span
        if raw_say_span is not None:
            retained_hidden = hidden
            detach = getattr(retained_hidden, "detach", None)
            if callable(detach):
                retained_hidden = detach()
                # Compiled Thinker forwards may expose a borrowed output view
                # whose storage is reused by the next decode step. Snapshot
                # only SAY states retained beyond this callback; private JSON
                # and heard/action token states are never copied or published.
                clone = getattr(retained_hidden, "clone", None)
                if callable(clone):
                    retained_hidden = clone()
            self._say_tokens.append(
                AudioResponseV2SayToken(
                    token_index=int(token_index),
                    token_id=int(token_id),
                    char_span=(say_char_start, len(self.parser.pending_say)),
                    hidden=retained_hidden,
                )
            )
        self._response_parts.append(text)
        # The current rollout validates the complete JSON before exposing text
        # or hidden states to synthesis, clients, V9, or history.
        return ""

    def finish(self, *, expected_text: str | None = None) -> dict[str, str]:
        if self._cancelled:
            raise AudioResponseProtocolError("cancelled")
        if self._finished:
            raise AudioResponseProtocolError("collector_already_finished")
        parsed = self.parser.finish()
        response_text = "".join(self._response_parts)
        if expected_text is not None and response_text.strip() != str(expected_text).strip():
            error = AudioResponseProtocolError("token_decode_mismatch")
            error.expected_text_repr = repr(str(expected_text)[:512])
            error.collector_text_repr = repr(response_text[:512])
            error.generated_token_ids = list(
                self._private_token_ids[:AUDIO_RESPONSE_V2_MAX_NEW_TOKENS]
            )
            self.parser.failure_reason = error.reason
            raise error
        if parsed["say"] and not self._say_tokens:
            self.parser._reject("say_hidden_missing")
        self._parsed = parsed
        self._finished = True
        return parsed

    def private_failure_snapshot(self, *, reason: str) -> dict[str, Any]:
        """Return bounded token IDs and parser counts for local diagnostics only."""
        phase = self.parser.phase
        raw_length = len(self.parser._string_raw)
        return {
            "reason": str(reason),
            "token_ids": list(self._private_token_ids[:AUDIO_RESPONSE_V2_MAX_NEW_TOKENS]),
            "token_count": len(self._private_token_ids),
            "parser_phase": phase,
            "heard_field_token_count": int(self.heard_field_token_count),
            "action_field_token_count": int(self.action_field_token_count),
            "say_field_token_count": int(self.say_field_token_count),
            "heard_token_cap": int(self.heard_token_cap),
            "action_token_cap": int(self.action_token_cap),
            "say_token_cap": int(self.say_token_cap),
            "heard_truncated": bool(self.heard_truncated),
            "say_truncated": bool(self.say_truncated),
            "current_string_raw_chars": int(raw_length),
            "completed_heard_chars": len(self.parser.heard or ""),
            "completed_say_chars": len(self.parser.pending_say),
        }

    def cancel(self) -> None:
        self._cancelled = True
        self._finished = False
        self._response_parts.clear()
        self._say_tokens.clear()
        self._private_token_ids.clear()
        self._parsed = None
        self.parser.cancel()


def finish_audio_response_v2_with_say_hidden(
    collector: AudioResponseV2Collector,
    *,
    expected_text: str,
) -> tuple[dict[str, str], Any]:
    """Validate a complete response and concatenate only its SAY token hiddens.

    This is shared by the realtime server and startup warmup so the Talker
    always receives hidden states aligned to the exact public ``say`` string.
    Private ``heard``/``action`` fields and JSON structure tokens are excluded.
    """
    import torch

    parsed = collector.finish(expected_text=expected_text)
    say_tokens = collector.say_tokens
    if not say_tokens:
        collector.parser._reject("say_hidden_missing")
    hidden_parts = [token.hidden for token in say_tokens]
    if any(not isinstance(part, torch.Tensor) or part.ndim != 3 for part in hidden_parts):
        collector.parser._reject("say_hidden_invalid_shape")
    batch_size = int(hidden_parts[0].shape[0])
    hidden_dim = int(hidden_parts[0].shape[2])
    if batch_size != 1 or any(
        int(part.shape[0]) != batch_size
        or int(part.shape[2]) != hidden_dim
        for part in hidden_parts
    ):
        collector.parser._reject("say_hidden_incompatible_shape")
    return parsed, torch.cat(hidden_parts, dim=1)


class AudioResponseV2ActionConstraint:
    """Keep the native action inside the controller-approved response set.

    A singleton ``expected_action`` is constrained to that exact class. A
    multi-action allowlist leaves the choice to the model and supplies only
    legal existing-token fallbacks. The tokenizer is used only for reversible
    text-to-existing-token encoding; no vocabulary entries or embeddings are
    changed. If a token crosses the action boundary, the collector validates
    its complete decoded prefix and fails closed on a disallowed action.
    """

    def __init__(
        self,
        *,
        expected_action: str | None,
        allowed_actions: Iterable[str] | None = None,
        tokenizer: Any,
        enforce_json_structure: bool = False,
    ) -> None:
        self.expected_action = (
            _normalize_action(expected_action) if expected_action is not None else None
        )
        self.allowed_actions = _normalize_allowed_actions(allowed_actions)
        if self.expected_action is not None:
            if self.allowed_actions is not None and self.allowed_actions != (self.expected_action,):
                raise ValueError("expected_action and allowed_actions conflict")
            self.allowed_actions = (self.expected_action,)
        self._constrain_action = expected_action is not None or allowed_actions is not None
        self.tokenizer = tokenizer
        self.enforce_json_structure = bool(enforce_json_structure)
        self._initial_started = False
        self._started = False
        self._pending_token_ids: list[tuple[int, str]] = []
        self.forced_token_count = 0
        self.forced_action_token_count = 0
        self.forced_structure_token_count = 0
        self.last_forced_kind: str | None = None
        self.bounded_output = False
        self._backchannel_lexicon_token_ids_by_prefix: dict[str, tuple[int, ...]] | None = None
        self.budget_preflight: dict[str, int] | None = None

    @property
    def started(self) -> bool:
        return self._started

    def _encode_roundtrippable(self, text: str, *, reason: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(encode) or not callable(decode):
            raise AudioResponseProtocolError("expected_action_tokenizer_api_unavailable")
        try:
            encoded = encode(text, add_special_tokens=False)
            if callable(getattr(encoded, "detach", None)):
                values = encoded.detach().cpu().reshape(-1).tolist()
            elif callable(getattr(encoded, "tolist", None)):
                values = encoded.tolist()
            else:
                values = list(encoded)
            while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            token_ids = [int(value) for value in values]
            decoded = decode(token_ids, skip_special_tokens=True)
        except Exception as exc:
            raise AudioResponseProtocolError("expected_action_tokenization_failed") from exc
        if not token_ids or decoded != text:
            raise AudioResponseProtocolError(reason)
        return token_ids

    def _force_text(self, text: str, *, kind: str) -> int:
        token_ids = self._encode_roundtrippable(
            text,
            reason=(
                "expected_action_tokenization_not_roundtrippable"
                if kind == "action"
                else "json_structure_tokenization_not_roundtrippable"
            ),
        )
        self._pending_token_ids.extend((token_id, kind) for token_id in token_ids[1:])
        self.last_forced_kind = kind
        self.forced_token_count += 1
        if kind == "action":
            self.forced_action_token_count += 1
        else:
            self.forced_structure_token_count += 1
        return token_ids[0]

    def preflight_generation_budget(
        self,
        collector: AudioResponseV2Collector,
        *,
        max_new_tokens: int,
        max_heard_tokens: int = AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS,
        max_action_tokens: int = AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS,
        max_say_tokens: int = AUDIO_RESPONSE_V2_MAX_SAY_TOKENS,
    ) -> dict[str, int]:
        """Prove the field caps, forced JSON pieces, and EOS fit this budget.

        Fixed grammar pieces are counted using round-tripping encodings from
        the active tokenizer. If even the bounded worst case cannot finish,
        fail before the multimodal Thinker prefill.
        """
        heard_cap = 0 if self.expected_action == "BACKCHANNEL" else int(max_heard_tokens)
        say_cap = int(max_say_tokens)
        if self.expected_action == "BACKCHANNEL" and collector.backchannel_say_lexicon_active:
            phrase_lengths: list[int] = []
            for phrase in collector._backchannel_say_phrase_texts:
                for value in (phrase, " " + phrase):
                    phrase_lengths.append(len(self._encode_roundtrippable(
                        value,
                        reason="backchannel_phrase_budget_tokenization_failed",
                    )))
            if phrase_lengths:
                say_cap = min(say_cap, max(phrase_lengths))

        # These strings mirror the exact pieces emitted by next_token_id.
        # Field bodies are accounted separately by their token caps.
        backchannel = self.expected_action == "BACKCHANNEL"
        heard_open = '"heard":""' if backchannel else '"heard":"'
        heard_close = () if backchannel else ('"',)
        fixed_fragments = (
            "{",
            heard_open,
            *heard_close,
            ',"action":"',
            '"',
            ',"say":"',
            '"',
            "}",
        )
        structural_tokens = sum(
            len(self._encode_roundtrippable(
                fragment,
                reason="json_structure_budget_tokenization_failed",
            ))
            for fragment in fixed_fragments
        )
        action_cap = int(max_action_tokens)
        action_labels = self.allowed_actions or AUDIO_RESPONSE_ACTIONS
        action_label_token_max = max(
            len(self._encode_roundtrippable(
                action,
                reason="audio_action_budget_tokenization_failed",
            ))
            for action in action_labels
        )
        if action_label_token_max > action_cap:
            raise AudioResponseProtocolError("v2_action_token_cap_too_small")
        required = heard_cap + action_cap + say_cap + structural_tokens + 1
        result = {
            "max_new_tokens": int(max_new_tokens),
            "heard_token_cap": heard_cap,
            "action_token_cap": action_cap,
            "say_token_cap": say_cap,
            "fixed_structure_tokens": structural_tokens,
            "eos_reserve_tokens": 1,
            "required_new_tokens": required,
        }
        if required > int(max_new_tokens):
            raise AudioResponseProtocolError("v2_generation_budget_preflight_failed")
        self.budget_preflight = result
        return dict(result)

    def candidate_within_generation_budget(
        self,
        collector: AudioResponseV2Collector,
        text: str,
        *,
        terminal: bool,
        generated_tokens: int,
        budget: dict[str, int],
        max_say_words: int = 4,
    ) -> bool:
        """Check a prospective token against field, escape, and total reserves."""
        if int(generated_tokens) + 1 > int(budget["max_new_tokens"]):
            return False
        if not collector.is_valid_next_text(
            text,
            terminal=terminal,
            max_say_words=max_say_words,
            max_heard_tokens=int(budget["heard_token_cap"]),
            max_action_tokens=int(budget["action_token_cap"]),
            max_say_tokens=int(budget["say_token_cap"]),
        ):
            return False
        if terminal or not text:
            return True

        phase = collector.parser.phase
        if phase not in {"heard_value", "action_value", "say_value"}:
            return True
        raw_before = "".join(collector.parser._string_raw)
        # Most ordinary tokens cannot start or continue an escape. Avoid a
        # second parser copy and tokenizer call on that hot path.
        existing_escape_suffix = (
            _json_string_escape_completion_suffix(raw_before)
            if "\\" in raw_before
            else ""
        )
        if existing_escape_suffix is None:
            return False
        if "\\" not in text and not existing_escape_suffix:
            return True
        preview = copy.deepcopy(collector.parser)
        try:
            preview.feed(text)
        except AudioResponseProtocolError:
            return False
        if preview.phase not in {"heard_value", "action_value", "say_value"}:
            return True
        suffix = _json_string_escape_completion_suffix("".join(preview._string_raw))
        if suffix is None or not suffix:
            return suffix == ""

        if preview.phase == "heard_value":
            field_count = collector.heard_field_token_count + int(
                preview.last_feed_heard_raw_span is not None
            )
            field_cap = int(budget["heard_token_cap"])
        elif preview.phase == "action_value":
            field_count = collector.action_field_token_count + int(
                preview.last_feed_action_raw_span is not None
            )
            field_cap = int(budget["action_token_cap"])
        else:
            field_count = collector.say_field_token_count + int(
                preview.last_feed_say_raw_span is not None
            )
            field_cap = int(budget["say_token_cap"])
        try:
            completion_tokens = len(self._encode_roundtrippable(
                suffix,
                reason="json_escape_reserve_tokenization_failed",
            ))
        except AudioResponseProtocolError:
            return False
        # One extra tokenizer boundary token covers a continuation whose BPE
        # merge differs from encoding the suffix in isolation.
        return field_count + completion_tokens + 1 <= field_cap

    def action_value_candidate_token_ids(
        self,
        collector: AudioResponseV2Collector,
    ) -> tuple[int, ...]:
        """Return a tokenizer-derived pool for the remaining selected action.

        This is a finite fallback when the model's top candidates all violate
        the action DFA. The decoder still checks every candidate in full
        context and selects the legal one with the highest native logit.
        """
        parser = collector.parser
        if (
            not self.enforce_json_structure
            or not self._constrain_action
            or parser.phase != "action_value"
            or parser._string_escaped
        ):
            return ()
        prefix = "".join(parser._string_raw)
        actions = self.allowed_actions or AUDIO_RESPONSE_ACTIONS
        matching_actions = tuple(action for action in actions if action.startswith(prefix))
        if not matching_actions:
            return ()
        candidates: set[int] = set()
        for action in matching_actions:
            remainder = action[len(prefix):]
            texts = ('"',) if not remainder else (remainder,)
            for text in texts:
                try:
                    candidates.add(self._encode_roundtrippable(
                        text,
                        reason="expected_action_tokenization_not_roundtrippable",
                    )[0])
                except AudioResponseProtocolError:
                    continue
        return tuple(sorted(candidates))

    def bounded_field_closure_token_id(
        self,
        collector: AudioResponseV2Collector,
        rejected_candidates: list[tuple[int, str, bool]],
        *,
        budget: dict[str, int],
    ) -> int | None:
        """Close a capped JSON string only at a validated, safe boundary."""
        parser = collector.parser
        if (
            not self.enforce_json_structure
            or collector._cancelled
            or collector._finished
            or parser.phase not in {"heard_value", "action_value", "say_value"}
            or parser._string_escaped
            or _json_string_escape_completion_suffix("".join(parser._string_raw)) != ""
            or not rejected_candidates
        ):
            return None
        if parser.phase == "heard_value":
            field_count, field_cap = collector.heard_field_token_count, budget["heard_token_cap"]
        elif parser.phase == "action_value":
            field_count, field_cap = collector.action_field_token_count, budget["action_token_cap"]
        else:
            field_count, field_cap = collector.say_field_token_count, budget["say_token_cap"]
        if field_count < field_cap:
            return None

        if parser.phase == "say_value":
            current_say = parser.pending_say
            safe_boundary = bool(
                collector.backchannel_say_lexicon_active
                and collector.backchannel_say_is_complete_phrase
            ) or bool(
                current_say
                and (current_say[-1].isspace() or current_say[-1] in ".,!?:;–—-")
            )
            if not safe_boundary:
                for _token_id, text, terminal in rejected_candidates:
                    if terminal or not text or not collector.is_valid_next_text(text, terminal=False):
                        continue
                    preview = copy.deepcopy(parser)
                    try:
                        preview.feed(text)
                    except AudioResponseProtocolError:
                        continue
                    delta = preview.pending_say[len(current_say):]
                    if delta and (delta[0].isspace() or current_say.endswith(tuple(".!?:;–—-"))):
                        safe_boundary = True
                        break
            if not safe_boundary:
                return None

        quote_ids = self._encode_roundtrippable(
            '"',
            reason="json_structure_tokenization_not_roundtrippable",
        )
        quote_text = self._decode_tokens(quote_ids)
        preview = copy.deepcopy(parser)
        try:
            preview.feed(quote_text)
        except AudioResponseProtocolError:
            return None
        if preview.phase == parser.phase:
            return None

        self.bounded_output = True
        if parser.phase == "heard_value":
            collector.heard_truncated = True
        elif parser.phase == "say_value":
            collector.say_truncated = True
        return self._force_text('"', kind="structure")

    def _decode_tokens(self, token_ids: list[int]) -> str:
        try:
            return str(self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
        except TypeError:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def bounded_say_closure_token_id(
        self,
        collector: AudioResponseV2Collector,
        rejected_candidates: list[tuple[int, str, bool]],
        *,
        max_say_words: int = 4,
    ) -> int | None:
        """Close a bounded BACKCHANNEL only at a proven next-word boundary.

        This is a last-resort structural token used after the decoder has
        rejected every high-ranked candidate solely because it would add a
        fifth word. A candidate that continues the current word, malformed
        JSON, EOS, or an unfinished Unicode escape never authorizes closure.
        """
        parser = collector.parser
        if (
            not self.enforce_json_structure
            or collector._cancelled
            or collector._finished
            or parser.phase != "say_value"
            or parser.action != "BACKCHANNEL"
            or parser._say_drain_offset != len(parser._string_raw)
            or parser._string_escaped
        ):
            return None

        if collector.backchannel_say_lexicon_active:
            if not collector.backchannel_say_is_complete_phrase:
                return None
        elif len(parser.pending_say.split()) != max_say_words:
            return None

        current_say = parser.pending_say
        if not collector.backchannel_say_lexicon_active:
            for _token_id, text, terminal in rejected_candidates:
                if terminal or not text or not collector.would_exceed_say_word_limit(
                    text, max_say_words=max_say_words
                ):
                    return None
                preview = copy.deepcopy(parser)
                try:
                    preview.feed(text)
                except AudioResponseProtocolError:
                    return None
                delta = preview.pending_say[len(current_say):]
                if not delta or not (current_say.endswith(tuple(" \t\r\n")) or delta[0].isspace()):
                    return None

        if not rejected_candidates:
            return None
        self.bounded_output = True
        return self._force_text('"', kind="structure")

    def backchannel_say_candidate_token_ids(
        self,
        collector: AudioResponseV2Collector,
    ) -> tuple[int, ...]:
        """Return a small tokenizer-derived pool of legal next BACKCHANNEL IDs.

        The pool comes only from the active train-split phrase lexicon. It is
        keyed by the normalized decoded prefix and cached for this response;
        callers still run each candidate through the full incremental JSON
        parser before selecting it by the model's native logit.
        """
        parser = collector.parser
        if (
            not self.enforce_json_structure
            or not collector.backchannel_say_lexicon_active
            or parser.action != "BACKCHANNEL"
            or parser.phase != "say_value"
            or parser._string_escaped
            or parser._say_drain_offset != len(parser._string_raw)
        ):
            return ()

        if self._backchannel_lexicon_token_ids_by_prefix is None:
            self._backchannel_lexicon_token_ids_by_prefix = (
                self._compile_backchannel_lexicon_token_ids(collector)
            )
        prefix = _normalize_backchannel_phrase(parser.pending_say)
        cached = self._backchannel_lexicon_token_ids_by_prefix.get(prefix)

        # Some tokenizers merge a full word with its continuation, so a
        # generated subword may not occur as a standalone decoded trie node,
        # and one phrase's cached tokenization may not match the active prefix.
        # Union context-independent trie IDs with encodings of each matching
        # suffix, then let the full-context JSON callback validate candidates.
        encode = getattr(self.tokenizer, "encode", None)
        candidates: set[int] = set(cached or ())
        if callable(encode) and prefix:
            for phrase in collector._backchannel_say_phrase_texts:
                for index in range(len(phrase) + 1):
                    if _normalize_backchannel_phrase(phrase[:index]) != prefix:
                        continue
                    suffix = phrase[index:]
                    for text in (suffix, " " + suffix):
                        values = encode(text, add_special_tokens=False)
                        if callable(getattr(values, "detach", None)):
                            values = values.detach().cpu().reshape(-1).tolist()
                        elif callable(getattr(values, "tolist", None)):
                            values = values.tolist()
                        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                            values = values[0]
                        if values:
                            candidates.add(int(values[0]))
                    if not suffix:
                        try:
                            candidates.update(self._encode_roundtrippable(
                                '"', reason="json_structure_tokenization_not_roundtrippable"
                            ))
                        except AudioResponseProtocolError:
                            pass
                    break
        result = tuple(sorted(candidates))
        self._backchannel_lexicon_token_ids_by_prefix[prefix] = result
        return result

    def _compile_backchannel_lexicon_token_ids(
        self,
        collector: AudioResponseV2Collector,
    ) -> dict[str, tuple[int, ...]]:
        encode = getattr(self.tokenizer, "encode", None)
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(encode) or not callable(decode):
            return {}

        def token_ids(text: str) -> tuple[int, ...]:
            values = encode(text, add_special_tokens=False)
            if callable(getattr(values, "detach", None)):
                values = values.detach().cpu().reshape(-1).tolist()
            elif callable(getattr(values, "tolist", None)):
                values = values.tolist()
            while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            return tuple(int(value) for value in values)

        def decoded_prefix(values: tuple[int, ...]) -> str:
            try:
                return str(decode(
                    list(values),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ))
            except TypeError:
                return str(decode(list(values), skip_special_tokens=True))

        next_ids: dict[str, set[int]] = {}
        for phrase in collector._backchannel_say_phrase_texts:
            # SentencePiece and byte-level BPE can encode the first word with
            # or without a leading word-boundary space. Both variants are
            # checked later against the actual generated JSON prefix.
            for text in (phrase, " " + phrase):
                sequence = token_ids(text)
                for index, token_id in enumerate(sequence):
                    decoded = decoded_prefix(sequence[:index])
                    key = _normalize_backchannel_phrase(decoded)
                    next_ids.setdefault(key, set()).add(token_id)
                if sequence:
                    complete_key = _normalize_backchannel_phrase(decoded_prefix(sequence))
                    for quote_id in token_ids('"'):
                        next_ids.setdefault(complete_key, set()).add(quote_id)

        return {key: tuple(sorted(values)) for key, values in next_ids.items()}

    def initial_token_id(self) -> int | None:
        """Return the existing-token JSON opener for opt-in structured decoding."""
        if not self.enforce_json_structure or self._initial_started:
            return None
        self._initial_started = True
        return self._force_text("{", kind="structure")

    def next_token_id(self, collector: AudioResponseV2Collector) -> int | None:
        if self._pending_token_ids:
            token_id, kind = self._pending_token_ids.pop(0)
            self.last_forced_kind = kind
            self.forced_token_count += 1
            if kind == "action":
                self.forced_action_token_count += 1
            else:
                self.forced_structure_token_count += 1
            return token_id
        self.last_forced_kind = None
        if self.enforce_json_structure:
            parser = collector.parser
            phase = parser.phase
            if parser.action == "BACKCHANNEL" and phase == "say_value":
                say = parser.pending_say
                if (
                    len(say.split()) >= 4
                    and say
                    and (say[-1].isspace() or say[-1] in ".,!?:;–—-")
                ):
                    self.bounded_output = True
                    return self._force_text('"', kind="structure")
            if phase == "heard_prefix" and self._initial_started:
                suffix = '"heard":""' if self.expected_action == "BACKCHANNEL" else '"heard":"'
                return self._force_text(suffix, kind="structure")
            if (
                phase == "heard_value"
                and self.expected_action == "BACKCHANNEL"
                and not parser._string_raw
            ):
                return self._force_text('"', kind="structure")
            if phase == "action_prefix":
                return self._force_text(',"action":"', kind="structure")
            if phase == "say_prefix":
                return self._force_text(',"say":"', kind="structure")
            if phase == "closing_brace":
                return self._force_text("}", kind="structure")
            if (
                phase == "action_value"
                and self.expected_action is not None
            ):
                action_prefix = "".join(parser._string_raw)
                if self.expected_action.startswith(action_prefix):
                    remainder = self.expected_action[len(action_prefix):]
                    if remainder:
                        self._started = True
                        return self._force_text(remainder, kind="action")
                    return self._force_text('"', kind="structure")
        if (
            self.expected_action is None
            or self._started
            or not collector.action_value_pending
        ):
            return None
        self._started = True
        return self._force_text(self.expected_action, kind="action")


__all__ = [
    "AUDIO_RESPONSE_ACTIONS",
    "AUDIO_RESPONSE_PROTOCOL_V2",
    "CONTROLLER_GUIDED_SAY_V1",
    "AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM",
    "AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS",
    "AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS",
    "AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS",
    "AUDIO_RESPONSE_V2_MAX_NEW_TOKENS",
    "AUDIO_RESPONSE_V2_MAX_SAY_TOKENS",
    "AudioResponseV2ActionConstraint",
    "AudioResponseProtocolError",
    "AudioResponseV2Collector",
    "AudioResponseV2Parser",
    "AudioResponseV2SayToken",
    "audio_response_disposition_for_actions",
    "audio_response_actions_for_wake",
    "controller_guided_response_action",
    "controller_backchannel_candidate_allowed",
    "controller_backchannel_should_stop",
    "finish_audio_response_v2_with_say_hidden",
    "format_audio_response_v2_target",
    "normalize_audio_response_protocol",
    "slice_controller_guided_say_hidden",
]
