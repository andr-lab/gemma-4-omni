"""Token-preserving batching of incremental Gemma SAY text into spoken units."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from duplex_kernel.session import SpokenTokenPacket


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class SpokenSourceToken:
    """One original Gemma token and its exclusive KV position.

    Empty ``text_piece`` entries carry source-token ownership without creating
    Qwen/FIFO packets. Visible entries retain their original packet object.
    """

    token_id: int
    exclusive_kv_cursor: int
    text_piece: str
    speech_packet: SpokenTokenPacket | None = None


@dataclass(frozen=True)
class SpokenTextUnit:
    """One closed or final partial text unit with exact source-token ownership."""

    text: str
    token_packets: tuple[SpokenTokenPacket, ...]
    exclusive_kv_cursor: int
    source_tokens: tuple[SpokenSourceToken, ...]

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Original Gemma IDs, including source tokens with no visible delta."""
        return tuple(source.token_id for source in self.source_tokens)


class SpokenTextUnitAssembler:
    """Batch original SAY packets without retokenizing or splitting packets.

    The word limit is soft at packet boundaries: a packet is indivisible, and a
    unit closes only when its accumulated text reaches a Unicode whitespace or
    punctuation boundary after at least ``max_words`` words.
    """

    def __init__(self, max_words: int = 2) -> None:
        if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
            raise ValueError("max_words deve essere un intero positivo")
        self.max_words = max_words
        self._source_tokens: list[SpokenSourceToken] = []
        # Empty visible deltas are ambiguous until another SAY token arrives:
        # they may be protocol/control IDs inside the spoken prefix, or closing
        # protocol IDs after SAY. Hold them aside so finish() can discard the
        # latter without losing the former.
        self._pending_empty_tokens: list[SpokenSourceToken] = []
        self._text_parts: list[str] = []
        self._last_cursor: int | None = None
        self._closed = False

    def push(self, packet: SpokenTokenPacket) -> list[SpokenTextUnit]:
        """Append one visible packet, preserving its exact identity and IDs."""
        if not isinstance(packet, SpokenTokenPacket):
            raise TypeError("packet deve essere SpokenTokenPacket")
        return self.push_source(
            token_id=packet.token_id,
            exclusive_kv_cursor=packet.exclusive_kv_cursor,
            text_piece=packet.text_piece,
            speech_packet=packet,
        )

    def push_source(
        self,
        token_id: int,
        exclusive_kv_cursor: int,
        text_piece: str,
        *,
        speech_packet: SpokenTokenPacket | None = None,
    ) -> list[SpokenTextUnit]:
        """Append a source token; empty visible text carries ownership only."""
        if self._closed:
            raise RuntimeError("spoken text unit assembler is finished")
        for name, value in (("token_id", token_id), ("exclusive_kv_cursor", exclusive_kv_cursor)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} deve essere un intero non negativo")
        if self._last_cursor is not None and exclusive_kv_cursor <= self._last_cursor:
            raise ValueError("exclusive KV cursor non monotono")
        if not isinstance(text_piece, str):
            raise TypeError("text_piece deve essere string")
        if speech_packet is None:
            if text_piece:
                raise ValueError("un visible SAY delta richiede il suo SpokenTokenPacket originale")
        elif not isinstance(speech_packet, SpokenTokenPacket):
            raise TypeError("speech_packet deve essere SpokenTokenPacket o None")
        elif (
            not text_piece
            or speech_packet.token_id != token_id
            or speech_packet.exclusive_kv_cursor != exclusive_kv_cursor
            or speech_packet.text_piece != text_piece
        ):
            raise ValueError("speech_packet non corrisponde ai metadata source token")

        source = SpokenSourceToken(
            token_id=token_id,
            exclusive_kv_cursor=exclusive_kv_cursor,
            text_piece=text_piece,
            speech_packet=speech_packet,
        )
        self._last_cursor = exclusive_kv_cursor
        if not text_piece:
            self._pending_empty_tokens.append(source)
            return []

        if self._pending_empty_tokens:
            self._source_tokens.extend(self._pending_empty_tokens)
            self._pending_empty_tokens.clear()
        self._source_tokens.append(source)
        self._text_parts.append(text_piece)
        if self._is_boundary(text_piece[-1]):
            text = "".join(self._text_parts)
            if len(_WORD_PATTERN.findall(text)) >= self.max_words:
                return [self._take_unit(text)]
        return []

    def finish(self) -> list[SpokenTextUnit]:
        """Close once, returning remaining visible text and dropping empty carry."""
        if self._closed:
            return []
        self._closed = True
        if not self._text_parts:
            self._source_tokens.clear()
            self._pending_empty_tokens.clear()
            return []
        # A unit's exclusive cursor belongs to its final visible SAY token.
        # Empty source IDs after it are post-SAY protocol close and are not
        # part of the reconstructed spoken prefix.
        self._pending_empty_tokens.clear()
        return [self._take_unit("".join(self._text_parts))]

    @staticmethod
    def _is_boundary(char: str) -> bool:
        return char.isspace() or unicodedata.category(char).startswith("P")

    def _take_unit(self, text: str) -> SpokenTextUnit:
        source_tokens = tuple(self._source_tokens)
        token_packets = tuple(
            source.speech_packet
            for source in source_tokens
            if source.speech_packet is not None
        )
        if not source_tokens or not token_packets or not text:
            raise RuntimeError("spoken unit deve possedere testo e packet visibili")
        unit = SpokenTextUnit(
            text=text,
            token_packets=token_packets,
            exclusive_kv_cursor=source_tokens[-1].exclusive_kv_cursor,
            source_tokens=source_tokens,
        )
        self._source_tokens = []
        self._text_parts = []
        return unit
