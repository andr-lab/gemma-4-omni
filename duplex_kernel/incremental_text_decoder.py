from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class DecodedSourceToken:
    """One committed source token, including tokens with an empty text delta."""

    token_id: int
    source_sequence: int | None
    source_token_id: int | None
    text_delta: str


class IncrementalTextDecoder:
    """Incrementally decode token IDs while keeping speculative previews isolated.

    Fast tokenizers expose their Rust ``backend_tokenizer`` and tokenizers
    0.22.2's ``DecodeStream`` can emit a stable delta without decoding the
    complete response prefix. Small test doubles and slow-tokenizer adapters
    retain the compatible full-decode fallback.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        skip_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = bool(skip_special_tokens)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None and callable(getattr(tokenizer, "get_vocab_size", None)):
            backend = tokenizer
        self.backend_tokenizer = backend
        self._decode_stream = None
        if backend is not None:
            try:
                from tokenizers.decoders import DecodeStream
            except ImportError as exc:
                raise RuntimeError("fast tokenizer backend requires tokenizers DecodeStream") from exc
            self._decode_stream_type = DecodeStream
            self._decode_stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)
        else:
            self._decode_stream_type = None
        self._token_ids: list[int] = []
        self._source_tokens: list[DecodedSourceToken] = []
        self._pending_text_sources: list[DecodedSourceToken] = []
        self._last_text_sources: tuple[DecodedSourceToken, ...] = ()
        self._text = ""
        self._finished = False

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self._token_ids)

    @property
    def source_tokens(self) -> tuple[DecodedSourceToken, ...]:
        return tuple(self._source_tokens)

    @property
    def text(self) -> str:
        return self._text

    @property
    def last_text_sources(self) -> tuple[DecodedSourceToken, ...]:
        """Source tokens that contributed to the most recent visible delta."""
        return self._last_text_sources

    @property
    def last_text_source_group(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(source.source_sequence), int(source.source_token_id))
            for source in self._last_text_sources
            if source.source_sequence is not None and source.source_token_id is not None
        )

    def _validate_token_id(self, token_id: int) -> int:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TypeError("token_id must be an integer")
        value = int(token_id)
        if value < 0:
            raise ValueError("token_id must be nonnegative")
        return value

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("incremental text decoder is already finalized")

    def _fallback_decode(self, token_ids: list[int]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return ""
        try:
            result = decode(
                token_ids,
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            result = decode(token_ids, skip_special_tokens=self.skip_special_tokens)
        if not isinstance(result, str):
            raise TypeError("tokenizer.decode must return str")
        return result

    def preview(self, token_id: int) -> str:
        """Return a candidate's text delta without changing committed state."""
        self._ensure_open()
        value = self._validate_token_id(token_id)
        if self.backend_tokenizer is not None:
            preview_stream = self._decode_stream_type(
                ids=self._token_ids,
                skip_special_tokens=self.skip_special_tokens,
            )
            piece = preview_stream.step(self.backend_tokenizer, value)
            return "" if piece is None else str(piece)
        candidate = self._fallback_decode([*self._token_ids, value])
        if not candidate.startswith(self._text):
            raise RuntimeError("Thinker token decode is not prefix-stable for candidate validation")
        return candidate[len(self._text):]

    def append(
        self,
        token_id: int,
        *,
        source_sequence: int | None = None,
        source_token_id: int | None = None,
    ) -> str:
        """Commit one token and retain its source identity even for empty output."""
        self._ensure_open()
        value = self._validate_token_id(token_id)
        if source_sequence is not None:
            if isinstance(source_sequence, bool) or not isinstance(source_sequence, Integral):
                raise TypeError("source_sequence must be an integer or None")
            source_sequence = int(source_sequence)
            if source_sequence < 0:
                raise ValueError("source_sequence must be nonnegative")
        if source_token_id is not None:
            source_token_id = self._validate_token_id(source_token_id)
        elif source_sequence is not None:
            # Source sequence ownership is still meaningful for an empty
            # DecodeStream delta (for example, one byte of a UTF-8 fallback).
            # When callers omit the parallel source ID, the decoded token is
            # itself the source token.
            source_token_id = value
        if self.backend_tokenizer is not None:
            piece = self._decode_stream.step(self.backend_tokenizer, value)
            delta = "" if piece is None else str(piece)
        else:
            decoded = self._fallback_decode([*self._token_ids, value])
            if not decoded.startswith(self._text):
                raise RuntimeError("Thinker token decode is not prefix-stable for generated output")
            delta = decoded[len(self._text):]
        self._token_ids.append(value)
        source = DecodedSourceToken(
            token_id=value,
            source_sequence=source_sequence,
            source_token_id=source_token_id,
            text_delta=delta,
        )
        self._source_tokens.append(source)
        if delta:
            self._last_text_sources = tuple((*self._pending_text_sources, source))
            self._pending_text_sources.clear()
        else:
            self._pending_text_sources.append(source)
            self._last_text_sources = ()
        self._text += delta
        return delta

    def ensure_prefix(self, token_ids: list[int] | tuple[int, ...]) -> None:
        """Synchronize a fresh/forked sampler to its committed token history."""
        self._ensure_open()
        expected = [self._validate_token_id(value) for value in token_ids]
        if expected == self._token_ids:
            return
        if len(expected) >= len(self._token_ids) and expected[:len(self._token_ids)] == self._token_ids:
            for value in expected[len(self._token_ids):]:
                self.append(value)
            return
        self._reset_to(expected)

    def _reset_to(self, token_ids: list[int]) -> None:
        clone = IncrementalTextDecoder(
            self.tokenizer,
            skip_special_tokens=self.skip_special_tokens,
        )
        for value in token_ids:
            clone.append(value)
        self._decode_stream = clone._decode_stream
        self._token_ids = clone._token_ids
        self._source_tokens = clone._source_tokens
        self._pending_text_sources = clone._pending_text_sources
        self._last_text_sources = clone._last_text_sources
        self._text = clone._text

    def clone(self) -> IncrementalTextDecoder:
        """Copy the committed decoder state for an isolated native branch."""
        clone = IncrementalTextDecoder(
            self.tokenizer,
            skip_special_tokens=self.skip_special_tokens,
        )
        for source in self._source_tokens:
            clone.append(
                source.token_id,
                source_sequence=source.source_sequence,
                source_token_id=source.source_token_id,
            )
        return clone

    def finish(self, token_ids: list[int] | tuple[int, ...] | None = None) -> str:
        """Flush the stable final suffix and match canonical tokenizer.decode.

        The canonical decode occurs once, at termination. It is required for
        incomplete byte fallback sequences, whose replacement characters are
        not emitted by DecodeStream until the end-of-response boundary.
        """
        if self._finished:
            return ""
        if token_ids is not None:
            self.ensure_prefix(token_ids)
        decoded = self._fallback_decode(self._token_ids)
        if not decoded.startswith(self._text):
            raise RuntimeError("incremental decode changed text that was already emitted")
        suffix = decoded[len(self._text):]
        self._text = decoded
        self._last_text_sources = (
            tuple(self._pending_text_sources) if suffix else ()
        )
        self._pending_text_sources.clear()
        self._finished = True
        return suffix


__all__ = ["DecodedSourceToken", "IncrementalTextDecoder"]
