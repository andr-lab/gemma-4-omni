"""Small async decode loop over the native listener's single KV cache.

The caller owns cache state and supplies only the transition token IDs that
must be appended to the already-prefilled system/user/audio context. Each
forward callback must atomically commit those IDs through the native path and
return logits plus the final-normalized hidden state for its last logical
token. This module never builds an HF cache or forwards audio.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import threading
import time
import weakref
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral
from typing import Any, Callable, Sequence

import torch

from duplex_kernel.incremental_text_decoder import DecodedSourceToken, IncrementalTextDecoder
from duplex_kernel.live_say import LiveSayCancelled
from duplex_kernel.models.thinker import normalize_eos_token_ids
from duplex_kernel.thinker_text_budget import ends_at_sentence_boundary


DEFAULT_CONTINUATION_GENERATED_BUDGET = 768


@dataclass(frozen=True)
class TokenTiming:
    """Sampling and own-token-forward timestamps for one published token."""

    forward_started_ns: int
    logits_ready_ns: int
    sampled_ns: int
    forward_duration_ns: int
    generation_elapsed_ns: int
    token_forward_started_ns: int | None
    token_hidden_ready_ns: int | None
    token_forward_duration_ns: int | None
    token_ready_elapsed_ns: int | None


@dataclass(frozen=True)
class FirstLogitsReady:
    logits: torch.Tensor
    final_hidden: torch.Tensor
    ready_ns: int
    elapsed_ns: int


@dataclass(frozen=True)
class GeneratedToken:
    """A sampled non-EOS token paired with its explicitly sourced hidden state."""

    token_id: int
    hidden: torch.Tensor
    timing: TokenTiming
    token_index: int = 0
    logical_source_sequence: int | None = None
    hidden_provenance: str = "token"
    kv_committed: bool = True
    packet_published_ns: int | None = None
    token_text: str = ""
    text_source_sequences: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class NativeDecodeResult:
    generated_tokens: tuple[GeneratedToken, ...]
    termination_reason: str
    terminal_token_id: int | None
    generated_count: int
    committed_count: int
    sampled_count: int
    transition_token_count: int
    first_logits_ready: FirstLogitsReady | None
    first_token: GeneratedToken | None
    pending_token_id: int | None
    owner_action: str

    @property
    def finish_reason(self) -> str:
        """Expose protocol finish semantics while preserving the raw stop cause."""
        if self.termination_reason == "eos":
            return "stop"
        if self.termination_reason == "cancelled":
            return "cancelled"
        # Both explicit length and a consumed generation budget are truncation.
        return "length"


class NativeDecodeSessionStatus(str, Enum):
    """Status returned by one bounded step of a resumable decode."""

    READY = "ready"
    MORE = "more"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class NativeDecodeProgress:
    """Public text and token view; the sampled lookahead stays session-private."""

    generated_token_ids: tuple[int, ...]
    generated_text: str
    stable_text: str
    source_tokens: tuple[DecodedSourceToken, ...]
    source_offset: int
    native_cursor: int | None


@dataclass(frozen=True)
class NativeDecodeSessionStep:
    """One bounded session result. ``generated_tokens`` is only this step's batch."""

    status: NativeDecodeSessionStatus
    generated_tokens: tuple[GeneratedToken, ...]
    progress: NativeDecodeProgress
    native_handle: Any | None
    native_cursor: int | None
    source_offset: int
    result: NativeDecodeResult | None = None


@dataclass
class _NativeDecodeSessionState:
    transition_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    budget: int
    hard_budget: int
    logical_offset: int
    anticipate_first_token: bool
    graceful_sentence_completion: bool
    first_logits_ready_callback: Callable[[FirstLogitsReady], Any] | None
    first_token_callback: Callable[[GeneratedToken], Any] | None
    on_token_callback: Callable[[int, torch.Tensor, TokenTiming], Any] | None
    on_token_packet_callback: Callable[[GeneratedToken], Any] | None
    on_token_committed_callback: Callable[[GeneratedToken], Any] | None
    on_text_flush_callback: Callable[[str, GeneratedToken], Any] | None
    phase: str = "prefill"
    prefill_offset: int = 0
    logits: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    candidate: Any = None
    candidate_sampled_ns: int = 0
    generated: list[GeneratedToken] = field(default_factory=list)
    sampled_count: int = 0
    committed_count: int = 0
    native_forwarded_count: int = 0
    token_index: int = 0
    first_logits_event: FirstLogitsReady | None = None
    terminal_token_id: int | None = None
    pending_token_id: int | None = None
    termination_reason: str | None = None
    decode_started_ns: int = 0
    source_forward_started_ns: int = 0
    source_logits_ready_ns: int = 0
    text_source_token_offset: int = 0


class NativeGenerationSession:
    """Explicit resumable state for a single native KV branch.

    Every ``advance`` call completes all awaitables in its caller's loop and
    stores only plain decoder/sampler state and tensors. No generator,
    coroutine, task, or future survives the call, so a caller may use
    ``asyncio.run`` for each step while keeping this object and its one native
    handle alive.
    """

    def __init__(
        self,
        decoder: UnifiedNativeDecoder,
        state: _NativeDecodeSessionState,
        *,
        native_handle: Any | None,
        native_cursor: int | None,
    ) -> None:
        self._decoder: UnifiedNativeDecoder | None = decoder
        self._state: _NativeDecodeSessionState | None = state
        self._native_handle = native_handle
        self._base_native_cursor = native_cursor
        self._lifecycle_lock = threading.RLock()
        self._cancel_requested = False
        self._in_step = False
        self._failed = False
        self._fatal_error: BaseException | None = None
        self._closed = False
        self._close_requested = False
        self._cleanup_requested = False
        self._cleanup_started = False
        self._final_step: NativeDecodeSessionStep | None = None
        self._final_step_replay: NativeDecodeSessionStep | None = None

    @property
    def native_handle(self) -> Any | None:
        with self._lifecycle_lock:
            return self._native_handle

    def adopt_native_handle(self, native_handle: Any) -> None:
        """Replace a promoted handle alias without rebuilding decoder state."""
        with self._lifecycle_lock:
            self.validate_native_handle_adoption(native_handle)
            self._native_handle = native_handle

    def validate_native_handle_adoption(self, native_handle: Any) -> None:
        """Preflight a handle transfer while a checkpoint is paused."""
        current_id = getattr(self._native_handle, "handle_id", None)
        next_id = getattr(native_handle, "handle_id", None)
        if current_id is None or next_id != current_id:
            raise RuntimeError("promoted native handle must preserve the session handle identity")

    @property
    def native_cursor(self) -> int | None:
        with self._lifecycle_lock:
            state = self._state
            if self._closed:
                return None
            if state is None and self._final_step is not None:
                return self._final_step.native_cursor
            forwarded = state.native_forwarded_count if state is not None else 0
            if self._base_native_cursor is None:
                return None
            return self._base_native_cursor + forwarded

    @property
    def source_offset(self) -> int:
        with self._lifecycle_lock:
            state = self._state
            if state is not None:
                return state.logical_offset + state.committed_count
            if self._final_step is not None:
                return self._final_step.source_offset
            return 0

    @property
    def generated_token_ids(self) -> tuple[int, ...]:
        with self._lifecycle_lock:
            state = self._state
            if state is not None:
                return tuple(token.token_id for token in state.generated)
            if self._final_step is not None and self._final_step.result is not None:
                return tuple(token.token_id for token in self._final_step.result.generated_tokens)
            return ()

    @property
    def status(self) -> NativeDecodeSessionStatus | None:
        with self._lifecycle_lock:
            if self._fatal_error is not None:
                return NativeDecodeSessionStatus.FAILED
            return self._final_step.status if self._final_step is not None else None

    @property
    def fatal_error(self) -> BaseException | None:
        with self._lifecycle_lock:
            return self._fatal_error

    def _live(self) -> tuple[UnifiedNativeDecoder, _NativeDecodeSessionState]:
        with self._lifecycle_lock:
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._closed:
                raise RuntimeError("native generation session is closed")
            if self._failed:
                raise RuntimeError("native generation session failed; close it before reuse")
            if self._decoder is None or self._state is None:
                raise RuntimeError("native generation session is no longer active")
            return self._decoder, self._state

    def _make_progress(
        self,
        decoder: UnifiedNativeDecoder,
        state: _NativeDecodeSessionState,
    ) -> NativeDecodeProgress:
        text_decoder = decoder._text_decoder
        source_tokens: tuple[DecodedSourceToken, ...] = ()
        stable_text = ""
        if isinstance(text_decoder, IncrementalTextDecoder):
            stable_text = text_decoder.text
            source_tokens = text_decoder.source_tokens[state.text_source_token_offset:]
        generated = tuple(state.generated)
        return NativeDecodeProgress(
            generated_token_ids=tuple(token.token_id for token in generated),
            generated_text="".join(token.token_text for token in generated),
            stable_text=stable_text,
            source_tokens=source_tokens,
            source_offset=state.logical_offset + state.committed_count,
            native_cursor=self.native_cursor,
        )

    def _make_step(
        self,
        status: NativeDecodeSessionStatus,
        *,
        batch_start: int,
        result: NativeDecodeResult | None = None,
    ) -> NativeDecodeSessionStep:
        decoder, state = self._live()
        batch = tuple(state.generated[batch_start:])
        progress = self._make_progress(decoder, state)
        step = NativeDecodeSessionStep(
            status=status,
            generated_tokens=batch,
            progress=progress,
            native_handle=self._native_handle,
            native_cursor=self.native_cursor,
            source_offset=state.logical_offset + state.committed_count,
            result=result,
        )
        if result is not None:
            self._complete(step)
        return step

    def _complete(self, step: NativeDecodeSessionStep) -> None:
        with self._lifecycle_lock:
            decoder = self._decoder
            if decoder is not None:
                active_ref = getattr(decoder, "_active_session_ref", None)
                if callable(active_ref) and active_ref() is self:
                    decoder._active_session_ref = None
            self._final_step = step
            self._final_step_replay = replace(step, generated_tokens=())
            self._state = None
            self._decoder = None
            if self._close_requested:
                self._release_closed_state()

    def _release_closed_state(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            self._decoder = None
            self._state = None
            self._native_handle = None

    def _request_safe_cleanup_once(self) -> None:
        """Drop live decoder state after the current step has unwound."""
        with self._lifecycle_lock:
            if self._cleanup_requested:
                return
            self._cleanup_requested = True
            should_cleanup = not self._in_step
        if should_cleanup:
            self._cleanup_once()

    def _cleanup_once(self) -> None:
        with self._lifecycle_lock:
            if self._cleanup_started or self._in_step:
                return
            self._cleanup_started = True
            decoder = self._decoder
            if decoder is not None:
                active_ref = getattr(decoder, "_active_session_ref", None)
                if callable(active_ref) and active_ref() is self:
                    decoder._active_session_ref = None
            self._release_closed_state()

    def _is_cancel_requested(self) -> bool:
        with self._lifecycle_lock:
            return self._cancel_requested

    def _is_expected_cancel(self, exc: BaseException) -> bool:
        return isinstance(exc, (asyncio.CancelledError, LiveSayCancelled)) or (
            self._is_cancel_requested()
        )

    def _record_cancelled_exception(
        self,
        decoder: UnifiedNativeDecoder,
        state: _NativeDecodeSessionState,
        *,
        batch_start: int,
    ) -> NativeDecodeSessionStep:
        pending = (
            _sampled_token_id(state.candidate)
            if state.candidate is not None else None
        )
        result = decoder._result(
            state.generated,
            "cancelled",
            state.terminal_token_id,
            state.committed_count,
            state.sampled_count,
            len(state.transition_ids),
            state.first_logits_event,
            pending_token_id=pending,
        )
        step = NativeDecodeSessionStep(
            status=NativeDecodeSessionStatus.CANCELLED,
            generated_tokens=tuple(state.generated[batch_start:]),
            progress=self._make_progress(decoder, state),
            native_handle=self._native_handle,
            native_cursor=self.native_cursor,
            source_offset=state.logical_offset + state.committed_count,
            result=result,
        )
        self._complete(step)
        return step

    def _replay_terminal_step(self) -> NativeDecodeSessionStep:
        with self._lifecycle_lock:
            assert self._final_step is not None
            replay = self._final_step_replay
            if replay is not None:
                return replay
            return replace(self._final_step, generated_tokens=())

    async def _make_terminal_result(
        self,
        decoder: UnifiedNativeDecoder,
        state: _NativeDecodeSessionState,
        *,
        termination_reason: str,
        pending_token_id: int | None = None,
    ) -> NativeDecodeResult:
        state.termination_reason = termination_reason
        state.pending_token_id = pending_token_id
        if termination_reason in {
            "eos", "budget_exhausted", "sentence_boundary_after_budget", "producer_limit"
        } and state.generated:
            final_ids = tuple([*state.prefix_ids, *(token.token_id for token in state.generated)])
            suffix = ""
            if decoder._text_decoder is not None:
                suffix = decoder._text_decoder.finish()
            sampler_flush = getattr(decoder.sampler, "flush_decoded_text", None)
            if callable(sampler_flush):
                flush_kwargs: dict[str, Any] = {
                    "final_hidden": state.generated[-1].hidden,
                }
                try:
                    if "notify_callback" in inspect.signature(sampler_flush).parameters:
                        flush_kwargs["notify_callback"] = False
                except (TypeError, ValueError):
                    pass
                sampler_suffix = str(sampler_flush(final_ids, **flush_kwargs) or "")
                if suffix and sampler_suffix != suffix:
                    raise RuntimeError("Thinker samplers disagree on the final text flush")
                if not suffix:
                    suffix = sampler_suffix
            if suffix:
                state.generated[-1] = replace(
                    state.generated[-1],
                    token_text=state.generated[-1].token_text + suffix,
                    text_source_sequences=(
                        decoder._text_decoder.last_text_source_group
                        if decoder._text_decoder is not None
                        else state.generated[-1].text_source_sequences
                    ),
                )
                if state.on_text_flush_callback is not None:
                    await _resolve(
                        state.on_text_flush_callback(suffix, state.generated[-1])
                    )
        return decoder._result(
            state.generated,
            termination_reason,
            state.terminal_token_id,
            state.committed_count,
            state.sampled_count,
            len(state.transition_ids),
            state.first_logits_event,
            pending_token_id=pending_token_id,
        )

    async def _terminate(
        self,
        termination_reason: str,
        *,
        batch_start: int,
        pending_token_id: int | None = None,
        no_output: bool = False,
    ) -> NativeDecodeSessionStep:
        decoder, state = self._live()
        if no_output:
            result = NativeDecodeResult(
                generated_tokens=(),
                termination_reason="budget_exhausted",
                terminal_token_id=None,
                generated_count=0,
                committed_count=0,
                sampled_count=0,
                transition_token_count=len(state.transition_ids),
                first_logits_ready=None,
                first_token=None,
                pending_token_id=None,
                owner_action="no_output",
            )
        else:
            result = await self._make_terminal_result(
                decoder,
                state,
                termination_reason=termination_reason,
                pending_token_id=pending_token_id,
            )
        status = (
            NativeDecodeSessionStatus.CANCELLED
            if termination_reason == "cancelled"
            else NativeDecodeSessionStatus.DONE
        )
        return self._make_step(status, batch_start=batch_start, result=result)

    def cancel(self) -> NativeDecodeSessionStep | None:
        """Cancel without flushing text; safe between ``advance`` calls.

        If an advance is currently awaiting caller code, this records a request
        that the next decoder boundary will honor. Paused sessions finish the
        cancellation synchronously and do no native forward work.
        """
        with self._lifecycle_lock:
            if self._closed:
                return self._replay_terminal_step() if self._final_step is not None else None
            if self._fatal_error is not None:
                return None
            if self._final_step is not None:
                return self._replay_terminal_step()
            self._cancel_requested = True
            if self._in_step:
                return None
            decoder, state = self._live()
            self._in_step = True

        try:
            pending = (
                _sampled_token_id(state.candidate)
                if state.candidate is not None else None
            )
            result = decoder._result(
                state.generated,
                "cancelled",
                state.terminal_token_id,
                state.committed_count,
                state.sampled_count,
                len(state.transition_ids),
                state.first_logits_event,
                pending_token_id=pending,
            )
            return self._make_step(
                NativeDecodeSessionStatus.CANCELLED,
                batch_start=len(state.generated),
                result=result,
            )
        finally:
            with self._lifecycle_lock:
                self._in_step = False
                cleanup_requested = self._cleanup_requested
            if cleanup_requested:
                self._cleanup_once()

    def close(self) -> NativeDecodeSessionStep | None:
        """Abort an active checkpoint and release decoder/callback references."""
        with self._lifecycle_lock:
            if self._final_step is not None:
                step = self._replay_terminal_step()
                if self._in_step:
                    self._cancel_requested = True
                    self._close_requested = True
            elif self._fatal_error is not None:
                step = None
                if self._in_step:
                    self._close_requested = True
            elif self._closed:
                return None
            elif self._in_step:
                self._cancel_requested = True
                self._close_requested = True
                return None
            else:
                step = None
                self._cancel_requested = True
                self._close_requested = True
        if self._final_step is None and self._fatal_error is None and not self._closed:
            step = self.cancel()
        self._request_safe_cleanup_once()
        return step

    async def advance(
        self,
        *,
        max_committed_tokens: int = 8,
        ready_when: Callable[[NativeDecodeProgress], Any],
        stop_check: Callable[[], Any] | None = None,
        producer_limit_check: Callable[[], Any] | None = None,
    ) -> NativeDecodeSessionStep:
        """Advance at most eight committed output tokens, then checkpoint.

        ``ready_when`` is the caller's stable-token/Qwen predicate. A true
        result returns READY immediately; exhausting the per-call token bound
        without stable caller-visible text returns MORE while retaining all KV,
        lookahead, sampler, and incremental-text state. Neither status calls
        ``finish`` or flushes pending Unicode bytes.
        """
        if isinstance(max_committed_tokens, bool) or not isinstance(max_committed_tokens, Integral):
            raise TypeError("max_committed_tokens must be an integer")
        limit = int(max_committed_tokens)
        if limit < 1 or limit > 8:
            raise ValueError("max_committed_tokens must be between 1 and 8")
        if not callable(ready_when):
            raise TypeError("ready_when must be a caller-supplied stable-text predicate")
        with self._lifecycle_lock:
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._final_step is not None:
                return self._replay_terminal_step()
            if self._closed:
                raise RuntimeError("native generation session is closed")
            if self._in_step:
                raise RuntimeError("native generation session advance is already in progress")
            decoder, state = self._live()
            self._in_step = True
        batch_start = len(state.generated)
        advanced = 0
        try:
            if state.budget == 0:
                return await self._terminate(
                    "budget_exhausted", batch_start=batch_start, no_output=True
                )

            if state.phase == "prefill":
                while state.prefill_offset < len(state.transition_ids):
                    if self._is_cancel_requested() or await decoder._should_stop(stop_check):
                        return await self._terminate("cancelled", batch_start=batch_start)
                    group = state.transition_ids[state.prefill_offset:state.prefill_offset + 8]
                    state.source_forward_started_ns = int(decoder.monotonic_ns())
                    state.logits, state.hidden = await decoder._forward(list(group))
                    state.source_logits_ready_ns = int(decoder.monotonic_ns())
                    state.native_forwarded_count += len(group)
                    state.prefill_offset += len(group)
                assert state.logits is not None and state.hidden is not None
                state.first_logits_event = FirstLogitsReady(
                    logits=state.logits,
                    final_hidden=state.hidden,
                    ready_ns=state.source_logits_ready_ns,
                    elapsed_ns=state.source_logits_ready_ns - state.decode_started_ns,
                )
                if state.first_logits_ready_callback is not None:
                    await _resolve(state.first_logits_ready_callback(state.first_logits_event))
                state.phase = "initial_sample"
                if self._is_cancel_requested() or await decoder._should_stop(stop_check):
                    return await self._terminate("cancelled", batch_start=batch_start)

            if state.phase == "initial_sample":
                assert state.logits is not None
                state.candidate = await decoder._sample(
                    state.logits,
                    tuple(state.prefix_ids),
                    initial=True,
                )
                state.candidate_sampled_ns = int(decoder.monotonic_ns())
                state.sampled_count += 1
                state.phase = "decode"

            while True:
                if self._is_cancel_requested() or await decoder._should_stop(stop_check):
                    pending = _sampled_token_id(state.candidate)
                    return await self._terminate(
                        "cancelled",
                        batch_start=batch_start,
                        pending_token_id=pending,
                    )
                if await decoder._should_stop(producer_limit_check):
                    return await self._terminate("producer_limit", batch_start=batch_start)

                assert state.logits is not None and state.hidden is not None
                generated_history = tuple(token.token_id for token in state.generated)
                history = (*state.prefix_ids, *generated_history)
                state.candidate = await decoder._select_candidate(
                    state.candidate,
                    state.logits,
                    history,
                )
                token_id = _sampled_token_id(state.candidate)
                if token_id in decoder.eos_token_ids and len(state.generated) < state.hard_budget:
                    min_new_tokens = int(getattr(decoder.sampler, "min_new_tokens", 0))
                    if len(state.generated) < min_new_tokens:
                        raise RuntimeError(
                            "MinNewTokensLengthLogitsProcessor ha restituito EOS prima della soglia "
                            f"({len(state.generated)} < {min_new_tokens})"
                        )
                    state.terminal_token_id = token_id
                    return await self._terminate("eos", batch_start=batch_start)

                if await decoder._should_stop(producer_limit_check):
                    return await self._terminate("producer_limit", batch_start=batch_start)
                if (
                    state.graceful_sentence_completion
                    and len(state.generated) >= state.budget
                    and ends_at_sentence_boundary(
                        decoder._text_decoder.text if decoder._text_decoder is not None else ""
                    )
                ):
                    return await self._terminate("sentence_boundary_after_budget", batch_start=batch_start)
                if len(state.generated) >= state.hard_budget:
                    return await self._terminate("budget_exhausted", batch_start=batch_start)

                if self._is_cancel_requested() or await decoder._should_stop(stop_check):
                    return await self._terminate(
                        "cancelled",
                        batch_start=batch_start,
                        pending_token_id=token_id,
                    )
                if await decoder._should_stop(producer_limit_check):
                    return await self._terminate("producer_limit", batch_start=batch_start)

                token_text = (
                    decoder._text_decoder.preview(token_id)
                    if decoder._text_decoder is not None else ""
                )
                anticipate_token = bool(
                    state.anticipate_first_token
                    and state.token_index == 0
                    and state.on_token_packet_callback is not None
                )
                transition_hidden = state.hidden
                if anticipate_token:
                    early_token = GeneratedToken(
                        token_id=token_id,
                        hidden=transition_hidden,
                        timing=TokenTiming(
                            forward_started_ns=state.source_forward_started_ns,
                            logits_ready_ns=state.source_logits_ready_ns,
                            sampled_ns=state.candidate_sampled_ns,
                            forward_duration_ns=state.source_logits_ready_ns - state.source_forward_started_ns,
                            generation_elapsed_ns=state.candidate_sampled_ns - state.decode_started_ns,
                            token_forward_started_ns=None,
                            token_hidden_ready_ns=None,
                            token_forward_duration_ns=None,
                            token_ready_elapsed_ns=None,
                        ),
                        token_index=state.token_index,
                        logical_source_sequence=state.logical_offset + state.token_index,
                        hidden_provenance="transition",
                        kv_committed=False,
                        token_text=token_text,
                        text_source_sequences=((state.logical_offset + state.token_index, token_id),),
                    )
                    state.generated.append(early_token)
                    await _resolve(state.on_token_packet_callback(early_token))
                    early_token = replace(
                        early_token,
                        packet_published_ns=int(decoder.monotonic_ns()),
                    )
                    state.generated[-1] = early_token
                    if state.first_token_callback is not None:
                        await _resolve(state.first_token_callback(early_token))
                    if self._is_cancel_requested() or await decoder._should_stop(stop_check):
                        return await self._terminate(
                            "cancelled",
                            batch_start=batch_start,
                            pending_token_id=token_id,
                        )

                token_forward_started_ns = int(decoder.monotonic_ns())
                state.logits, token_hidden = await decoder._forward([token_id])
                token_hidden_ready_ns = int(decoder.monotonic_ns())
                state.native_forwarded_count += 1
                state.committed_count += 1
                if decoder._text_decoder is not None:
                    committed_text = decoder._text_decoder.append(
                        token_id,
                        source_sequence=state.logical_offset + state.token_index,
                        source_token_id=token_id,
                    )
                    if committed_text != token_text:
                        raise RuntimeError("Thinker incremental preview changed during token commit")
                if anticipate_token and state.on_token_committed_callback is not None:
                    committed_packet = replace(
                        early_token,
                        timing=TokenTiming(
                            forward_started_ns=state.source_forward_started_ns,
                            logits_ready_ns=state.source_logits_ready_ns,
                            sampled_ns=state.candidate_sampled_ns,
                            forward_duration_ns=state.source_logits_ready_ns - state.source_forward_started_ns,
                            generation_elapsed_ns=state.candidate_sampled_ns - state.decode_started_ns,
                            token_forward_started_ns=token_forward_started_ns,
                            token_hidden_ready_ns=token_hidden_ready_ns,
                            token_forward_duration_ns=token_hidden_ready_ns - token_forward_started_ns,
                            token_ready_elapsed_ns=token_hidden_ready_ns - state.decode_started_ns,
                        ),
                        kv_committed=True,
                    )
                    await _resolve(state.on_token_committed_callback(committed_packet))

                next_generated_ids = tuple([*generated_history, token_id])
                sampler_generated_ids = tuple([*state.prefix_ids, *next_generated_ids])
                state.candidate = await decoder._after_generated_token(
                    token_index=state.token_index,
                    logical_source_sequence=state.logical_offset + state.token_index,
                    token_id=token_id,
                    token_hidden=token_hidden,
                    generated_ids=sampler_generated_ids,
                    source_logits=state.logits,
                    sample_next=lambda: decoder._sample(state.logits, sampler_generated_ids),
                )
                next_sampled_ns = int(decoder.monotonic_ns())
                state.sampled_count += 1
                token = GeneratedToken(
                    token_id=token_id,
                    hidden=transition_hidden if anticipate_token else token_hidden,
                    timing=TokenTiming(
                        forward_started_ns=state.source_forward_started_ns,
                        logits_ready_ns=state.source_logits_ready_ns,
                        sampled_ns=state.candidate_sampled_ns,
                        forward_duration_ns=state.source_logits_ready_ns - state.source_forward_started_ns,
                        generation_elapsed_ns=state.candidate_sampled_ns - state.decode_started_ns,
                        token_forward_started_ns=token_forward_started_ns,
                        token_hidden_ready_ns=token_hidden_ready_ns,
                        token_forward_duration_ns=token_hidden_ready_ns - token_forward_started_ns,
                        token_ready_elapsed_ns=token_hidden_ready_ns - state.decode_started_ns,
                    ),
                    token_index=state.token_index,
                    logical_source_sequence=state.logical_offset + state.token_index,
                    hidden_provenance="transition" if anticipate_token else "token",
                    kv_committed=True,
                    packet_published_ns=(
                        early_token.packet_published_ns if anticipate_token else None
                    ),
                    token_text=token_text,
                    text_source_sequences=(
                        decoder._text_decoder.last_text_source_group
                        if decoder._text_decoder is not None else ()
                    ),
                )
                if anticipate_token:
                    state.generated[-1] = token
                else:
                    state.generated.append(token)
                    if state.on_token_packet_callback is not None:
                        await _resolve(state.on_token_packet_callback(token))
                        token = replace(token, packet_published_ns=int(decoder.monotonic_ns()))
                        state.generated[-1] = token
                if state.on_token_callback is not None:
                    await _resolve(state.on_token_callback(token.token_id, token.hidden, token.timing))
                if state.token_index == 0 and state.first_token_callback is not None and not anticipate_token:
                    await _resolve(state.first_token_callback(token))

                state.candidate_sampled_ns = next_sampled_ns
                state.source_forward_started_ns = token_forward_started_ns
                state.source_logits_ready_ns = token_hidden_ready_ns
                state.hidden = token_hidden
                state.token_index += 1
                advanced += 1

                if self._is_cancel_requested() or await decoder._should_stop(stop_check):
                    return await self._terminate(
                        "cancelled",
                        batch_start=batch_start,
                        pending_token_id=_sampled_token_id(state.candidate),
                    )
                if await decoder._should_stop(producer_limit_check):
                    return await self._terminate("producer_limit", batch_start=batch_start)

                progress = self._make_progress(decoder, state)
                if await _resolve(ready_when(progress)):
                    return self._make_step(
                        NativeDecodeSessionStatus.READY,
                        batch_start=batch_start,
                    )
                if advanced >= limit:
                    return self._make_step(
                        NativeDecodeSessionStatus.MORE,
                        batch_start=batch_start,
                    )
        except BaseException as exc:
            if self._is_expected_cancel(exc):
                self._record_cancelled_exception(
                    decoder,
                    state,
                    batch_start=batch_start,
                )
            else:
                self._failed = True
                self._fatal_error = exc
            self._request_safe_cleanup_once()
            raise
        finally:
            with self._lifecycle_lock:
                self._in_step = False
                cleanup_requested = self._cleanup_requested
            if cleanup_requested:
                self._cleanup_once()

    async def resume(self, **kwargs: Any) -> NativeDecodeSessionStep:
        """Alias for ``advance`` intended for a previously READY/MORE session."""
        return await self.advance(**kwargs)



async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _token_ids(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("transition_token_ids deve contenere solo interi")
        token_id = int(value)
        if token_id < 0:
            raise ValueError("transition_token_ids non può contenere ID negativi")
        result.append(token_id)
    return result


def _resolve_thinker_eos_ids(thinker: Any) -> set[int]:
    model = getattr(thinker, "model", thinker)
    generation_config = getattr(model, "generation_config", None)
    generation_eos = getattr(generation_config, "eos_token_id", None)
    if generation_eos is not None:
        return normalize_eos_token_ids(generation_eos)

    model_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if model_eos is not None:
        return normalize_eos_token_ids(model_eos)

    tokenizer = getattr(thinker, "tokenizer", None)
    eos_ids = normalize_eos_token_ids(getattr(tokenizer, "eos_token_id", None))
    eos_ids.update(normalize_eos_token_ids(getattr(tokenizer, "eot_id", None)))
    return eos_ids


def _sampled_token_id(value: Any) -> int:
    # The Thinker/engine sampler returns ``(token_tensor, token_id, scores)``.
    # Accept that existing contract directly so callers do not need to wrap or
    # replace the configured sampler.
    if isinstance(value, (tuple, list)) and len(value) == 3:
        token_tensor, token_id, _scores = value
        normalized_id = _sampled_token_id(token_id)
        if isinstance(token_tensor, torch.Tensor):
            if token_tensor.numel() != 1 or int(token_tensor.item()) != normalized_id:
                raise ValueError("sampler token tensor e token_id non coincidono")
        return normalized_id
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("sample_token deve restituire un solo token ID")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("sample_token deve restituire un intero o un Tensor scalare")
    token_id = int(value)
    if token_id < 0:
        raise ValueError("sample_token ha restituito un ID negativo")
    return token_id


def _sampled_candidate(value: Any) -> Any:
    """Validate ThinkerSampler's candidate tuple while retaining its scores."""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        token_tensor, token_id, scores = value
        normalized_id = _sampled_token_id(token_id)
        if isinstance(token_tensor, torch.Tensor):
            if token_tensor.numel() != 1 or int(token_tensor.item()) != normalized_id:
                raise ValueError("sampler token tensor e token_id non coincidono")
        return token_tensor, normalized_id, scores
    return _sampled_token_id(value)


class UnifiedNativeDecoder:
    """Decode against native KV using caller-owned token prep and sampling."""

    def __init__(
        self,
        thinker: Any,
        *,
        forward_tokens: Callable[[list[int]], Any],
        sample_token: Callable[[torch.Tensor, tuple[int, ...]], Any] | None = None,
        sampler: Any | None = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        eos_token_ids: Sequence[int] | set[int] | None = None,
    ) -> None:
        self.thinker = thinker
        self.forward_tokens = forward_tokens
        self.sampler = sampler
        pipeline_sample = getattr(sampler, "sample_token", None)
        self.sample_token = sample_token or pipeline_sample
        if not callable(self.sample_token):
            raise TypeError("serve sample_token o un ThinkerSampler con sample_token")
        self.monotonic_ns = monotonic_ns
        self.eos_token_ids = (
            _resolve_thinker_eos_ids(thinker)
            if eos_token_ids is None
            else {int(token_id) for token_id in eos_token_ids}
        )
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("eos_token_ids non può contenere ID negativi")
        tokenizer = getattr(sampler, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(thinker, "tokenizer", None)
        sampler_decoder = getattr(sampler, "_text_decoder", None)
        if isinstance(sampler_decoder, IncrementalTextDecoder):
            self._text_decoder = sampler_decoder.clone()
        elif callable(getattr(tokenizer, "decode", None)) or callable(
            getattr(getattr(tokenizer, "backend_tokenizer", None), "decode", None)
        ):
            self._text_decoder = IncrementalTextDecoder(tokenizer)
        else:
            self._text_decoder = None
        self._active_session_ref: weakref.ReferenceType[NativeGenerationSession] | None = None
        self._session_started = False

    def _assert_no_active_session(self) -> None:
        active_ref = self._active_session_ref
        active = active_ref() if callable(active_ref) else None
        if active is not None and not active._closed and active._state is not None:
            raise RuntimeError("UnifiedNativeDecoder already has an active resumable session")

    @staticmethod
    def _fork_generator(source_generator: Any, fork_id: str) -> torch.Generator | None:
        if source_generator is None:
            return None
        get_state = getattr(source_generator, "get_state", None)
        device = getattr(source_generator, "device", None)
        if not callable(get_state) or device is None:
            raise TypeError("sampler.generator must expose get_state() and device")
        source_state = get_state()
        if not isinstance(source_state, torch.Tensor):
            raise TypeError("sampler.generator.get_state() must return a Tensor")
        # Derive a private deterministic stream from the source RNG checkpoint
        # and branch identity. Reading the state does not advance the canonical
        # generator or touch PyTorch's process-global RNG.
        state_bytes = bytes(source_state.detach().cpu().reshape(-1).tolist())
        digest = hashlib.sha256(state_bytes + fork_id.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        fork_generator = torch.Generator(device=device)
        fork_generator.manual_seed(seed)
        return fork_generator

    @staticmethod
    def _copy_sampler_local_state(source: Any, destination: Any) -> None:
        """Copy known decoder history while leaving callbacks/state owner-local."""
        if source is None:
            return
        callback_fields = (
            "on_generation_start",
            "on_generated_token_callback",
            "on_candidate_token",
            "on_candidate_fallback",
            "on_action_logits_probe",
            "best_native_logit_candidate_id",
        )
        for name in callback_fields:
            source_callback = getattr(source, name, None)
            if source_callback is not None and getattr(destination, name, None) is source_callback:
                raise ValueError(f"fork sampler must own a separate {name} callback")
        source_state = getattr(source, "state", None)
        if source_state is not None and getattr(destination, "state", None) is source_state:
            raise ValueError("fork sampler must use branch-owned mutable state")

        source_metrics = getattr(source, "runtime_metrics", None)
        destination_metrics = getattr(destination, "runtime_metrics", None)
        if isinstance(source_metrics, dict):
            if destination_metrics is source_metrics:
                raise ValueError("fork sampler must use a private runtime_metrics mapping")
            if isinstance(destination_metrics, dict):
                destination_metrics.clear()
                destination_metrics.update(copy.deepcopy(source_metrics))
            else:
                destination.runtime_metrics = copy.deepcopy(source_metrics)

        for name in (
            "last_decoded_output",
            "forced_action_overrides",
            "forced_structure_overrides",
            "action_logits_probe_emitted",
            "forced_action_logprobs",
            "forced_structure_logprobs",
            "backchannel_trie",
            "min_new_tokens_processor",
        ):
            if not hasattr(source, name) or not hasattr(destination, name):
                continue
            value = getattr(source, name)
            if isinstance(value, list) and value and all(
                isinstance(item, torch.Tensor) for item in value
            ):
                cloned = [item.detach().clone() for item in value]
            else:
                cloned = copy.deepcopy(value)
            setattr(destination, name, cloned)
        source_decoder = getattr(source, "_text_decoder", None)
        if isinstance(source_decoder, IncrementalTextDecoder):
            destination._text_decoder = source_decoder.clone()
            destination._base_token_ids = tuple(source_decoder.token_ids)
            destination.last_decoded_output = source_decoder.text

    def fork(
        self,
        *,
        forward_tokens: Callable[[list[int]], Any],
        sampler_factory: Callable[[Any, torch.Generator | None, str], Any],
        fork_id: str,
    ) -> UnifiedNativeDecoder:
        """Create an isolated decoder branch over caller-owned forked native KV.

        ``sampler_factory`` must create branch-local callbacks/state and accept
        the private RNG generated here. The decoder shares only the read-only
        Thinker reference and EOS token set; native forward ownership and every
        sampler-mutated object belong to the branch. Decode callbacks are
        call-scoped and must be supplied by the branch caller.
        """
        self._assert_no_active_session()
        if not callable(forward_tokens):
            raise TypeError("fork forward_tokens must be callable and bound to the forked KV slot")
        if not callable(sampler_factory):
            raise TypeError("fork sampler_factory must be callable")
        if not isinstance(fork_id, str) or not fork_id.strip():
            raise ValueError("fork_id must be non-empty")
        source_generator = getattr(self.sampler, "generator", None)
        if self.sampler is not None and source_generator is None:
            temperature = getattr(self.sampler, "temperature", None)
            if temperature is None or float(temperature) > 1e-5:
                raise RuntimeError("stochastic decoder fork requires a private generator")
        fork_generator = self._fork_generator(source_generator, fork_id)
        fork_sampler = sampler_factory(self.sampler, fork_generator, fork_id)
        if self.sampler is not None and fork_sampler is self.sampler:
            raise ValueError("fork sampler_factory returned the source sampler")
        if fork_sampler is None:
            raise TypeError("fork sampler_factory must return an independent sampler")
        sampler_generator = getattr(fork_sampler, "generator", None)
        if fork_generator is not None and sampler_generator is not fork_generator:
            raise ValueError("fork sampler must use the private generator supplied by the decoder")
        if sampler_generator is source_generator and source_generator is not None:
            raise ValueError("fork sampler cannot share the source generator")
        if not callable(getattr(fork_sampler, "sample_token", None)):
            raise TypeError("fork sampler must expose sample_token")
        self._copy_sampler_local_state(self.sampler, fork_sampler)
        return UnifiedNativeDecoder(
            self.thinker,
            forward_tokens=forward_tokens,
            sampler=fork_sampler,
            monotonic_ns=self.monotonic_ns,
            eos_token_ids=set(self.eos_token_ids),
        )

    async def _sample(self, logits: torch.Tensor, history: tuple[int, ...], *, initial: bool = False) -> Any:
        sample = None
        if initial and self.sampler is not None:
            sample = getattr(self.sampler, "sample_initial", None)
        candidate = await _resolve((sample or self.sample_token)(logits, history))
        return _sampled_candidate(candidate)

    async def _select_candidate(self, candidate: Any, logits: torch.Tensor, history: tuple[int, ...]) -> Any:
        select = getattr(self.sampler, "select_valid_candidate", None)
        if not callable(select) or not isinstance(candidate, (tuple, list)) or len(candidate) != 3:
            return candidate
        selected = await _resolve(select(candidate, source_logits=logits, generated_ids=history))
        return _sampled_candidate(selected)

    async def _after_generated_token(
        self,
        *,
        token_index: int,
        logical_source_sequence: int,
        token_id: int,
        token_hidden: torch.Tensor,
        generated_ids: tuple[int, ...],
        source_logits: torch.Tensor,
        sampled_next: Any = None,
        sample_next: Callable[[], Any] | None = None,
    ) -> Any:
        async_callback = getattr(self.sampler, "after_generated_token_async", None)
        if callable(async_callback):
            result = await _resolve(async_callback(
                token_index=token_index,
                logical_source_sequence=logical_source_sequence,
                token_id=token_id,
                token_hidden=token_hidden,
                generated_ids=generated_ids,
                source_logits=source_logits,
                sampled_next=sampled_next,
                sample_next=sample_next,
            ))
            return _sampled_candidate(result)

        callback = getattr(self.sampler, "after_generated_token", None)
        if not callable(callback):
            return await _resolve(sample_next()) if sample_next is not None else sampled_next
        if sample_next is not None:
            sampled_next = await _resolve(sample_next())
        if not isinstance(sampled_next, (tuple, list)) or len(sampled_next) != 3:
            return sampled_next
        result = await _resolve(callback(
            token_index=token_index,
            logical_source_sequence=logical_source_sequence,
            token_id=token_id,
            token_hidden=token_hidden,
            generated_ids=generated_ids,
            source_logits=source_logits,
            sampled_next=sampled_next,
        ))
        return _sampled_candidate(result)

    async def _should_stop(self, stop_check: Callable[[], Any] | None) -> bool:
        return bool(await _resolve(stop_check())) if stop_check is not None else False

    async def _forward(self, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not token_ids or len(token_ids) > 8:
            raise ValueError("forward_tokens accetta gruppi da uno a otto token")
        result = await _resolve(self.forward_tokens(list(token_ids)))
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("forward_tokens deve restituire (logits, final_hidden)")
        logits, final_hidden = result
        if not isinstance(logits, torch.Tensor) or not isinstance(final_hidden, torch.Tensor):
            raise TypeError("forward_tokens deve restituire Tensor per logits e final_hidden")
        return logits, final_hidden

    def start_session(
        self,
        *,
        transition_token_ids: Sequence[int],
        max_new_tokens: int,
        first_logits_ready: Callable[[FirstLogitsReady], Any] | None = None,
        first_token: Callable[[GeneratedToken], Any] | None = None,
        on_token: Callable[[int, torch.Tensor, TokenTiming], Any] | None = None,
        on_token_packet: Callable[[GeneratedToken], Any] | None = None,
        on_token_committed: Callable[[GeneratedToken], Any] | None = None,
        on_text_flush: Callable[[str, GeneratedToken], Any] | None = None,
        anticipate_first_token: bool = False,
        graceful_sentence_completion: bool = False,
        sentence_completion_max_tokens: int = 0,
        continuation_prefix_token_ids: Sequence[int] = (),
        logical_source_offset: int = 0,
        native_handle: Any | None = None,
        native_cursor: int | None = None,
    ) -> NativeGenerationSession:
        """Start an explicit stateful decoder that can span separate event loops.

        The caller owns the native handle and supplies its current token cursor.
        The cursor reported by each step is that base cursor plus the transition
        and generated tokens successfully forwarded by this decoder. A decoder
        instance supports only one resumable session because its KV callback and
        sampler belong to that one branch.

        Stop and producer checks are passed to each ``advance`` call so a
        loop-bound event/future is never retained.
        """
        if self._session_started:
            raise RuntimeError("UnifiedNativeDecoder already owns a resumable session")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
            raise TypeError("max_new_tokens deve essere un intero")
        budget = int(max_new_tokens)
        if budget < 0:
            raise ValueError("max_new_tokens non può essere negativo")
        if isinstance(sentence_completion_max_tokens, bool) or not isinstance(
            sentence_completion_max_tokens, Integral
        ):
            raise TypeError("sentence_completion_max_tokens deve essere un intero")
        sentence_extension = int(sentence_completion_max_tokens)
        if sentence_extension < 0 or sentence_extension > 32:
            raise ValueError("sentence_completion_max_tokens deve essere tra 0 e 32")
        if sentence_extension and not graceful_sentence_completion:
            raise ValueError("sentence completion extension requires graceful_sentence_completion")
        transition_ids = _token_ids(transition_token_ids)
        prefix_ids = _token_ids(continuation_prefix_token_ids)
        if prefix_ids and (
            len(transition_ids) < len(prefix_ids)
            or transition_ids[-len(prefix_ids):] != prefix_ids
        ):
            raise ValueError("continuation prefix IDs must be the replayed transition suffix")
        if budget > 0 and not transition_ids:
            raise ValueError("serve almeno un transition token per ottenere i primi logits")
        if isinstance(logical_source_offset, bool) or not isinstance(
            logical_source_offset, Integral
        ):
            raise TypeError("logical_source_offset deve essere un intero")
        logical_offset = int(logical_source_offset)
        if logical_offset < 0:
            raise ValueError("logical_source_offset non può essere negativo")
        if native_cursor is not None:
            if isinstance(native_cursor, bool) or not isinstance(native_cursor, Integral):
                raise TypeError("native_cursor must be an integer or None")
            native_cursor = int(native_cursor)
            if native_cursor < 0:
                raise ValueError("native_cursor cannot be negative")

        if prefix_ids and self._text_decoder is not None:
            self._text_decoder.ensure_prefix(prefix_ids)
        text_source_token_offset = (
            len(self._text_decoder.token_ids)
            if isinstance(self._text_decoder, IncrementalTextDecoder)
            else 0
        )
        decode_started_ns = int(self.monotonic_ns())
        state = _NativeDecodeSessionState(
            transition_ids=tuple(transition_ids),
            prefix_ids=tuple(prefix_ids),
            budget=budget,
            hard_budget=budget + sentence_extension,
            logical_offset=logical_offset,
            anticipate_first_token=bool(anticipate_first_token),
            graceful_sentence_completion=bool(graceful_sentence_completion),
            first_logits_ready_callback=first_logits_ready,
            first_token_callback=first_token,
            on_token_callback=on_token,
            on_token_packet_callback=on_token_packet,
            on_token_committed_callback=on_token_committed,
            on_text_flush_callback=on_text_flush,
            decode_started_ns=decode_started_ns,
            source_forward_started_ns=decode_started_ns,
            source_logits_ready_ns=decode_started_ns,
            text_source_token_offset=text_source_token_offset,
        )
        session = NativeGenerationSession(
            self,
            state,
            native_handle=native_handle,
            native_cursor=native_cursor,
        )
        self._active_session_ref = weakref.ref(session)
        self._session_started = True
        return session

    async def decode_continuation(
        self,
        *,
        frozen_prefix_token_ids: Sequence[int],
        context_transition_token_ids: Sequence[int] = (),
        logical_source_offset: int,
        generated_budget_used: int,
        total_generated_budget: int = DEFAULT_CONTINUATION_GENERATED_BUDGET,
        max_new_tokens: int | None = None,
        **decode_kwargs: Any,
    ) -> NativeDecodeResult:
        """Replay a frozen assistant prefix and decode only its unpublished suffix.

        The required owner transition is forwarded first, followed by the
        frozen generated prefix to reconstruct the native KV branch. Only the
        frozen generated prefix enters sampler/text history, and neither part
        produces token, packet, or publication callbacks. ``logical_source_offset`` is
        supplied by the response ledger; it need not equal the prefix length
        because discarded revisions can leave gaps in logical source IDs.
        ``generated_budget_used`` counts adopted/frozen output only. Replayed
        prefix tokens and discarded speculative candidates do not consume the
        response-wide 768-token allowance.
        """
        if isinstance(logical_source_offset, bool) or not isinstance(
            logical_source_offset, Integral
        ):
            raise TypeError("logical_source_offset deve essere un intero")
        logical_offset = int(logical_source_offset)
        if logical_offset < 0:
            raise ValueError("logical_source_offset non può essere negativo")
        for name, value in (
            ("generated_budget_used", generated_budget_used),
            ("total_generated_budget", total_generated_budget),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} deve essere un intero")
            if int(value) < 0:
                raise ValueError(f"{name} non può essere negativo")
        used = int(generated_budget_used)
        total = int(total_generated_budget)
        if used > total:
            raise ValueError("generated_budget_used supera il budget totale della risposta")
        remaining = total - used
        requested = remaining if max_new_tokens is None else max_new_tokens
        if isinstance(requested, bool) or not isinstance(requested, Integral):
            raise TypeError("max_new_tokens deve essere un intero")
        if int(requested) < 0:
            raise ValueError("max_new_tokens non può essere negativo")
        budget = min(int(requested), remaining)
        prefix_ids = _token_ids(frozen_prefix_token_ids)
        context_ids = _token_ids(context_transition_token_ids)
        replay_ids = [*context_ids, *prefix_ids] if budget > 0 else []

        extension = decode_kwargs.get("sentence_completion_max_tokens", 0)
        if isinstance(extension, bool) or not isinstance(extension, Integral):
            raise TypeError("sentence_completion_max_tokens deve essere un intero")
        decode_kwargs["sentence_completion_max_tokens"] = min(
            int(extension), max(0, remaining - budget)
        )
        return await self.decode(
            transition_token_ids=replay_ids,
            continuation_prefix_token_ids=prefix_ids if budget > 0 else (),
            logical_source_offset=logical_offset,
            max_new_tokens=budget,
            **decode_kwargs,
        )

    async def decode(
        self,
        *,
        transition_token_ids: Sequence[int],
        max_new_tokens: int,
        stop_check: Callable[[], Any] | None = None,
        first_logits_ready: Callable[[FirstLogitsReady], Any] | None = None,
        first_token: Callable[[GeneratedToken], Any] | None = None,
        on_token: Callable[[int, torch.Tensor, TokenTiming], Any] | None = None,
        on_token_packet: Callable[[GeneratedToken], Any] | None = None,
        on_token_committed: Callable[[GeneratedToken], Any] | None = None,
        on_text_flush: Callable[[str, GeneratedToken], Any] | None = None,
        anticipate_first_token: bool = False,
        graceful_sentence_completion: bool = False,
        sentence_completion_max_tokens: int = 0,
        producer_limit_check: Callable[[], Any] | None = None,
        continuation_prefix_token_ids: Sequence[int] = (),
        logical_source_offset: int = 0,
    ) -> NativeDecodeResult:
        """Append only transition IDs and sample from their final returned logits.

        The transition IDs must already be obtained from the active Thinker's
        tokenizer and verified as the suffix required by the owner. The owner
        callback handles embedding/PLE preparation, serial cache commit,
        cancellation safety, and assistant-branch rollback.

        ``anticipate_first_token`` lets the packet callback publish token0
        using the transition hidden before that token is appended to native KV.
        The packet is marked uncommitted until its own forward succeeds, then
        ``on_token_committed`` receives the same token index. Its token-forward
        timing fields are ``None`` during the first callback; the returned
        result carries completed timings. Later tokens use their own
        post-forward hidden states and are committed before publication.
        """
        self._assert_no_active_session()
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
            raise TypeError("max_new_tokens deve essere un intero")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens non può essere negativo")
        budget = int(max_new_tokens)
        if isinstance(sentence_completion_max_tokens, bool) or not isinstance(
            sentence_completion_max_tokens, Integral
        ):
            raise TypeError("sentence_completion_max_tokens deve essere un intero")
        sentence_extension = int(sentence_completion_max_tokens)
        if sentence_extension < 0 or sentence_extension > 32:
            raise ValueError("sentence_completion_max_tokens deve essere tra 0 e 32")
        if sentence_extension and not graceful_sentence_completion:
            raise ValueError("sentence completion extension requires graceful_sentence_completion")
        hard_budget = budget + sentence_extension
        transition_ids = _token_ids(transition_token_ids)
        prefix_ids = _token_ids(continuation_prefix_token_ids)
        if prefix_ids and (
            len(transition_ids) < len(prefix_ids)
            or transition_ids[-len(prefix_ids):] != prefix_ids
        ):
            raise ValueError("continuation prefix IDs must be the replayed transition suffix")
        if isinstance(logical_source_offset, bool) or not isinstance(
            logical_source_offset, Integral
        ):
            raise TypeError("logical_source_offset deve essere un intero")
        logical_offset = int(logical_source_offset)
        if logical_offset < 0:
            raise ValueError("logical_source_offset non può essere negativo")
        if budget > 0 and not transition_ids:
            raise ValueError("serve almeno un transition token per ottenere i primi logits")

        if prefix_ids and self._text_decoder is not None:
            self._text_decoder.ensure_prefix(prefix_ids)

        decode_started_ns = int(self.monotonic_ns())
        generated: list[GeneratedToken] = []
        sampled_count = 0
        committed_count = 0
        first_logits_event: FirstLogitsReady | None = None
        terminal_token_id: int | None = None
        pending_token_id: int | None = None
        termination_reason = "budget_exhausted"
        def token_text_piece(generated_ids: tuple[int, ...]) -> str:
            if self._text_decoder is None:
                return ""
            if not generated_ids:
                return ""
            return self._text_decoder.preview(int(generated_ids[-1]))

        if budget == 0:
            return NativeDecodeResult(
                generated_tokens=(),
                termination_reason=termination_reason,
                terminal_token_id=None,
                generated_count=0,
                committed_count=0,
                sampled_count=0,
                transition_token_count=len(transition_ids),
                first_logits_ready=None,
                first_token=None,
                pending_token_id=None,
                owner_action="no_output",
            )

        logits: torch.Tensor | None = None
        hidden: torch.Tensor | None = None
        source_forward_started_ns = decode_started_ns
        source_logits_ready_ns = decode_started_ns
        for start in range(0, len(transition_ids), 8):
            if await self._should_stop(stop_check):
                termination_reason = "cancelled"
                return self._result(
                    generated,
                    termination_reason,
                    terminal_token_id,
                    committed_count,
                    sampled_count,
                    len(transition_ids),
                    first_logits_event,
                )
            group = transition_ids[start : start + 8]
            source_forward_started_ns = int(self.monotonic_ns())
            logits, hidden = await self._forward(group)
            source_logits_ready_ns = int(self.monotonic_ns())

        assert logits is not None and hidden is not None
        first_logits_event = FirstLogitsReady(
            logits=logits,
            final_hidden=hidden,
            ready_ns=source_logits_ready_ns,
            elapsed_ns=source_logits_ready_ns - decode_started_ns,
        )
        if first_logits_ready is not None:
            await _resolve(first_logits_ready(first_logits_event))
        if await self._should_stop(stop_check):
            return self._result(
                generated,
                "cancelled",
                terminal_token_id,
                committed_count,
                sampled_count,
                len(transition_ids),
                first_logits_event,
            )

        candidate = await self._sample(logits, tuple(prefix_ids), initial=True)
        candidate_sampled_ns = int(self.monotonic_ns())
        sampled_count += 1
        token_index = 0
        while True:
            if await self._should_stop(stop_check):
                pending_token_id = _sampled_token_id(candidate)
                termination_reason = "cancelled"
                break
            if await self._should_stop(producer_limit_check):
                termination_reason = "producer_limit"
                break
            generated_history = tuple(token.token_id for token in generated)
            history = (*prefix_ids, *generated_history)
            candidate = await self._select_candidate(candidate, logits, history)
            token_id = _sampled_token_id(candidate)
            # The candidate is a lookahead sampled after the preceding token
            # was already committed. Once the hard generation allowance is
            # spent, that lookahead is outside the response budget and cannot
            # turn a length stop into natural EOS.
            if token_id in self.eos_token_ids and len(generated) < hard_budget:
                min_new_tokens = int(getattr(self.sampler, "min_new_tokens", 0))
                if len(generated) < min_new_tokens:
                    raise RuntimeError(
                        "MinNewTokensLengthLogitsProcessor ha restituito EOS prima della soglia "
                        f"({len(generated)} < {min_new_tokens})"
                    )
                terminal_token_id = token_id
                termination_reason = "eos"
                break

            if await self._should_stop(producer_limit_check):
                termination_reason = "producer_limit"
                break

            # A model-predicted EOS takes precedence over an imposed sentence
            # boundary. Once the requested budget is spent, stop at punctuation
            # or continue only within the explicitly reserved extension.
            if (
                graceful_sentence_completion
                and len(generated) >= budget
                and ends_at_sentence_boundary(
                    self._text_decoder.text if self._text_decoder is not None else ""
                )
            ):
                termination_reason = "sentence_boundary_after_budget"
                break
            if len(generated) >= hard_budget:
                termination_reason = "budget_exhausted"
                break

            if await self._should_stop(stop_check):
                pending_token_id = token_id
                termination_reason = "cancelled"
                break
            if await self._should_stop(producer_limit_check):
                termination_reason = "producer_limit"
                break

            token_text = token_text_piece(tuple([*history, token_id]))
            anticipate_token = bool(
                anticipate_first_token and token_index == 0 and on_token_packet is not None
            )
            transition_hidden = hidden
            if anticipate_token:
                early_token = GeneratedToken(
                    token_id=token_id,
                    hidden=transition_hidden,
                    timing=TokenTiming(
                        forward_started_ns=source_forward_started_ns,
                        logits_ready_ns=source_logits_ready_ns,
                        sampled_ns=candidate_sampled_ns,
                        forward_duration_ns=source_logits_ready_ns - source_forward_started_ns,
                        generation_elapsed_ns=candidate_sampled_ns - decode_started_ns,
                        token_forward_started_ns=None,
                        token_hidden_ready_ns=None,
                        token_forward_duration_ns=None,
                        token_ready_elapsed_ns=None,
                    ),
                    token_index=token_index,
                    logical_source_sequence=logical_offset + token_index,
                    hidden_provenance="transition",
                    kv_committed=False,
                    token_text=token_text,
                    text_source_sequences=((logical_offset + token_index, token_id),),
                )
                generated.append(early_token)
                await _resolve(on_token_packet(early_token))
                early_token = replace(
                    early_token,
                    packet_published_ns=int(self.monotonic_ns()),
                )
                generated[-1] = early_token
                if first_token is not None:
                    await _resolve(first_token(early_token))
                if await self._should_stop(stop_check):
                    pending_token_id = token_id
                    termination_reason = "cancelled"
                    break

            token_forward_started_ns = int(self.monotonic_ns())
            logits, token_hidden = await self._forward([token_id])
            token_hidden_ready_ns = int(self.monotonic_ns())
            committed_count += 1
            if self._text_decoder is not None:
                committed_text = self._text_decoder.append(
                    token_id,
                    source_sequence=logical_offset + token_index,
                    source_token_id=token_id,
                )
                if committed_text != token_text:
                    raise RuntimeError("Thinker incremental preview changed during token commit")
            if anticipate_token and on_token_committed is not None:
                committed_packet = replace(
                    early_token,
                    timing=TokenTiming(
                        forward_started_ns=source_forward_started_ns,
                        logits_ready_ns=source_logits_ready_ns,
                        sampled_ns=candidate_sampled_ns,
                        forward_duration_ns=source_logits_ready_ns - source_forward_started_ns,
                        generation_elapsed_ns=candidate_sampled_ns - decode_started_ns,
                        token_forward_started_ns=token_forward_started_ns,
                        token_hidden_ready_ns=token_hidden_ready_ns,
                        token_forward_duration_ns=token_hidden_ready_ns - token_forward_started_ns,
                        token_ready_elapsed_ns=token_hidden_ready_ns - decode_started_ns,
                    ),
                    kv_committed=True,
                )
                await _resolve(on_token_committed(committed_packet))
            next_generated_ids = tuple([*generated_history, token_id])
            sampler_generated_ids = tuple([*prefix_ids, *next_generated_ids])
            next_candidate = await self._after_generated_token(
                token_index=token_index,
                logical_source_sequence=logical_offset + token_index,
                token_id=token_id,
                token_hidden=token_hidden,
                generated_ids=sampler_generated_ids,
                source_logits=logits,
                sample_next=lambda: self._sample(logits, sampler_generated_ids),
            )
            next_sampled_ns = int(self.monotonic_ns())
            sampled_count += 1
            token = GeneratedToken(
                token_id=token_id,
                hidden=transition_hidden if anticipate_token else token_hidden,
                timing=TokenTiming(
                    forward_started_ns=source_forward_started_ns,
                    logits_ready_ns=source_logits_ready_ns,
                    sampled_ns=candidate_sampled_ns,
                    forward_duration_ns=source_logits_ready_ns - source_forward_started_ns,
                    generation_elapsed_ns=candidate_sampled_ns - decode_started_ns,
                    token_forward_started_ns=token_forward_started_ns,
                    token_hidden_ready_ns=token_hidden_ready_ns,
                    token_forward_duration_ns=token_hidden_ready_ns - token_forward_started_ns,
                    token_ready_elapsed_ns=token_hidden_ready_ns - decode_started_ns,
                ),
                token_index=token_index,
                logical_source_sequence=logical_offset + token_index,
                hidden_provenance="transition" if anticipate_token else "token",
                kv_committed=True,
                packet_published_ns=(
                    early_token.packet_published_ns if anticipate_token else None
                ),
                token_text=token_text,
                text_source_sequences=(
                    self._text_decoder.last_text_source_group
                    if self._text_decoder is not None else ()
                ),
            )
            if anticipate_token:
                generated[-1] = token
            else:
                generated.append(token)
                if on_token_packet is not None:
                    await _resolve(on_token_packet(token))
                    token = replace(token, packet_published_ns=int(self.monotonic_ns()))
                    generated[-1] = token
            if on_token is not None:
                await _resolve(on_token(token.token_id, token.hidden, token.timing))
            if token_index == 0 and first_token is not None and not anticipate_token:
                await _resolve(first_token(token))

            candidate = next_candidate
            candidate_sampled_ns = next_sampled_ns
            source_forward_started_ns = token_forward_started_ns
            source_logits_ready_ns = token_hidden_ready_ns
            token_index += 1

        if termination_reason in {"eos", "budget_exhausted", "sentence_boundary_after_budget", "producer_limit"} and generated:
            final_ids = tuple([*prefix_ids, *(token.token_id for token in generated)])
            suffix = ""
            if self._text_decoder is not None:
                suffix = self._text_decoder.finish()
            sampler_flush = getattr(self.sampler, "flush_decoded_text", None)
            if callable(sampler_flush):
                flush_kwargs: dict[str, Any] = {
                    "final_hidden": generated[-1].hidden,
                }
                try:
                    if "notify_callback" in inspect.signature(sampler_flush).parameters:
                        flush_kwargs["notify_callback"] = False
                except (TypeError, ValueError):
                    pass
                sampler_suffix = str(sampler_flush(final_ids, **flush_kwargs) or "")
                if suffix and sampler_suffix != suffix:
                    raise RuntimeError("Thinker samplers disagree on the final text flush")
                if not suffix:
                    suffix = sampler_suffix
            if suffix:
                generated[-1] = replace(
                    generated[-1],
                    token_text=generated[-1].token_text + suffix,
                    text_source_sequences=(
                        self._text_decoder.last_text_source_group
                        if self._text_decoder is not None
                        else generated[-1].text_source_sequences
                    ),
                )
                if on_text_flush is not None:
                    await _resolve(on_text_flush(suffix, generated[-1]))

        return self._result(
            generated,
            termination_reason,
            terminal_token_id,
            committed_count,
            sampled_count,
            len(transition_ids),
            first_logits_event,
            pending_token_id=pending_token_id,
        )

    @staticmethod
    def _result(
        generated: list[GeneratedToken],
        termination_reason: str,
        terminal_token_id: int | None,
        committed_count: int,
        sampled_count: int,
        transition_token_count: int,
        first_logits_event: FirstLogitsReady | None,
        pending_token_id: int | None = None,
    ) -> NativeDecodeResult:
        if termination_reason == "cancelled":
            owner_action = "rollback_assistant_branch"
        elif termination_reason == "eos":
            owner_action = "handle_terminal_eos"
        elif termination_reason in {
            "budget_exhausted", "sentence_boundary_after_budget", "producer_limit"
        } and pending_token_id is not None:
            owner_action = "commit_pending_then_append_turn_eos"
        elif termination_reason in {
            "budget_exhausted", "sentence_boundary_after_budget", "producer_limit"
        }:
            owner_action = "append_turn_eos"
        else:
            owner_action = "none"
        return NativeDecodeResult(
            generated_tokens=tuple(generated),
            termination_reason=termination_reason,
            terminal_token_id=terminal_token_id,
            generated_count=len(generated),
            committed_count=committed_count,
            sampled_count=sampled_count,
            transition_token_count=transition_token_count,
            first_logits_ready=first_logits_event,
            first_token=generated[0] if generated else None,
            pending_token_id=pending_token_id,
            owner_action=owner_action,
        )


__all__ = [
    "FirstLogitsReady",
    "GeneratedToken",
    "NativeDecodeProgress",
    "NativeDecodeResult",
    "NativeDecodeSessionStatus",
    "NativeDecodeSessionStep",
    "NativeGenerationSession",
    "TokenTiming",
    "UnifiedNativeDecoder",
]
