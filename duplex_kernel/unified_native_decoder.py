"""Small async decode loop over the native listener's single KV cache.

The caller owns cache state and supplies only the transition token IDs that
must be appended to the already-prefilled system/user/audio context. Each
forward callback must atomically commit those IDs through the native path and
return logits plus the final-normalized hidden state for its last logical
token. This module never builds an HF cache or forwards audio.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Sequence

import torch

from duplex_kernel.models.thinker import normalize_eos_token_ids


@dataclass(frozen=True)
class TokenTiming:
    """Sampling and own-token-forward timestamps for one published token."""

    forward_started_ns: int
    logits_ready_ns: int
    sampled_ns: int
    forward_duration_ns: int
    generation_elapsed_ns: int
    token_forward_started_ns: int
    token_hidden_ready_ns: int
    token_forward_duration_ns: int
    token_ready_elapsed_ns: int


@dataclass(frozen=True)
class FirstLogitsReady:
    logits: torch.Tensor
    final_hidden: torch.Tensor
    ready_ns: int
    elapsed_ns: int


@dataclass(frozen=True)
class GeneratedToken:
    """A sampled non-EOS token paired with the hidden state that selected it."""

    token_id: int
    hidden: torch.Tensor
    timing: TokenTiming


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
        token_id: int,
        token_hidden: torch.Tensor,
        generated_ids: tuple[int, ...],
        source_logits: torch.Tensor,
        sampled_next: Any,
    ) -> Any:
        callback = getattr(self.sampler, "after_generated_token_async", None)
        if not callable(callback):
            callback = getattr(self.sampler, "after_generated_token", None)
        if not callable(callback) or not isinstance(sampled_next, (tuple, list)) or len(sampled_next) != 3:
            return sampled_next
        result = await _resolve(callback(
            token_index=token_index,
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

    async def decode(
        self,
        *,
        transition_token_ids: Sequence[int],
        max_new_tokens: int,
        stop_check: Callable[[], Any] | None = None,
        first_logits_ready: Callable[[FirstLogitsReady], Any] | None = None,
        first_token: Callable[[GeneratedToken], Any] | None = None,
        on_token: Callable[[int, torch.Tensor, TokenTiming], Any] | None = None,
    ) -> NativeDecodeResult:
        """Append only transition IDs and sample from their final returned logits.

        The transition IDs must already be obtained from the active Thinker's
        tokenizer and verified as the suffix required by the owner. The owner
        callback handles embedding/PLE preparation, serial cache commit,
        cancellation safety, and assistant-branch rollback.
        """
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
            raise TypeError("max_new_tokens deve essere un intero")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens non può essere negativo")
        budget = int(max_new_tokens)
        transition_ids = _token_ids(transition_token_ids)
        if budget > 0 and not transition_ids:
            raise ValueError("serve almeno un transition token per ottenere i primi logits")

        decode_started_ns = int(self.monotonic_ns())
        generated: list[GeneratedToken] = []
        sampled_count = 0
        committed_count = 0
        first_logits_event: FirstLogitsReady | None = None
        terminal_token_id: int | None = None
        pending_token_id: int | None = None
        termination_reason = "budget_exhausted"

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

        candidate = await self._sample(logits, (), initial=True)
        candidate_sampled_ns = int(self.monotonic_ns())
        sampled_count += 1
        for token_index in range(budget):
            if await self._should_stop(stop_check):
                pending_token_id = _sampled_token_id(candidate)
                termination_reason = "cancelled"
                break
            history = tuple(token.token_id for token in generated)
            candidate = await self._select_candidate(candidate, logits, history)
            token_id = _sampled_token_id(candidate)
            if token_id in self.eos_token_ids:
                min_new_tokens = int(getattr(self.sampler, "min_new_tokens", 0))
                if len(generated) < min_new_tokens:
                    raise RuntimeError(
                        "MinNewTokensLengthLogitsProcessor ha restituito EOS prima della soglia "
                        f"({len(generated)} < {min_new_tokens})"
                    )
                terminal_token_id = token_id
                termination_reason = "eos"
                break

            if await self._should_stop(stop_check):
                pending_token_id = token_id
                termination_reason = "cancelled"
                break

            token_forward_started_ns = int(self.monotonic_ns())
            logits, token_hidden = await self._forward([token_id])
            token_hidden_ready_ns = int(self.monotonic_ns())
            committed_count += 1
            next_candidate = await self._sample(logits, tuple([*history, token_id]))
            next_sampled_ns = int(self.monotonic_ns())
            sampled_count += 1
            next_candidate = await self._after_generated_token(
                token_index=token_index,
                token_id=token_id,
                token_hidden=token_hidden,
                generated_ids=tuple([*history, token_id]),
                source_logits=logits,
                sampled_next=next_candidate,
            )
            token = GeneratedToken(
                token_id=token_id,
                hidden=token_hidden,
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
            )
            generated.append(token)
            if on_token is not None:
                await _resolve(on_token(token.token_id, token.hidden, token.timing))
            if token_index == 0 and first_token is not None:
                await _resolve(first_token(token))

            if token_index + 1 >= budget:
                termination_reason = "budget_exhausted"
                break
            candidate = next_candidate
            candidate_sampled_ns = next_sampled_ns
            source_forward_started_ns = token_forward_started_ns
            source_logits_ready_ns = token_hidden_ready_ns

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
        elif termination_reason == "budget_exhausted" and pending_token_id is not None:
            owner_action = "commit_pending_then_append_turn_eos"
        elif termination_reason == "budget_exhausted":
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
    "NativeDecodeResult",
    "TokenTiming",
    "UnifiedNativeDecoder",
]
