from __future__ import annotations

import inspect
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Sequence

import torch

from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_ACTIONS,
    AudioResponseFallbackProposal,
    AudioResponseProtocolError,
)
from duplex_kernel.incremental_text_decoder import IncrementalTextDecoder


_NO_FORCED_TOKEN_OVERRIDE = object()


@dataclass(frozen=True)
class CandidateFallbackPool:
    """Candidate IDs proposed as a pool, with an optional execution metric tag."""

    candidate_ids: Sequence[int]
    metric: str | None = None


class ThinkerSampler:
    """Shared token-sampling pipeline used by HF and native Thinker decoders."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        device: Any,
        eos_token_ids: set[int],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        min_new_tokens: int = 0,
        backchannel_trie: dict[tuple[int, ...], Any] | None = None,
        runtime_metrics: dict[str, Any] | None = None,
        on_generation_start: Callable[[], Any] | None = None,
        on_generated_token: Callable[[int, int, str, torch.Tensor], Any] | None = None,
        on_decoded_text_flush: Callable[[str, int, int, torch.Tensor | None], Any] | None = None,
        on_candidate_token: Callable[[int, str, bool], bool] | None = None,
        on_candidate_fallback: Callable[[list[tuple[int, str, bool]]], Any] | None = None,
        on_action_logits_probe: Callable[[dict[str, Any]], Any] | None = None,
        generation_id: int | None = None,
        state: Any | None = None,
        best_native_logit_candidate_id: Callable[[list[int], torch.Tensor], int | None] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_ids = set(eos_token_ids)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.min_new_tokens = int(min_new_tokens)
        self.backchannel_trie = backchannel_trie
        self.runtime_metrics = runtime_metrics if runtime_metrics is not None else {}
        self.on_generation_start = on_generation_start
        self.on_generated_token_callback = on_generated_token
        self.on_decoded_text_flush_callback = on_decoded_text_flush
        self.on_candidate_token = on_candidate_token
        self.on_candidate_fallback = on_candidate_fallback
        self.on_action_logits_probe = on_action_logits_probe
        self.generation_id = generation_id
        self.state = state
        self.best_native_logit_candidate_id = best_native_logit_candidate_id
        self._pending_candidate_pool_override: tuple[int, str | None] | None = None
        self._pending_candidate_fallback: tuple[
            int, AudioResponseFallbackProposal, torch.Tensor
        ] | None = None
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator deve essere torch.Generator o None")
        self.generator = generator
        self.last_decoded_output = ""
        self._text_decoder = IncrementalTextDecoder(tokenizer, skip_special_tokens=True)
        self._base_token_ids: tuple[int, ...] = ()
        self.forced_action_logprobs: list[torch.Tensor] = []
        self.forced_action_overrides = 0
        self.forced_structure_logprobs: list[torch.Tensor] = []
        self.forced_structure_overrides = 0
        self.action_logits_probe_emitted = False

        self.min_new_tokens_processor = None
        if self.min_new_tokens > 0 and self.eos_token_ids:
            from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor

            self.min_new_tokens_processor = MinNewTokensLengthLogitsProcessor(
                prompt_length_to_skip=0,
                min_new_tokens=self.min_new_tokens,
                eos_token_id=sorted(self.eos_token_ids),
                device=str(device),
            )

    def _prepare_constrained_logits(
        self,
        logits: torch.Tensor,
        generated_ids: Sequence[int],
    ) -> torch.Tensor:
        """Apply history, trie, and current protocol EOS constraints before sampling."""
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("Thinker sampling expects logits with shape (1, vocabulary)")
        history = [int(token_id) for token_id in generated_ids]
        if self.repetition_penalty != 1.0 and history:
            curr_logits = logits.clone()
            for prev_id in set(history):
                if curr_logits[0, prev_id] < 0:
                    curr_logits[0, prev_id] = curr_logits[0, prev_id] * self.repetition_penalty
                else:
                    curr_logits[0, prev_id] = curr_logits[0, prev_id] / self.repetition_penalty
        else:
            curr_logits = logits
        if self.min_new_tokens_processor is not None:
            generated_tensor = torch.tensor([history], device=curr_logits.device, dtype=torch.long)
            curr_logits = self.min_new_tokens_processor(generated_tensor, curr_logits)
        if self.backchannel_trie is not None:
            allowed = self.backchannel_trie.get(tuple(history))
            if not allowed:
                raise RuntimeError(
                    "BACKCHANNEL token prefix non appartiene a nessuna frase train; "
                    f"generated_prefix={history!r}"
                )
            allowed_ids = torch.tensor(sorted(allowed), device=curr_logits.device, dtype=torch.long)
            if int(allowed_ids.min()) < 0 or int(allowed_ids.max()) >= curr_logits.shape[-1]:
                raise RuntimeError("BACKCHANNEL trie contiene ID fuori dal vocabolario Thinker")
            allowed_mask = torch.zeros_like(curr_logits, dtype=torch.bool)
            allowed_mask.index_fill_(-1, allowed_ids, True)
            curr_logits = curr_logits.masked_fill(~allowed_mask, float("-inf"))

        callback = self.on_candidate_token
        if callback is not None:
            illegal_eos_ids = [
                int(token_id)
                for token_id in sorted(self.eos_token_ids)
                if 0 <= int(token_id) < int(curr_logits.shape[-1])
                and bool(torch.isfinite(curr_logits[0, int(token_id)]))
                and not callback(int(token_id), "", True)
            ]
            if illegal_eos_ids:
                curr_logits = curr_logits.clone()
                curr_logits[0, illegal_eos_ids] = float("-inf")
        if bool(torch.isnan(curr_logits).any()) or bool(torch.isposinf(curr_logits).any()):
            raise RuntimeError("Thinker constrained logits contain NaN or positive infinity")
        if not bool(torch.isfinite(curr_logits).any()):
            raise RuntimeError("Thinker constrained logits contain no finite candidate")
        return curr_logits

    def sample_token(
        self,
        curr_logits: torch.Tensor,
        generated_ids: Sequence[int],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Apply Thinker constraints, temperature and nucleus sampling."""
        curr_logits = self._prepare_constrained_logits(curr_logits, generated_ids)
        if self.backchannel_trie is not None:
            self.runtime_metrics["backchannel_trie_steps"] = (
                self.runtime_metrics.get("backchannel_trie_steps", 0) + 1
            )
        if self.temperature <= 1e-5:
            token_tensor = curr_logits.argmax(dim=-1, keepdim=True)
            candidate_scores = curr_logits
        else:
            probs = torch.softmax(curr_logits / self.temperature, dim=-1)
            if bool(torch.isnan(probs).any()) or bool(torch.isposinf(probs).any()):
                raise RuntimeError("Thinker sampling probabilities are numerically invalid")
            if 0.0 < self.top_p < 1.0:
                values, indices = torch.sort(probs, descending=True)
                keep = torch.cumsum(values, dim=-1) <= self.top_p
                keep[..., 0] = True
                probs = torch.zeros_like(probs).scatter(-1, indices, values * keep)
                normalizer = probs.sum(dim=-1, keepdim=True)
                if not bool(torch.isfinite(normalizer).all()) or bool((normalizer <= 0).any()):
                    raise RuntimeError("Thinker top-p filtering produced an invalid distribution")
                probs = probs / normalizer
            elif not bool(torch.isfinite(probs.sum(dim=-1)).all()) or bool((probs.sum(dim=-1) <= 0).any()):
                raise RuntimeError("Thinker softmax produced an invalid distribution")
            candidate_scores = probs
            token_tensor = torch.multinomial(probs, 1, generator=self.generator)
        return token_tensor, int(token_tensor.item()), candidate_scores

    def sample_initial(
        self,
        logits: torch.Tensor,
        generated_ids: Sequence[int] = (),
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Sample the first token, then apply the exact initial-force contract."""
        forced_id = self.on_generation_start() if self.on_generation_start is not None else None
        candidate = self.sample_token(logits, generated_ids)
        if forced_id is None:
            return candidate
        if isinstance(forced_id, bool) or not isinstance(forced_id, int):
            raise RuntimeError("audio response initial forced token id must be an integer")
        if forced_id < 0 or forced_id >= int(logits.shape[-1]):
            raise RuntimeError("audio response initial forced token id outside Thinker vocabulary")
        self.forced_structure_logprobs.append(
            torch.log_softmax(logits.float(), dim=-1)[0, forced_id].detach()
        )
        if candidate[1] != forced_id:
            self.forced_structure_overrides += 1
        return torch.tensor([[forced_id]], device=self.device, dtype=torch.long), forced_id, candidate[2]

    def select_valid_candidate(
        self,
        candidate: tuple[torch.Tensor, int, torch.Tensor],
        *,
        source_logits: torch.Tensor,
        generated_ids: Sequence[int],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        token_tensor, token_id, candidate_scores = candidate
        callback = self.on_candidate_token
        if callback is None:
            return candidate
        if self.generation_id is not None and self.state is not None and not self.state.is_generation_current(
            generation_id=self.generation_id
        ):
            return candidate

        self.runtime_metrics["audio_response_protocol_v2_candidate_checks"] = (
            self.runtime_metrics.get("audio_response_protocol_v2_candidate_checks", 0) + 1
        )
        history = self._full_history(generated_ids)
        self._text_decoder.ensure_prefix(history)

        def candidate_text(candidate_id: int) -> tuple[str, bool]:
            terminal = candidate_id in self.eos_token_ids
            if terminal:
                return "", True
            return self._text_decoder.preview(int(candidate_id)), False

        text, terminal = candidate_text(token_id)
        if callback(token_id, text, terminal):
            return candidate
        rejected: list[tuple[int, str, bool]] = [(token_id, text, terminal)]
        constrained_logits = self._prepare_constrained_logits(source_logits, history).reshape(-1)
        finite_ids = torch.isfinite(constrained_logits).nonzero(as_tuple=False).reshape(-1)
        if finite_ids.numel() == 0:
            raise RuntimeError("Thinker constrained logits contain no finite candidate")
        ranked_ids = finite_ids[
            torch.argsort(constrained_logits.index_select(0, finite_ids), descending=True)
        ][:64]
        for candidate_value in ranked_ids.tolist():
            candidate_id = int(candidate_value)
            if candidate_id == token_id:
                continue
            text, terminal = candidate_text(candidate_id)
            self.runtime_metrics["audio_response_protocol_v2_candidate_checks"] += 1
            if not callback(candidate_id, text, terminal):
                rejected.append((candidate_id, text, terminal))
                continue
            self.forced_structure_logprobs.append(
                torch.log_softmax(source_logits.float(), dim=-1)[0, candidate_id].detach()
            )
            self.forced_structure_overrides += 1
            self.runtime_metrics["audio_response_protocol_v2_predecode_replacements"] = (
                self.runtime_metrics.get("audio_response_protocol_v2_predecode_replacements", 0) + 1
            )
            return torch.tensor([[candidate_id]], device=self.device, dtype=torch.long), candidate_id, candidate_scores

        if self.on_candidate_fallback is not None:
            fallback = self.on_candidate_fallback(rejected)
            fallback_id = None
            fallback_metric: str | None = None
            fallback_proposal: AudioResponseFallbackProposal | None = None
            if isinstance(fallback, AudioResponseFallbackProposal):
                fallback_proposal = fallback
                fallback_id = fallback.token_id
            elif isinstance(fallback, CandidateFallbackPool):
                fallback_metric = fallback.metric
                fallback = fallback.candidate_ids
            if fallback_id is None and isinstance(fallback, (tuple, list, set, frozenset)):
                legal_pool: list[int] = []
                for candidate_value in fallback:
                    if isinstance(candidate_value, bool) or not isinstance(candidate_value, Integral):
                        raise RuntimeError("audio response candidate pool IDs must be integers")
                    candidate_id = int(candidate_value)
                    if candidate_id < 0 or candidate_id >= int(source_logits.shape[-1]):
                        raise RuntimeError("audio response candidate pool ID outside Thinker vocabulary")
                    candidate_text_value, candidate_terminal = candidate_text(candidate_id)
                    self.runtime_metrics["audio_response_protocol_v2_candidate_checks"] += 1
                    if callback(candidate_id, candidate_text_value, candidate_terminal):
                        legal_pool.append(candidate_id)
                if self.best_native_logit_candidate_id is None:
                    best = (
                        max(
                            legal_pool,
                            key=lambda candidate_id: float(
                                source_logits.reshape(-1)[candidate_id].item()
                            ),
                        )
                        if legal_pool else None
                    )
                else:
                    best = self.best_native_logit_candidate_id(legal_pool, source_logits)
                if best is not None:
                    if isinstance(best, bool) or not isinstance(best, Integral) or int(best) not in legal_pool:
                        raise RuntimeError("native candidate selector returned an ID outside the legal fallback pool")
                    best = int(best)
                    self._pending_candidate_pool_override = (best, fallback_metric)
                    return torch.tensor([[best]], device=self.device, dtype=torch.long), best, candidate_scores
            elif fallback_id is None and fallback is not None:
                fallback_id = fallback
            if fallback_id is not None:
                if isinstance(fallback_id, bool) or not isinstance(fallback_id, int):
                    raise RuntimeError("audio response candidate fallback id must be an integer")
                if fallback_id < 0 or fallback_id >= int(source_logits.shape[-1]):
                    raise RuntimeError("audio response candidate fallback id outside Thinker vocabulary")
                fallback_text, fallback_terminal = candidate_text(fallback_id)
                if not callback(fallback_id, fallback_text, fallback_terminal):
                    raise AudioResponseProtocolError("bounded_closure_not_valid")
                if fallback_proposal is not None:
                    self._pending_candidate_fallback = (
                        fallback_id,
                        fallback_proposal,
                        torch.log_softmax(source_logits.float(), dim=-1)[0, fallback_id].detach(),
                    )
                else:
                    self.forced_structure_logprobs.append(
                        torch.log_softmax(source_logits.float(), dim=-1)[0, fallback_id].detach()
                    )
                    self.forced_structure_overrides += 1
                    self.runtime_metrics["audio_response_protocol_v2_predecode_replacements"] = (
                        self.runtime_metrics.get("audio_response_protocol_v2_predecode_replacements", 0) + 1
                    )
                    self.runtime_metrics["audio_response_protocol_v2_bounded_closures"] = (
                        self.runtime_metrics.get("audio_response_protocol_v2_bounded_closures", 0) + 1
                    )
                return torch.tensor([[fallback_id]], device=self.device, dtype=torch.long), fallback_id, candidate_scores
        raise AudioResponseProtocolError("no_legal_candidate_within_top64")

    def after_generated_token(
        self,
        *,
        token_index: int,
        token_id: int,
        token_hidden: torch.Tensor,
        generated_ids: Sequence[int],
        source_logits: torch.Tensor,
        logical_source_sequence: int | None = None,
        sampled_next: tuple[torch.Tensor, int, torch.Tensor] | None = None,
        sample_next: Callable[[], tuple[torch.Tensor, int, torch.Tensor]] | None = None,
        forced_override: Any = _NO_FORCED_TOKEN_OVERRIDE,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Commit the generated token callback before sampling the next candidate."""
        if sampled_next is not None and sample_next is not None:
            raise ValueError("pass sampled_next or sample_next, not both")

        def get_sampled_next() -> tuple[torch.Tensor, int, torch.Tensor]:
            candidate = sample_next() if sample_next is not None else sampled_next
            if not isinstance(candidate, (tuple, list)) or len(candidate) != 3:
                raise TypeError("Thinker next-token sampler must return a three-item candidate")
            return candidate

        callback = self.on_generated_token_callback
        generation_current = not (
            self.generation_id is not None
            and self.state is not None
            and not self.state.is_generation_current(generation_id=self.generation_id)
        )
        generated_history = [int(value) for value in generated_ids]
        if not generated_history or generated_history[-1] != int(token_id):
            raise ValueError("generated_ids must end with the committed token_id")
        token_text = self._commit_text_token(
            token_id=int(token_id),
            generated_ids=generated_history,
            source_sequence=(
                int(token_index)
                if logical_source_sequence is None
                else int(logical_source_sequence)
            ),
        )
        if callback is None or not generation_current:
            if generation_current:
                self._commit_candidate_fallback(int(token_id))
            else:
                self._pending_candidate_fallback = None
            sampled_next = get_sampled_next()
            self._record_candidate_pool_execution(int(token_id), accepted=generation_current)
            return sampled_next
        self._commit_candidate_fallback(int(token_id))
        forced = (
            callback(token_index, int(token_id), token_text, token_hidden.detach())
            if forced_override is _NO_FORCED_TOKEN_OVERRIDE
            else forced_override
        )
        sampled_next = get_sampled_next()
        self._record_candidate_pool_execution(int(token_id), accepted=True)
        if forced is None:
            return sampled_next
        forced_kind = "action"
        if isinstance(forced, tuple):
            if len(forced) != 2:
                raise RuntimeError("audio response forced token tuple must contain id and kind")
            forced, forced_kind = forced
        if isinstance(forced, bool) or not isinstance(forced, int):
            raise RuntimeError("audio response forced token id must be an integer")
        if forced_kind not in {"action", "structure"}:
            raise RuntimeError("audio response forced token kind is invalid")
        if forced < 0 or forced >= int(source_logits.shape[-1]):
            raise RuntimeError("audio response forced token id outside Thinker vocabulary")
        forced_logprob = torch.log_softmax(source_logits.float(), dim=-1)[0, forced].detach()
        if forced_kind == "action" and self.on_action_logits_probe is not None and not self.action_logits_probe_emitted:
            label_token_ids: dict[str, list[int]] = {}
            encode = getattr(self.tokenizer, "encode", None)
            if callable(encode):
                for label in AUDIO_RESPONSE_ACTIONS:
                    try:
                        encoded = encode(label, add_special_tokens=False)
                        if callable(getattr(encoded, "detach", None)):
                            encoded = encoded.detach().cpu().reshape(-1).tolist()
                        elif callable(getattr(encoded, "tolist", None)):
                            encoded = encoded.tolist()
                        while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
                            encoded = encoded[0]
                        label_token_ids[label] = [int(value) for value in encoded]
                    except Exception:
                        label_token_ids[label] = []
            distinct_first_ids = list(dict.fromkeys(
                values[0] for values in label_token_ids.values()
                if values and 0 <= values[0] < int(source_logits.shape[-1])
            ))
            if distinct_first_ids:
                candidate_ids = torch.tensor(distinct_first_ids, device=source_logits.device, dtype=torch.long)
                candidate_logits = source_logits[0].index_select(0, candidate_ids)
                raw_top_id_tensor = source_logits[0].argmax().reshape(1).to(candidate_logits.dtype)
                probe_values = torch.cat((raw_top_id_tensor, candidate_logits), dim=0).detach().float().cpu().tolist()
                raw_argmax_token_id = int(round(probe_values[0]))
                first_token_logits_by_id = {
                    token_id: float(value) for token_id, value in zip(distinct_first_ids, probe_values[1:])
                }
            else:
                raw_argmax_token_id = int(source_logits[0].argmax().item())
                first_token_logits_by_id = {}
            action_first_logits = {
                label: first_token_logits_by_id[values[0]]
                for label, values in label_token_ids.items()
                if values and values[0] in first_token_logits_by_id
            }
            raw_argmax_action_candidates = [
                label for label, values in label_token_ids.items()
                if values and values[0] == raw_argmax_token_id
            ]
            best_first_logit = max(action_first_logits.values(), default=None)
            best_first_action_candidates = (
                [label for label, value in action_first_logits.items() if value == best_first_logit]
                if best_first_logit is not None else []
            )
            self.on_action_logits_probe({
                "raw_argmax_token_id": raw_argmax_token_id,
                "sampled_token_id": int(sampled_next[1]),
                "forced_token_id": int(forced),
                "action_label_token_ids": label_token_ids,
                "action_label_first_token_logits": action_first_logits,
                "raw_argmax_action_candidates": raw_argmax_action_candidates,
                "best_first_logit_action_candidates": best_first_action_candidates,
            })
            self.action_logits_probe_emitted = True
        if forced_kind == "action":
            self.forced_action_logprobs.append(forced_logprob)
            if sampled_next[1] != forced:
                self.forced_action_overrides += 1
        else:
            self.forced_structure_logprobs.append(forced_logprob)
            if sampled_next[1] != forced:
                self.forced_structure_overrides += 1
        return torch.tensor([[forced]], device=self.device, dtype=torch.long), forced, sampled_next[2]

    def _commit_text_token(
        self,
        *,
        token_id: int,
        generated_ids: Sequence[int],
        source_sequence: int | None = None,
    ) -> str:
        history = self._full_history(generated_ids)
        if not history or history[-1] != int(token_id):
            raise ValueError("generated_ids must end with the committed token_id")
        self._text_decoder.ensure_prefix(history[:-1])
        if self._text_decoder.token_ids == tuple(history):
            token_text = self._text_decoder.source_tokens[-1].text_delta
        else:
            token_text = self._text_decoder.append(
                token_id,
                source_sequence=source_sequence,
                source_token_id=token_id,
            )
        self.last_decoded_output = self._text_decoder.text
        return token_text

    def _full_history(self, generated_ids: Sequence[int]) -> list[int]:
        history = [int(value) for value in generated_ids]
        if self._base_token_ids and history[:len(self._base_token_ids)] == list(self._base_token_ids):
            return history
        return [*self._base_token_ids, *history]

    def flush_decoded_text(
        self,
        generated_ids: Sequence[int],
        *,
        final_hidden: torch.Tensor | None = None,
        notify_callback: bool = True,
    ) -> str:
        """Finalize canonical text and report any replacement-at-end suffix."""
        self._text_decoder.ensure_prefix(self._full_history(generated_ids))
        suffix = self._text_decoder.finish()
        self.last_decoded_output = self._text_decoder.text
        if (
            notify_callback
            and suffix
            and self.on_decoded_text_flush_callback is not None
            and generated_ids
        ):
            token_id = int(generated_ids[-1])
            token_index = len(generated_ids) - 1
            self.on_decoded_text_flush_callback(
                suffix,
                token_index,
                token_id,
                final_hidden.detach() if isinstance(final_hidden, torch.Tensor) else None,
            )
        return suffix

    def _commit_candidate_fallback(self, token_id: int) -> None:
        pending = self._pending_candidate_fallback
        self._pending_candidate_fallback = None
        if pending is None or int(pending[0]) != int(token_id):
            return
        _candidate_id, proposal, logprob = pending
        if proposal.on_executed() is False:
            return
        self.forced_structure_logprobs.append(logprob)
        self.forced_structure_overrides += 1
        self.runtime_metrics["audio_response_protocol_v2_predecode_replacements"] = (
            self.runtime_metrics.get("audio_response_protocol_v2_predecode_replacements", 0) + 1
        )
        self.runtime_metrics["audio_response_protocol_v2_bounded_closures"] = (
            self.runtime_metrics.get("audio_response_protocol_v2_bounded_closures", 0) + 1
        )

    def _record_candidate_pool_execution(self, token_id: int, *, accepted: bool) -> None:
        pending = self._pending_candidate_pool_override
        self._pending_candidate_pool_override = None
        if not accepted or pending is None or int(pending[0]) != int(token_id):
            return
        self.runtime_metrics["audio_response_protocol_v2_lexicon_pool_overrides"] = (
            self.runtime_metrics.get("audio_response_protocol_v2_lexicon_pool_overrides", 0) + 1
        )
        if pending[1] is not None:
            metric_name = f"audio_response_protocol_v2_{pending[1]}_overrides"
            self.runtime_metrics[metric_name] = self.runtime_metrics.get(metric_name, 0) + 1

    async def after_generated_token_async(
        self,
        *,
        token_index: int,
        token_id: int,
        token_hidden: torch.Tensor,
        generated_ids: Sequence[int],
        source_logits: torch.Tensor,
        logical_source_sequence: int | None = None,
        sampled_next: tuple[torch.Tensor, int, torch.Tensor] | None = None,
        sample_next: Callable[[], Any] | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Await token-state updates before sampling the next constrained candidate."""
        if sampled_next is not None and sample_next is not None:
            raise ValueError("pass sampled_next or sample_next, not both")

        async def get_sampled_next() -> tuple[torch.Tensor, int, torch.Tensor]:
            candidate = await sample_next() if sample_next is not None else sampled_next
            if not isinstance(candidate, (tuple, list)) or len(candidate) != 3:
                raise TypeError("Thinker next-token sampler must return a three-item candidate")
            return candidate

        callback = self.on_generated_token_callback
        generation_current = not (
            self.generation_id is not None
            and self.state is not None
            and not self.state.is_generation_current(generation_id=self.generation_id)
        )
        if callback is None or not generation_current:
            candidate = await get_sampled_next()
            return self.after_generated_token(
                token_index=token_index,
                token_id=token_id,
                token_hidden=token_hidden,
                generated_ids=generated_ids,
                source_logits=source_logits,
                logical_source_sequence=logical_source_sequence,
                sampled_next=candidate,
            )
        self._commit_candidate_fallback(int(token_id))
        token_text = self._commit_text_token(
            token_id=int(token_id),
            generated_ids=generated_ids,
            source_sequence=(
                int(token_index)
                if logical_source_sequence is None
                else int(logical_source_sequence)
            ),
        )
        forced = callback(token_index, int(token_id), token_text, token_hidden.detach())
        if inspect.isawaitable(forced):
            forced = await forced
        candidate = await get_sampled_next()
        return self.after_generated_token(
            token_index=token_index,
            token_id=token_id,
            token_hidden=token_hidden,
            generated_ids=generated_ids,
            source_logits=source_logits,
            logical_source_sequence=logical_source_sequence,
            sampled_next=candidate,
            forced_override=forced,
        )
