"""Prepare exact Gemma 4 text embeddings for native listener steps.

The active Transformers 5.5.0 ``Gemma4TextModel.forward`` builds text inputs
with ``embed_tokens``, ``get_per_layer_inputs`` and
``project_per_layer_inputs`` in that order. This module calls those same model
methods and applies the native kernel's even-width physical padding contract.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


_EXPECTED_LAYERS = 35
_EXPECTED_HIDDEN_SIZE = 1536
_EXPECTED_PLE_SIZE = 256
_PHYSICAL_Q = 2
_MAX_LOGICAL_Q = 8
_PREPARED_TRANSITION_CACHE_MAX_ENTRIES = 8


@dataclass(frozen=True)
class PreparedTokenInputs:
    """Native-ready input for one physical listener call.

    ``per_layer_inputs`` is the *projected* PLE tensor returned by the
    Thinker's ``project_per_layer_inputs`` method, not the token-identity
    intermediate. ``logical_q_lens`` records each row's valid prefix. Q1/Q2
    retain physical Q2; Q3..Q8 use the next even physical width.
    """

    inputs_embeds: torch.Tensor
    per_layer_inputs: torch.Tensor
    logical_q_lens: tuple[int, ...]
    padded_input_ids: torch.Tensor


def _is_language_model(candidate: Any) -> bool:
    return all(
        callable(getattr(candidate, name, None))
        for name in ("embed_tokens", "get_per_layer_inputs", "project_per_layer_inputs")
    ) and getattr(candidate, "config", None) is not None


def _walk_models(root: Any) -> Iterator[Any]:
    queue: deque[Any] = deque((root,))
    seen: set[int] = set()
    followed = ("model", "language_model", "base_model", "thinker")
    while queue:
        current = queue.popleft()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for name in followed:
            child = getattr(current, name, None)
            if child is not None and id(child) not in seen:
                queue.append(child)


def _resolve_language_model(root: Any) -> Any:
    for candidate in _walk_models(root):
        if _is_language_model(candidate):
            return candidate
    raise TypeError(
        "Thinker privo di Gemma4TextModel con embed_tokens, "
        "get_per_layer_inputs e project_per_layer_inputs"
    )


def _resolve_embedding_weight(language_model: Any) -> torch.Tensor:
    weight = getattr(language_model.embed_tokens, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    parameters = getattr(language_model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if isinstance(parameter, torch.Tensor):
                return parameter
    raise TypeError("embed_tokens non espone pesi Tensor per device e dtype")


def _resolve_text_config(root: Any, language_model: Any) -> Any:
    """Resolve Gemma4Config.get_text_config(), as used by the LM forward."""
    for candidate in _walk_models(root):
        config = getattr(candidate, "config", None)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            text_config = get_text_config()
            if text_config is not None:
                return text_config
    return language_model.config


def _normalize_ids(input_ids: Any) -> torch.Tensor:
    if isinstance(input_ids, torch.Tensor):
        integer_types = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if input_ids.dtype not in integer_types:
            raise TypeError("input_ids deve contenere interi")
        ids = input_ids.to(dtype=torch.long)
    else:
        try:
            ids = torch.as_tensor(input_ids)
        except (TypeError, ValueError) as exc:
            raise TypeError("input_ids deve essere una sequenza rettangolare di interi") from exc
        if ids.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("input_ids deve contenere interi")
        ids = ids.to(dtype=torch.long)

    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError(f"input_ids deve avere rank 1 o 2, ricevuto rank {ids.ndim}")
    if ids.shape[0] <= 0 or not 1 <= ids.shape[1] <= _MAX_LOGICAL_Q:
        raise ValueError(
            "input_ids deve avere almeno una riga e da una a otto colonne Q; "
            f"ricevuta shape {tuple(ids.shape)}"
        )
    return ids


def _normalize_sequence_ids(input_ids: Any) -> torch.Tensor:
    """Normalize one arbitrary-length sequence for sequential Q2 chunking."""
    if isinstance(input_ids, torch.Tensor):
        integer_types = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if input_ids.dtype not in integer_types:
            raise TypeError("input_ids deve contenere interi")
        ids = input_ids.to(dtype=torch.long)
    else:
        try:
            ids = torch.as_tensor(input_ids)
        except (TypeError, ValueError) as exc:
            raise TypeError("input_ids deve essere una sequenza di interi") from exc
        if ids.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError("input_ids deve contenere interi")
        ids = ids.to(dtype=torch.long)
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError("prepare_sequence accetta solo una sequenza rank 1 o [1,L]")
    return ids


def _normalize_logical_lengths(
    logical_q_lens: int | Sequence[int] | torch.Tensor | None,
    *,
    batch_size: int,
    provided_q: int,
) -> tuple[int, ...]:
    if logical_q_lens is None:
        return (int(provided_q),) * batch_size
    if isinstance(logical_q_lens, torch.Tensor):
        if logical_q_lens.ndim == 0:
            values = (logical_q_lens.item(),)
        elif logical_q_lens.ndim == 1:
            values = tuple(logical_q_lens.tolist())
        else:
            raise ValueError("logical_q_lens deve essere uno scalare o un vettore")
    elif isinstance(logical_q_lens, int) and not isinstance(logical_q_lens, bool):
        values = (logical_q_lens,)
    else:
        try:
            values = tuple(logical_q_lens)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("logical_q_lens deve essere uno scalare o una sequenza di interi") from exc

    if len(values) == 1 and batch_size == 1:
        values = (values[0],)
    if len(values) != batch_size:
        raise ValueError(
            f"logical_q_lens deve contenere {batch_size} valori, ricevuti {len(values)}"
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_LOGICAL_Q:
            raise ValueError("ogni logical_q_len deve essere tra 1 e 8")
        if value > provided_q:
            raise ValueError("logical_q_len supera il numero di token forniti per riga")
        normalized.append(int(value))
    return tuple(normalized)


class UnifiedTokenAdapter:
    """Convert Thinker chat-template token IDs to exact Gemma text inputs.

    Audio listener ticks remain physical Q2. Text appends may use Q3..Q8 and
    are padded internally to the next even width. By default this adapter
    validates the production ABI requirements (CUDA BF16); ``require_cuda``
    can be disabled for small CPU contract tests using a fake text model.
    """

    def __init__(
        self,
        thinker: Any,
        tokenizer: Any | None = None,
        *,
        require_cuda: bool = True,
    ) -> None:
        self.thinker = thinker
        self.language_model = _resolve_language_model(thinker)
        self.tokenizer = tokenizer if tokenizer is not None else getattr(thinker, "tokenizer", None)
        self.require_cuda = bool(require_cuda)
        self.config = self.language_model.config
        self.text_config = _resolve_text_config(thinker, self.language_model)
        self.pad_token_id = getattr(self.config, "pad_token_id", None)
        self.vocab_size = getattr(self.config, "vocab_size", None)

        dimensions = (
            ("num_hidden_layers", getattr(self.config, "num_hidden_layers", None), _EXPECTED_LAYERS),
            ("hidden_size", getattr(self.config, "hidden_size", None), _EXPECTED_HIDDEN_SIZE),
            (
                "hidden_size_per_layer_input",
                getattr(self.config, "hidden_size_per_layer_input", None),
                _EXPECTED_PLE_SIZE,
            ),
        )
        for name, actual, expected in dimensions:
            if actual != expected:
                raise ValueError(f"config Gemma incompatibile: {name}={actual}, atteso {expected}")
        if isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int) or self.pad_token_id < 0:
            raise ValueError("config Gemma deve fornire un pad_token_id valido")
        if self.vocab_size is None:
            self.vocab_size = getattr(self.language_model, "vocab_size", None)
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError("config Gemma deve fornire vocab_size valido")
        if self.pad_token_id >= self.vocab_size:
            raise ValueError("pad_token_id fuori dal vocabolario Gemma")

        self._embedding_weight = _resolve_embedding_weight(self.language_model)
        self._prepared_transition_cache: OrderedDict[
            tuple[Any, ...], PreparedTokenInputs
        ] = OrderedDict()
        self._prepared_transition_cache_hits = 0
        self._prepared_transition_cache_misses = 0
        if self.require_cuda and self._embedding_weight.device.type != "cuda":
            raise ValueError("adapter nativo richiede pesi Gemma su CUDA")
        if self.require_cuda and self._embedding_weight.dtype != torch.bfloat16:
            raise ValueError("adapter nativo richiede pesi Gemma BF16")

    def prepare(
        self,
        input_ids: Any,
        logical_q_lens: int | Sequence[int] | torch.Tensor | None = None,
        *,
        protocol: str | None = None,
    ) -> PreparedTokenInputs:
        """Return exact model embeddings and fully projected PLE for physical Q2.

        A rank-1 ID sequence represents one row. Rank-2 input is ``[N,Q]``;
        rows with logical Q=1 always receive the configured Gemma PAD ID in
        their physical second slot, even if a caller supplied another value.
        """
        ids = _normalize_ids(input_ids)
        batch_size, provided_q = (int(ids.shape[0]), int(ids.shape[1]))
        logical_lengths = _normalize_logical_lengths(
            logical_q_lens,
            batch_size=batch_size,
            provided_q=provided_q,
        )
        cache_key: tuple[Any, ...] | None = None
        if protocol is not None:
            if not isinstance(protocol, str) or not protocol.strip():
                raise ValueError("protocol must be a non-empty string when caching a transition")
            if batch_size == 1:
                cache_key = (
                    protocol.strip(),
                    tuple(int(value) for value in ids[0].detach().cpu().tolist()),
                    logical_lengths,
                    str(self._embedding_weight.device),
                    str(self._embedding_weight.dtype),
                    id(self.language_model),
                    id(self._embedding_weight),
                )
                cached = self._prepared_transition_cache.get(cache_key)
                if cached is not None:
                    self._prepared_transition_cache.move_to_end(cache_key)
                    self._prepared_transition_cache_hits += 1
                    return cached
                self._prepared_transition_cache_misses += 1
        physical_q = max(
            _PHYSICAL_Q,
            max(value + (value & 1) for value in logical_lengths),
        )

        active_mask = torch.arange(provided_q, device=ids.device).unsqueeze(0) < torch.tensor(
            logical_lengths, device=ids.device
        ).unsqueeze(1)
        active_ids = ids[active_mask]
        if active_ids.numel() and (
            bool((active_ids < 0).any()) or bool((active_ids >= self.vocab_size).any())
        ):
            raise ValueError("input_ids contiene token fuori dal vocabolario Gemma")

        device = self._embedding_weight.device
        physical_ids = torch.full(
            (batch_size, physical_q),
            int(self.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        ids = ids.to(device=device)
        copy_q = min(provided_q, physical_q)
        physical_ids[:, :copy_q] = ids[:, :copy_q]
        for row, logical_q in enumerate(logical_lengths):
            if logical_q < physical_q:
                physical_ids[row, logical_q:] = int(self.pad_token_id)

        with torch.inference_mode():
            inputs_embeds = self.language_model.embed_tokens(physical_ids)
            token_identity = self.language_model.get_per_layer_inputs(physical_ids, inputs_embeds)
            per_layer_inputs = self.language_model.project_per_layer_inputs(inputs_embeds, token_identity)

        expected_embeds = (batch_size, physical_q, _EXPECTED_HIDDEN_SIZE)
        expected_ple = (batch_size, physical_q, _EXPECTED_LAYERS, _EXPECTED_PLE_SIZE)
        if not isinstance(inputs_embeds, torch.Tensor) or tuple(inputs_embeds.shape) != expected_embeds:
            raise ValueError(
                f"inputs_embeds Gemma inattesi: atteso {expected_embeds}, "
                f"ricevuto {getattr(inputs_embeds, 'shape', None)}"
            )
        if not isinstance(token_identity, torch.Tensor) or tuple(token_identity.shape) != expected_ple:
            raise ValueError(
                f"PLE token-identity inatteso: atteso {expected_ple}, "
                f"ricevuto {getattr(token_identity, 'shape', None)}"
            )
        if not isinstance(per_layer_inputs, torch.Tensor) or tuple(per_layer_inputs.shape) != expected_ple:
            raise ValueError(
                f"PLE proiettato inatteso: atteso {expected_ple}, "
                f"ricevuto {getattr(per_layer_inputs, 'shape', None)}"
            )
        if inputs_embeds.device != self._embedding_weight.device:
            raise ValueError("inputs_embeds deve restare sul device dei pesi Gemma")
        if token_identity.device != inputs_embeds.device or per_layer_inputs.device != inputs_embeds.device:
            raise ValueError("inputs_embeds e PLE devono condividere il device")
        if token_identity.dtype != inputs_embeds.dtype or per_layer_inputs.dtype != inputs_embeds.dtype:
            raise ValueError("inputs_embeds e PLE devono condividere il dtype")
        if self.require_cuda and (inputs_embeds.device.type != "cuda" or inputs_embeds.dtype != torch.bfloat16):
            raise ValueError("ABI listener richiede inputs_embeds e PLE CUDA BF16")

        prepared = PreparedTokenInputs(
            inputs_embeds=inputs_embeds.contiguous(),
            per_layer_inputs=per_layer_inputs.contiguous(),
            logical_q_lens=logical_lengths,
            padded_input_ids=physical_ids.contiguous(),
        )
        if cache_key is not None:
            self._prepared_transition_cache[cache_key] = prepared
            self._prepared_transition_cache.move_to_end(cache_key)
            while len(self._prepared_transition_cache) > _PREPARED_TRANSITION_CACHE_MAX_ENTRIES:
                self._prepared_transition_cache.popitem(last=False)
        return prepared

    @property
    def prepared_transition_cache_info(self) -> dict[str, int]:
        return {
            "size": len(self._prepared_transition_cache),
            "hits": self._prepared_transition_cache_hits,
            "misses": self._prepared_transition_cache_misses,
            "max_entries": _PREPARED_TRANSITION_CACHE_MAX_ENTRIES,
        }

    def prepare_sequence(self, input_ids: Any) -> Iterator[PreparedTokenInputs]:
        """Yield physical Q2 calls for a single arbitrary-length token sequence."""
        sequence = _normalize_sequence_ids(input_ids)
        if sequence.numel() == 0:
            raise ValueError("prepare_sequence richiede almeno un token")
        for start in range(0, int(sequence.numel()), _PHYSICAL_Q):
            chunk = sequence[start : start + _PHYSICAL_Q].unsqueeze(0)
            yield self.prepare(chunk, logical_q_lens=int(chunk.shape[1]))

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
        **template_kwargs: Any,
    ) -> torch.Tensor:
        """Tokenize messages with the loaded Thinker's actual chat template."""
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            raise RuntimeError("serve il tokenizer del Thinker con apply_chat_template")
        if "tokenize" in template_kwargs:
            raise ValueError("encode_chat forza tokenize=True per mantenere gli ID del template")
        encoded = apply_template(
            messages,
            tokenize=True,
            add_generation_prompt=bool(add_generation_prompt),
            **template_kwargs,
        )
        if isinstance(encoded, Mapping):
            if "input_ids" not in encoded:
                raise ValueError("apply_chat_template non ha restituito input_ids")
            encoded = encoded["input_ids"]
        if isinstance(encoded, torch.Tensor):
            ids = encoded.to(dtype=torch.long)
        else:
            ids = torch.as_tensor(encoded, dtype=torch.long)
        if ids.ndim == 2 and ids.shape[0] == 1:
            ids = ids[0]
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError(f"chat template deve restituire una sequenza non vuota, ricevuto {tuple(ids.shape)}")
        return ids

    def prepare_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
        **template_kwargs: Any,
    ) -> Iterator[PreparedTokenInputs]:
        """Tokenize a full chat header/body through Thinker, then yield Q2 calls."""
        return self.prepare_sequence(
            self.encode_chat(
                messages,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
        )

    def logits_from_final_norm_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the exact Thinker logits path to natively final-normed hidden states."""
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden deve essere un torch.Tensor")
        for candidate in _walk_models(self.thinker):
            lm_head = getattr(candidate, "lm_head", None)
            if callable(lm_head):
                logits = lm_head(hidden)
                break
            get_output_embeddings = getattr(candidate, "get_output_embeddings", None)
            if callable(get_output_embeddings):
                output_embeddings = get_output_embeddings()
                if callable(output_embeddings):
                    logits = output_embeddings(hidden)
                    break
        else:
            raise TypeError("Thinker privo di lm_head utilizzabile")

        # Gemma4ForConditionalGeneration.forward applies this model-configured
        # transform after lm_head. Keep the same dtype and operation order.
        if (softcap := getattr(self.text_config, "final_logit_softcapping", None)) is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        return logits


__all__ = ["PreparedTokenInputs", "UnifiedTokenAdapter"]
