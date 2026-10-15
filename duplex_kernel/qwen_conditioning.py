"""Shared Qwen Talker audio-input conditioning contract.

The upstream Qwen3-TTS Talker forms every generated audio input after the BOS
from the sum of the sixteen RVQ codebook embeddings and exactly one text term.
When the synthesis text has already been consumed by prefill, that term is
``tts_pad_embed``; when the caller deliberately keeps trailing text explicit,
the corresponding trailing-text hidden is used until it is exhausted.

This module is intentionally independent of the runtime backends so eager,
slot-KV/native, differentiable-training, and reference paths cannot silently
carry different audio-input recipes.
"""
from __future__ import annotations

from typing import Any

import torch


QWEN_AUDIO_CONDITIONING_RECIPE = "qwen_code_sum_plus_text_term_v2"
QWEN_AUDIO_CONDITIONING_LEGACY_RECIPE = "qwen_code_sum_only_v1"
QWEN_CODEBOOK_COUNT = 16


def _talker_model(model: Any) -> Any:
    """Return the Qwen Talker model from either a wrapper or the model itself."""
    return getattr(model, "talker", model)


def _normalise_codes(codes: torch.Tensor) -> torch.Tensor:
    if not isinstance(codes, torch.Tensor):
        codes = torch.as_tensor(codes)
    if codes.dim() == 2:
        # Compatibility with the historical (B,T) CB0-only helper input.  The
        # conditioning contract itself still rejects it below: production
        # audio inputs must contain all sixteen codebooks.
        codes = codes.unsqueeze(-1)
    if codes.dim() != 3:
        raise ValueError("Qwen codes devono avere shape (B,T,16)")
    if int(codes.shape[-1]) != QWEN_CODEBOOK_COUNT:
        raise ValueError(
            "Qwen audio conditioning richiede tutti i 16 codebook RVQ: "
            f"got shape={tuple(codes.shape)}"
        )
    return codes


def aggregate_qwen_code_embeddings(
    talker_model: Any,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Sum CB0..CB15 exactly as the upstream Qwen Talker does."""
    codes = _normalise_codes(codes)
    model = _talker_model(talker_model)
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        raise TypeError("Qwen Talker non espone get_input_embeddings()")
    predictor = getattr(model, "code_predictor", None)
    predictor_embeddings = (
        None if predictor is None else predictor.get_input_embeddings()
    )
    if predictor_embeddings is None or len(predictor_embeddings) != QWEN_CODEBOOK_COUNT - 1:
        raise RuntimeError(
            "Qwen Talker code_predictor deve esporre 15 embedding residui"
        )

    parts = [get_input_embeddings()(codes[..., :1])]
    for codebook in range(1, QWEN_CODEBOOK_COUNT):
        parts.append(
            predictor_embeddings[codebook - 1](
                codes[..., codebook : codebook + 1]
            )
        )
    return torch.cat(parts, dim=-2).sum(dim=-2)


def _one_term(
    term: torch.Tensor,
    *,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(term, torch.Tensor):
        term = torch.as_tensor(term)
    if term.dim() == 2:
        term = term.unsqueeze(1)
    if term.dim() != 3 or int(term.shape[1]) != 1:
        raise ValueError(f"{name} deve avere shape (B,1,D), got {tuple(term.shape)}")
    if int(term.shape[-1]) != hidden_size:
        raise ValueError(
            f"{name} hidden size inatteso: {term.shape[-1]} != {hidden_size}"
        )
    if int(term.shape[0]) not in (1, batch_size):
        raise ValueError(
            f"{name} batch inatteso: {term.shape[0]} != {batch_size}"
        )
    term = term.to(device=device, dtype=dtype)
    if int(term.shape[0]) == 1 and batch_size != 1:
        term = term.expand(batch_size, -1, -1)
    if not bool(torch.isfinite(term).all()):
        raise ValueError(f"{name} contiene valori non finiti")
    return term


def build_legacy_qwen_audio_conditioning(
    talker_model: Any,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Immutable diagnostic for the pre-fix code-only recipe.

    This function is intentionally not used by production/training paths.  It
    exists only to quantify the effect of adding the upstream text term and to
    keep historical checkpoints/results interpretable.
    """
    return aggregate_qwen_code_embeddings(talker_model, codes)


def build_qwen_audio_conditioning(
    talker_model: Any,
    codes: torch.Tensor,
    *,
    tts_pad_embed: torch.Tensor | None = None,
    trailing_text_hidden: torch.Tensor | None = None,
    generation_step: int | None = None,
) -> torch.Tensor:
    """Build one or more generated-audio decoder inputs.

    ``codes[t]`` is the previous/generated RVQ frame consumed to predict the
    next codec frame.  The returned tensor is therefore ``sum(CB0..CB15)``
    plus exactly one text term:

    * no explicit trailing text -> ``tts_pad_embed``;
    * explicit trailing text and ``generation_step`` still in range -> that
      trailing-text hidden;
    * explicit trailing text exhausted -> ``tts_pad_embed``.

    The function never modifies BOS (BOS is prefill input) and never touches
    SubTalker inputs, which receive the CB0 embedding separately upstream.
    """
    audio = aggregate_qwen_code_embeddings(talker_model, codes)
    if audio.dim() != 3:
        raise RuntimeError(f"Qwen audio embedding shape inattesa: {tuple(audio.shape)}")

    trailing = None
    if trailing_text_hidden is not None:
        if generation_step is None or int(generation_step) < 0:
            raise ValueError(
                "generation_step è obbligatorio con trailing_text_hidden"
            )
        trailing = trailing_text_hidden
        if trailing.dim() == 2:
            # A 2-D value is the batch-1 shorthand (T,D), not (B,1,D).
            trailing = trailing.unsqueeze(0)
        if trailing.dim() != 3:
            raise ValueError(
                "trailing_text_hidden deve avere shape (B,T,D), "
                f"got {tuple(trailing.shape)}"
            )
        step = int(generation_step)
        if step < int(trailing.shape[1]):
            term = _one_term(
                trailing[:, step : step + 1],
                batch_size=int(audio.shape[0]),
                hidden_size=int(audio.shape[-1]),
                device=audio.device,
                dtype=audio.dtype,
                name="trailing_text_hidden",
            )
            return audio + term

    if tts_pad_embed is None:
        raise ValueError(
            "Qwen audio conditioning richiede tts_pad_embed quando il testo "
            "è già in prefill o il trailing text è esaurito"
        )
    pad = _one_term(
        tts_pad_embed,
        batch_size=int(audio.shape[0]),
        hidden_size=int(audio.shape[-1]),
        device=audio.device,
        dtype=audio.dtype,
        name="tts_pad_embed",
    )
    return audio + pad


def build_qwen_audio_conditioning_sequence(
    talker_model: Any,
    codes: torch.Tensor,
    *,
    tts_pad_embed: torch.Tensor | None = None,
    trailing_text_hidden: torch.Tensor | None = None,
    generation_start: int = 0,
) -> torch.Tensor:
    """Build a sequence while advancing trailing-text position per frame.

    The scalar helper is intentionally the primitive used by one decoder step;
    this wrapper prevents fixed-sequence fallback/trainer paths from applying
    one trailing-text vector to every committed RVQ frame.
    """
    codes = _normalise_codes(codes)
    if int(codes.shape[1]) == 0:
        raise ValueError("Qwen audio conditioning sequence non può essere vuota")
    if trailing_text_hidden is None:
        return build_qwen_audio_conditioning(
            talker_model,
            codes,
            tts_pad_embed=tts_pad_embed,
        )
    if trailing_text_hidden.dim() == 2:
        trailing_text_hidden = trailing_text_hidden.unsqueeze(0)
    if trailing_text_hidden.dim() != 3:
        raise ValueError(
            "trailing_text_hidden deve avere shape (B,T,D), "
            f"got {tuple(trailing_text_hidden.shape)}"
        )
    if not isinstance(generation_start, int) or generation_start < 0:
        raise ValueError("generation_start deve essere un intero non negativo")
    return torch.cat(
        [
            build_qwen_audio_conditioning(
                talker_model,
                codes[:, frame_id : frame_id + 1],
                tts_pad_embed=tts_pad_embed,
                trailing_text_hidden=trailing_text_hidden,
                generation_step=generation_start + frame_id,
            )
            for frame_id in range(int(codes.shape[1]))
        ],
        dim=1,
    )


__all__ = [
    "QWEN_AUDIO_CONDITIONING_LEGACY_RECIPE",
    "QWEN_AUDIO_CONDITIONING_RECIPE",
    "QWEN_CODEBOOK_COUNT",
    "aggregate_qwen_code_embeddings",
    "build_legacy_qwen_audio_conditioning",
    "build_qwen_audio_conditioning",
    "build_qwen_audio_conditioning_sequence",
]
