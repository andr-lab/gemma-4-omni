"""Token-cursor-aligned local response states for the frozen Qwen Talker.

Qwen inserts its first response text token in the prefill, then consumes the
remaining text tokens through ``trailing_text_hidden`` once per audio frame.
This module uses that existing token cursor; it does not infer acoustic
word/frame alignment or append text while a segment is running.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn

from duplex_kernel.response_intent import (
    RESPONSE_INTENT_INPUT_DIM,
    RESPONSE_INTENT_OUTPUT_DIM,
    RESPONSE_INTENT_PREFIX_CAPACITY,
    ResponseIntentProjector,
)


QWEN_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
QWEN_ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def build_token_char_spans(
    token_ids: Sequence[int],
    decode_token: Callable[[int], str],
    *,
    expected_text: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Recover exact visible-text offsets from the captured Gemma token IDs.

    Single-token decoding is used only to assign character spans. The joined
    decoded pieces must reconstruct the canonical h_txt exactly after the
    same outer ``strip`` used by the server; ambiguous tokenization fails.
    The returned indices select the matching hidden rows, dropping only tokens
    whose entire visible piece was removed by the outer strip.
    """
    if not isinstance(expected_text, str) or not expected_text:
        raise ValueError("expected response text must be nonempty")
    pieces: list[str] = []
    for token_id in token_ids:
        piece = decode_token(int(token_id))
        if not isinstance(piece, str):
            raise TypeError("token decoder must return strings")
        pieces.append(piece)
    decoded = "".join(pieces)
    if decoded.strip() != expected_text:
        raise ValueError(
            "Gemma response token pieces do not reconstruct canonical h_txt"
        )
    leading = len(decoded) - len(decoded.lstrip())
    trailing = len(decoded.rstrip())
    if trailing <= leading:
        raise ValueError("Gemma response token pieces contain no visible text")

    spans: list[tuple[int, int]] = []
    kept: list[int] = []
    cursor = 0
    clipped_pieces: list[str] = []
    for index, piece in enumerate(pieces):
        next_cursor = cursor + len(piece)
        overlap_start = max(cursor, leading)
        overlap_end = min(next_cursor, trailing)
        if overlap_end > overlap_start:
            spans.append((overlap_start - leading, overlap_end - leading))
            kept.append(index)
            clipped_pieces.append(decoded[overlap_start:overlap_end])
        cursor = next_cursor
    if "".join(clipped_pieces) != expected_text:
        raise ValueError("Gemma token character spans do not cover canonical h_txt")
    return spans, kept


def qwen_assistant_text_offsets(
    text: str,
    *,
    tokenizer: Any,
    processor: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return the exact ``[3:-5]`` Qwen raw-text IDs/offsets used in prefill."""
    if not isinstance(text, str) or not text:
        raise ValueError("Qwen response text must be nonempty")
    assistant_text = QWEN_ASSISTANT_PREFIX + text + QWEN_ASSISTANT_SUFFIX
    encoded = tokenizer(assistant_text, return_offsets_mapping=True)
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if input_ids is None or offsets is None:
        raise ValueError("Qwen tokenizer must return input_ids and character offsets")
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1 or len(offsets) != 1:
            raise ValueError("Qwen assistant text tokenization must be unbatched")
        input_ids, offsets = input_ids[0], offsets[0]
    ids = [int(value) for value in input_ids]
    char_offsets = [(int(start), int(end)) for start, end in offsets]
    if len(ids) != len(char_offsets) or len(ids) <= 8:
        raise ValueError("Qwen assistant framing does not satisfy the pinned [3:-5] layout")

    # This is the same processor call used by TalkerModel.build_prefill_parts.
    processor_result = processor(text=assistant_text, return_tensors="pt")
    processor_ids = processor_result.get("input_ids")
    if not isinstance(processor_ids, torch.Tensor) or processor_ids.ndim != 2 or processor_ids.shape[0] != 1:
        raise ValueError("Qwen processor did not return one framed input_ids row")
    if processor_ids[0].detach().cpu().tolist() != ids:
        raise ValueError("Qwen offset tokenizer IDs differ from the live processor prefill IDs")

    raw_ids = ids[3:-5]
    raw_full_offsets = char_offsets[3:-5]
    text_start = len(QWEN_ASSISTANT_PREFIX)
    text_end = text_start + len(text)
    raw_offsets: list[tuple[int, int]] = []
    for start, end in raw_full_offsets:
        clipped_start = max(start, text_start)
        clipped_end = min(end, text_end)
        if clipped_end <= clipped_start:
            raise ValueError("Qwen raw-text token has no character overlap with h_txt")
        raw_offsets.append((clipped_start - text_start, clipped_end - text_start))
    if not raw_ids or len(raw_ids) != len(raw_offsets):
        raise ValueError("Qwen raw-text token IDs and offsets are empty or misaligned")
    decoded = tokenizer.decode(
        raw_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise ValueError("Qwen [3:-5] raw text IDs do not reconstruct canonical h_txt")
    return raw_ids, raw_offsets


def align_hidden_states_by_char_overlap(
    gemma_hidden: torch.Tensor,
    gemma_offsets: Sequence[tuple[int, int]],
    qwen_offsets: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Mean overlapping Gemma final states for every Qwen raw-text token."""
    if gemma_hidden.ndim != 3 or int(gemma_hidden.shape[0]) != 1:
        raise ValueError("Gemma response hidden must have shape (1,T,D)")
    if int(gemma_hidden.shape[1]) != len(gemma_offsets) or not gemma_offsets:
        raise ValueError("Gemma response hidden and character offsets do not align")
    if not qwen_offsets:
        raise ValueError("Qwen response token offsets are empty")
    for name, spans in (("Gemma", gemma_offsets), ("Qwen", qwen_offsets)):
        for start, end in spans:
            if int(start) < 0 or int(end) <= int(start):
                raise ValueError(f"{name} token has an invalid character offset")
    aligned: list[torch.Tensor] = []
    for qwen_start, qwen_end in qwen_offsets:
        indices = [
            index
            for index, (gemma_start, gemma_end) in enumerate(gemma_offsets)
            if max(int(qwen_start), int(gemma_start)) < min(int(qwen_end), int(gemma_end))
        ]
        if not indices:
            raise ValueError("Qwen text token has no overlapping Gemma response hidden state")
        selected = gemma_hidden.index_select(
            1, torch.tensor(indices, device=gemma_hidden.device, dtype=torch.long)
        )
        aligned.append(selected.mean(dim=1))
    return torch.stack(aligned, dim=1)


def align_generated_response_hidden_to_qwen_tokens(
    response_hidden: torch.Tensor,
    gemma_token_ids: Sequence[int],
    *,
    gemma_tokenizer: Any,
    qwen_tokenizer: Any,
    qwen_processor: Any,
    text: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Align post-token Gemma states to the exact Qwen raw-text token offsets.

    ``response_hidden`` and ``gemma_token_ids`` must already describe the same
    non-special generated SAY tokens.  Only outer whitespace removed by the
    server's canonical ``strip()`` is dropped here.  Any tokenization or text
    mismatch fails closed; this function never falls back to a different
    conditioning source.
    """
    if response_hidden.ndim != 3 or int(response_hidden.shape[0]) != 1:
        raise ValueError("generated Gemma response hidden must have shape (1,T,D)")
    ids = [int(token_id) for token_id in gemma_token_ids]
    if len(ids) != int(response_hidden.shape[1]):
        raise ValueError("generated Gemma token IDs and response hidden rows do not align")

    def decode_one(token_id: int) -> str:
        return gemma_tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    gemma_offsets, kept_piece_indices = build_token_char_spans(
        ids,
        decode_one,
        expected_text=text,
    )
    if len(kept_piece_indices) != int(response_hidden.shape[1]):
        response_hidden = response_hidden.index_select(
            1,
            torch.tensor(
                kept_piece_indices,
                device=response_hidden.device,
                dtype=torch.long,
            ),
        )
    qwen_ids, qwen_offsets = qwen_assistant_text_offsets(
        text,
        tokenizer=qwen_tokenizer,
        processor=qwen_processor,
    )
    aligned = align_hidden_states_by_char_overlap(
        response_hidden,
        gemma_offsets,
        qwen_offsets,
    )
    if tuple(aligned.shape[:2]) != (1, len(qwen_ids)):
        raise ValueError("aligned response hidden does not match Qwen raw-text token count")
    return aligned, {
        "source": "same_response_post_token_final_normalized_hidden",
        "gemma_response_hidden_tokens": int(response_hidden.shape[1]),
        "qwen_raw_text_tokens": len(qwen_ids),
        "text_chars": len(text),
        "alignment": "character_overlap_mean",
        "outer_strip_applied": True,
    }


def select_frame_hidden_by_qwen_cursor(
    qwen_token_hidden: torch.Tensor,
    *,
    frame_count: int,
) -> tuple[torch.Tensor, list[str]]:
    """Use Qwen's frame cursor: prefilled token at frame 0, then one token/step.

    The first frame consumes BOS while Qwen's first text token is already in
    its prefill. At audio frame ``j > 0``, Qwen passes ``generation_step=j-1``
    to its trailing-text helper, whose index 0 is the token after that prefill.
    Exhausted response text holds the final response-local state; it never
    fabricates hidden states.
    """
    if qwen_token_hidden.ndim != 3 or int(qwen_token_hidden.shape[0]) != 1:
        raise ValueError("Qwen-aligned response hidden must have shape (1,T,D)")
    tokens = int(qwen_token_hidden.shape[1])
    if tokens <= 0 or int(frame_count) <= 0:
        raise ValueError("Qwen token count and audio frame count must be positive")
    cursors = [min(frame_id, tokens - 1) for frame_id in range(int(frame_count))]
    index = torch.tensor(cursors, device=qwen_token_hidden.device, dtype=torch.long)
    selected = qwen_token_hidden.index_select(1, index)
    provenance = [
        "qwen_text_cursor" if frame_id < tokens else "tail_hold"
        for frame_id in range(int(frame_count))
    ]
    return selected, provenance


def build_local_response_prefix(
    response_hidden: torch.Tensor,
    gemma_token_ids: Sequence[int],
    *,
    gemma_tokenizer: Any,
    qwen_tokenizer: Any,
    qwen_processor: Any,
    text: str,
    projector: ResponseIntentProjector,
    capacity: int = RESPONSE_INTENT_PREFIX_CAPACITY,
    excluded_gemma_token_ids: Sequence[int] = (),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create a fixed runtime prefix from the exact response-token cursor."""
    if int(capacity) != RESPONSE_INTENT_PREFIX_CAPACITY:
        raise ValueError("local response-intent prefix capacity must match training capacity 256")
    if response_hidden.ndim != 3 or int(response_hidden.shape[0]) != 1:
        raise ValueError("response hidden must have shape (1,T,1536)")
    if int(response_hidden.shape[-1]) != RESPONSE_INTENT_INPUT_DIM:
        raise ValueError("response hidden width differs from Gemma E2B final hidden")
    if int(response_hidden.shape[1]) != len(gemma_token_ids):
        raise ValueError("response token IDs and post-token hidden states do not align")
    excluded = {int(value) for value in excluded_gemma_token_ids}
    filtered_ids = [int(value) for value in gemma_token_ids if int(value) not in excluded]
    filtered_indices = [index for index, value in enumerate(gemma_token_ids) if int(value) not in excluded]
    if not filtered_ids:
        raise ValueError("no non-special Gemma SAY token remains")
    kept_hidden = response_hidden.index_select(
        1, torch.tensor(filtered_indices, device=response_hidden.device, dtype=torch.long)
    )

    def decode_one(token_id: int) -> str:
        return gemma_tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    gemma_offsets, kept_piece_indices = build_token_char_spans(
        filtered_ids, decode_one, expected_text=text
    )
    if len(kept_piece_indices) != int(kept_hidden.shape[1]):
        kept_hidden = kept_hidden.index_select(
            1,
            torch.tensor(kept_piece_indices, device=kept_hidden.device, dtype=torch.long),
        )
    _qwen_ids, qwen_offsets = qwen_assistant_text_offsets(
        text, tokenizer=qwen_tokenizer, processor=qwen_processor
    )
    qwen_hidden = align_hidden_states_by_char_overlap(kept_hidden, gemma_offsets, qwen_offsets)
    frame_hidden, provenance = select_frame_hidden_by_qwen_cursor(
        qwen_hidden, frame_count=int(capacity)
    )
    dtype = projector.linear.weight.dtype
    projected = projector(frame_hidden.to(dtype=dtype))
    if tuple(projected.shape) != (1, int(capacity), RESPONSE_INTENT_OUTPUT_DIM):
        raise ValueError("local response projector returned an unexpected prefix shape")
    metrics = {
        "source": "gemma_generated_say_response_token_states",
        "hidden_semantics": "post-token final-normalized states; same generated SAY IDs",
        "say_text": text,
        "gemma_hidden_tokens_after_special_filter": int(kept_hidden.shape[1]),
        "qwen_raw_text_tokens": len(qwen_offsets),
        "qwen_frame_cursor": "frame 0 uses prefilled text token 0; frame j uses min(j,Q-1)",
        "tail_hold_frames": provenance.count("tail_hold"),
        "tail_hold_provenance": "last response-token hidden after Qwen text cursor exhaustion",
        "prefix_capacity": int(capacity),
    }
    return projected, metrics


def load_local_cursor_projector(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> ResponseIntentProjector:
    """Load only a checkpoint that declares the local Qwen-token cursor contract."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"local response-intent checkpoint not found: {path}")
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("local response-intent checkpoint must be a mapping")
    contract = payload.get("training_contract")
    if not isinstance(contract, dict):
        raise ValueError("local response-intent checkpoint has no training contract")
    if int(contract.get("prefix_capacity", -1)) != RESPONSE_INTENT_PREFIX_CAPACITY:
        raise ValueError("local response-intent prefix capacity differs from runtime capacity 256")
    if contract.get("qwen_cursor") != (
        "frame 0 uses prefilled raw token 0; frame j selects min(j,Q-1); tail holds final token state"
    ):
        raise ValueError("checkpoint was not trained for the local Qwen text cursor")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("local response-intent checkpoint has no projector state_dict")
    model = ResponseIntentProjector()
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)
    return model
