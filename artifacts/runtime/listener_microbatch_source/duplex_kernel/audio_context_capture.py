"""Helpers for an opt-in V9 audio-context capture from the live Thinker prefill.

The capture is a snapshot of Gemma's multimodal LM hidden states at real audio
placeholder positions.  It does not run another audio or language-model pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from duplex_kernel.config import AUDIO_FRAME_MS


@dataclass(frozen=True)
class AudioContextHiddenSnapshot:
    hidden: torch.Tensor
    audio_placeholder_count: int
    selected_frame_count: int
    past_length: int
    audio_ms_per_token: int
    raw_frame_period_ms: int
    source_pcm_samples: int | None


def capture_audio_context_hidden(
    thinker: Any,
    *,
    input_ids: torch.Tensor,
    prefill_hidden: torch.Tensor,
    past_length: int = 0,
    audio_ms_per_token: int,
    raw_frame_period_ms: int = AUDIO_FRAME_MS,
    source_pcm_samples: int | None = None,
) -> AudioContextHiddenSnapshot:
    """Clone one audio-position state per raw-clock frame from a real prefill.

    ``prefill_hidden`` covers ``input_ids[:, past_length:]``.  The model's
    public placeholder mask identifies only current processor audio positions.
    Cadence is derived from the processor's ``audio_ms_per_token`` property
    and the native raw-frame clock.  Every run contributes its first token per
    clock frame; a final partial group is retained instead of discarded.
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("audio context input_ids must have shape (1,L)")
    if isinstance(past_length, bool) or not isinstance(past_length, int) or past_length < 0:
        raise ValueError("audio context past_length must be a nonnegative integer")
    if past_length >= int(input_ids.shape[1]):
        raise ValueError("audio context prefix consumes the complete input")
    if isinstance(audio_ms_per_token, bool) or not isinstance(audio_ms_per_token, int) or audio_ms_per_token <= 0:
        raise ValueError("audio_ms_per_token must be a positive integer")
    if isinstance(raw_frame_period_ms, bool) or not isinstance(raw_frame_period_ms, int) or raw_frame_period_ms <= 0:
        raise ValueError("raw_frame_period_ms must be a positive integer")
    if raw_frame_period_ms % audio_ms_per_token:
        raise ValueError("raw frame period must be an integer multiple of processor audio cadence")
    if source_pcm_samples is not None and (
        isinstance(source_pcm_samples, bool)
        or not isinstance(source_pcm_samples, int)
        or source_pcm_samples < 0
    ):
        raise ValueError("source_pcm_samples must be a nonnegative integer or None")
    if not isinstance(prefill_hidden, torch.Tensor) or prefill_hidden.ndim != 3:
        raise ValueError("prefill_hidden must have shape (1,L,1536)")
    if prefill_hidden.shape[0] != 1 or prefill_hidden.shape[1] != input_ids.shape[1] - past_length:
        raise ValueError("prefill hidden positions do not align with the uncached input suffix")
    if prefill_hidden.shape[-1] != 1536:
        raise ValueError("prefill_hidden must use Gemma hidden size 1536")

    model = getattr(thinker, "model", None)
    get_placeholder_mask = getattr(model, "get_placeholder_mask", None)
    if not callable(get_placeholder_mask):
        raise RuntimeError("Gemma audio placeholder mask API unavailable")
    masks = get_placeholder_mask(input_ids=input_ids)
    if not isinstance(masks, (tuple, list)) or len(masks) != 3:
        raise RuntimeError("Gemma placeholder mask must return image/video/audio masks")
    audio_mask = masks[2]
    if not isinstance(audio_mask, torch.Tensor) or tuple(audio_mask.shape) != tuple(input_ids.shape):
        raise RuntimeError("Gemma audio placeholder mask does not match input_ids")

    positions = torch.nonzero(audio_mask[0].to(dtype=torch.bool), as_tuple=False).reshape(-1)
    if positions.numel() == 0:
        raise ValueError("current prefill has no audio placeholder positions")
    if bool(torch.any(positions < past_length)):
        raise RuntimeError("audio placeholder overlaps a reused text-only prefix")

    slots_per_raw_frame = raw_frame_period_ms // audio_ms_per_token
    # A separated run may represent a distinct audio clip. Reset grouping at
    # each run so a boundary can never shift frame alignment.
    breaks = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).reshape(-1)
    run_starts = [0] + [int(value) + 1 for value in breaks.tolist()]
    run_ends = [int(value) + 1 for value in breaks.tolist()] + [int(positions.numel())]
    selected_absolute: list[torch.Tensor] = []
    for start, end in zip(run_starts, run_ends):
        run = positions[start:end]
        selected_absolute.append(run[::slots_per_raw_frame])
    selected = torch.cat(selected_absolute).to(device=prefill_hidden.device) - past_length
    snapshot = prefill_hidden.index_select(1, selected).detach().clone().contiguous()
    if snapshot.shape[1] == 0 or not bool(torch.isfinite(snapshot).all()):
        raise RuntimeError("audio context snapshot is empty or non-finite")
    return AudioContextHiddenSnapshot(
        hidden=snapshot,
        audio_placeholder_count=int(positions.numel()),
        selected_frame_count=int(snapshot.shape[1]),
        past_length=past_length,
        audio_ms_per_token=audio_ms_per_token,
        raw_frame_period_ms=raw_frame_period_ms,
        source_pcm_samples=source_pcm_samples,
    )


__all__ = ["AudioContextHiddenSnapshot", "capture_audio_context_hidden"]
