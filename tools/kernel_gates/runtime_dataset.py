"""Shared validation-window alignment for runtime Gemma evidence."""
from __future__ import annotations

from collections.abc import Iterable


def aligned_validation_start(
    *,
    total_frames: int,
    validation_fraction: float,
    segment_ends: Iterable[int],
    window_frames: int = 64,
) -> int:
    """Return a complete validation-window boundary after the split boundary.

    ``text_by_window`` is indexed in fixed ``window_frames`` units.  A split
    beginning mid-window would pair its audio with unrelated text, so a segment
    boundary is first selected and then rounded *up* to the next full window.
    """
    total_frames = int(total_frames)
    window_frames = int(window_frames)
    validation_fraction = float(validation_fraction)
    if total_frames <= 0:
        raise ValueError("total_frames deve essere positivo")
    if window_frames <= 0:
        raise ValueError("window_frames deve essere positivo")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve essere strettamente tra 0 e 1")

    boundary = int((1.0 - validation_fraction) * total_frames)
    for end in sorted(int(value) for value in segment_ends):
        if end >= boundary:
            boundary = end
            break
    aligned = ((boundary + window_frames - 1) // window_frames) * window_frames
    if aligned >= total_frames:
        raise ValueError("nessuna finestra completa disponibile nella validation split")
    return aligned


def attested_validation_start(
    *,
    cache_start_frame: int | None,
    total_frames: int,
    validation_fraction: float,
    segment_ends: Iterable[int],
    window_frames: int = 64,
) -> int:
    """Validate the cache split provenance against the canonical window split."""
    if cache_start_frame is None:
        raise ValueError("cache priva di validation_start_frame attestato")
    expected = aligned_validation_start(
        total_frames=total_frames,
        validation_fraction=validation_fraction,
        segment_ends=segment_ends,
        window_frames=window_frames,
    )
    if int(cache_start_frame) != expected:
        raise ValueError(
            "validation_start_frame della cache non coincide con la split allineata: "
            f"cache={cache_start_frame}, expected={expected}"
        )
    return expected
