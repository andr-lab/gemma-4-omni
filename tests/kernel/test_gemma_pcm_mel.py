"""Equivalence gate for the incremental PCM path and Gemma's extractor."""
from __future__ import annotations

import numpy as np

from duplex_kernel.turnhead import TurnHeadRuntime
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor


def test_incremental_pcm_mel_matches_gemma_extractor() -> None:
    rng = np.random.default_rng(20260830)
    pcm = rng.standard_normal(1600).astype(np.float32)
    extractor = Gemma4AudioFeatureExtractor()
    features = extractor([pcm])
    reference_all = np.asarray(features["input_features"][0], dtype=np.float32)
    mask = np.asarray(features["input_features_mask"][0], dtype=bool)
    reference = reference_all[mask]

    runtime = TurnHeadRuntime(device="cpu")
    counts = []
    cursor = 0
    for size in (97, 113, 241, 160, 389, 600):
        chunk = pcm[cursor:cursor + size]
        if not len(chunk):
            break
        counts.append(runtime.feed_pcm(chunk))
        cursor += len(chunk)

    actual = runtime._mel_buffer  # noqa: SLF001 — gate inspects the causal buffer.
    assert cursor == len(pcm)
    assert sum(counts) == len(reference) == 9
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2e-5)
    assert runtime._next_mel_frame == len(reference)  # noqa: SLF001
