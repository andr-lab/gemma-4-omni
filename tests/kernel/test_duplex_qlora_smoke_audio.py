from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.launch import validate_smoke_audio


def test_smoke_audio_reports_physical_pcm_metrics():
    pcm = (0.4 * np.sin(np.arange(24_000, dtype=np.float32) * 0.03)).astype(np.float32)

    metrics = validate_smoke_audio(pcm)

    assert metrics["duration_s"] == 1.0
    assert 0.39 < metrics["peak"] < 0.41
    assert metrics["rms"] > 0.1
    assert metrics["clipping_fraction"] == 0.0


@pytest.mark.parametrize(
    "pcm, message",
    [
        (np.zeros(24000, dtype=np.float32), "silenzioso"),
        (np.full(24000, np.nan, dtype=np.float32), "non finito"),
        (np.ones(1200, dtype=np.float32) * 0.1, "durata"),
        (np.ones(24000, dtype=np.float32), "clipping"),
    ],
)
def test_smoke_audio_rejects_invalid_or_clipped_pcm(pcm, message):
    with pytest.raises(RuntimeError, match=message):
        validate_smoke_audio(pcm)
