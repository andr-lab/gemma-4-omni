import json

import numpy as np

from duplex_kernel.tts_quality_diagnostics import (
    DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET,
    SYNTHETIC_DIALOGUE_FIXTURE_ID,
    TTSQualityDiagnosticCapture,
    apply_diagnostic_frame_budget,
)


def test_only_opted_in_diagnostics_extend_budget_and_stay_bounded():
    assert apply_diagnostic_frame_budget(
        48,
        diagnostic_max_frames=None,
        absolute_max_frames=140,
    ) == 48
    assert apply_diagnostic_frame_budget(
        48,
        diagnostic_max_frames=DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET,
        absolute_max_frames=140,
    ) == 96
    assert apply_diagnostic_frame_budget(
        120,
        diagnostic_max_frames=96,
        absolute_max_frames=140,
    ) == 96


def test_diagnostic_capture_is_opt_in_fixture_scoped_and_bounded(tmp_path):
    disabled = TTSQualityDiagnosticCapture(output_dir=tmp_path / "disabled", enabled=False)
    assert disabled.claim(SYNTHETIC_DIALOGUE_FIXTURE_ID) is None

    capture = TTSQualityDiagnosticCapture(
        output_dir=tmp_path / "enabled",
        enabled=True,
        max_records=2,
    )
    assert capture.claim("live_microphone") is None
    slots = [capture.claim(SYNTHETIC_DIALOGUE_FIXTURE_ID) for _ in range(3)]
    assert slots[0] is not None and slots[1] is not None and slots[2] is None


def test_diagnostic_capture_keeps_arrays_private_and_roundtrips_them(tmp_path):
    capture = TTSQualityDiagnosticCapture(output_dir=tmp_path, enabled=True, max_records=1)
    slot = capture.claim(SYNTHETIC_DIALOGUE_FIXTURE_ID)
    assert slot is not None
    hidden = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    audio_context = np.arange(24, dtype=np.float32).reshape(1, 6, 4)
    semantic_anchor = np.arange(4, dtype=np.float32).reshape(1, 1, 4)
    codes = np.arange(32, dtype=np.int32).reshape(2, 16)
    paths = capture.write(
        slot,
        {
            "metadata": {"response_id": "resp-test", "say_text": "Ciao."},
            "audio_context_hidden": audio_context,
            "semantic_anchor_hidden": semantic_anchor,
            "say_hidden": hidden,
            "v9_prefix": hidden[:, :2, :],
            "rvq_codes": codes,
            "streamed_pcm": np.array([0.1, -0.1], dtype=np.float32),
            "final_pcm": np.array([0.2, -0.2], dtype=np.float32),
        },
    )
    with paths["arrays"].open("rb") as handle:
        with np.load(handle, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["audio_context_hidden"], audio_context)
            np.testing.assert_array_equal(archive["semantic_anchor_hidden"], semantic_anchor)
            np.testing.assert_array_equal(archive["say_hidden"], hidden)
            np.testing.assert_array_equal(archive["rvq_codes"], codes)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["fixture_id"] == SYNTHETIC_DIALOGUE_FIXTURE_ID
    assert metadata["arrays"]["say_hidden"]["shape"] == [1, 3, 4]
    assert metadata["arrays"]["audio_context_hidden"]["shape"] == [1, 6, 4]
    assert metadata["arrays"]["semantic_anchor_hidden"]["shape"] == [1, 1, 4]
    assert "say_hidden" not in metadata


def test_preclamp_diagnostic_preserves_nonfinite_decoder_values(tmp_path):
    capture = TTSQualityDiagnosticCapture(output_dir=tmp_path, enabled=True, max_records=1)
    slot = capture.claim(SYNTHETIC_DIALOGUE_FIXTURE_ID)
    assert slot is not None
    preclamp = np.array([0.2, np.nan, 1.4], dtype=np.float32)

    paths = capture.write(slot, {"metadata": {"preclamp_nonfinite_count": 1}, "preclamp_pcm": preclamp})

    with paths["arrays"].open("rb") as handle:
        with np.load(handle, allow_pickle=False) as archive:
            np.testing.assert_equal(archive["preclamp_pcm"], preclamp)
