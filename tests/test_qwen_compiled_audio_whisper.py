import wave

import numpy as np
import pytest

from tools.benchmarks.eval_qwen_compiled_audio_whisper import (
    make_report,
    normalize_number_tokens,
    read_wav_metadata,
    transcript_metrics,
    validate_model_snapshot,
)


def test_number_normalization_matches_spoken_digits_and_split_numeric_groups():
    reference = "Perfetto, codice ordine: 4829."

    spoken = transcript_metrics(reference, "Perfetto, codice ordine: quattro otto due nove.", "4829")
    split_number = transcript_metrics(reference, "Perfetto, codice ordine: 48 29.", "4829")

    assert spoken["raw_wer"] > 0
    assert spoken["number_normalized_wer"] == 0
    assert spoken["expected_order_number_present"] is True
    assert split_number["number_normalized_wer"] == 0
    assert normalize_number_tokens("4829", "4829") == normalize_number_tokens("48 29", "4829")


def test_number_normalization_does_not_infer_unlisted_cardinal_morphology():
    normalized = normalize_number_tokens("quattromila ottocentoventinove", "4829")

    assert "number4829" not in normalized.split()


def test_report_marks_unavailable_compiled_comparison_explicitly():
    report = make_report(
        reference_text="Perfetto, ordine 4829.",
        expected_digits="4829",
        model={"model_id": "Systran/faster-whisper-small"},
        sample_results=[{"role": "eager", "transcript": "Perfetto, ordine 4829."}],
        compiled_requested=False,
    )

    assert report["status"] == "eager_only_compiled_unavailable"
    assert report["comparison"]["status"] == "compiled_wav_not_supplied"
    assert report["comparison"]["transcript_exact_match"] is None


def test_report_compares_only_when_both_audio_transcripts_exist():
    report = make_report(
        reference_text="Ordine 4829.",
        expected_digits="4829",
        model={"model_id": "Systran/faster-whisper-small"},
        sample_results=[
            {"role": "eager", "transcript": "Ordine 4829.", "metrics": {"expected_order_number_present": True}},
            {"role": "compiled", "transcript": "Ordine quattro otto due nove.", "metrics": {"expected_order_number_present": True}},
        ],
        compiled_requested=True,
    )

    assert report["status"] == "pair_complete"
    assert report["comparison"]["status"] == "paired"
    assert report["comparison"]["transcript_exact_match"] is False
    assert report["comparison"]["both_asr_transcripts_contain_expected_number"] is True


def test_wav_metadata_reads_real_pcm16_shape(tmp_path):
    path = tmp_path / "tiny.wav"
    samples = np.asarray([0, 1, -1, 32767, -32768], dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(samples.tobytes())

    metadata = read_wav_metadata(path)

    assert metadata["channels"] == 1
    assert metadata["sample_width_bytes"] == 2
    assert metadata["sample_rate_hz"] == 24000
    assert metadata["frames"] == 5


def test_model_snapshot_requires_payload_not_just_directory(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="model.bin"):
        validate_model_snapshot(snapshot)
