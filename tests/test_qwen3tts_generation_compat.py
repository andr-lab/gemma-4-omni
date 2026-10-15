import hashlib
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dialogue_eval.audio_metrics import compute_physical_audio_metrics
from dialogue_eval.evaluator import calculate_wer, contains_italian_digit_sequence
from tools.generate_scenario_audio_qwen3tts import (
    _find_local_snapshot,
    _force_offline_model_loading,
    _inspect_clip,
    _manifest_is_complete,
    _numeric_normalized_text,
    _prepare_output_dir,
    _pretest_passed,
    _snapshot_verified_audio_cache,
    _spoken_text_for_tts,
    _validate_cache_clip,
)


def test_italian_numeric_codes_are_pronounced_digit_by_digit_without_changing_reference():
    expected = "Buongiorno, mi chiamo Marco e il mio codice ordine è 4829. Ricordalo per favore."

    spoken = _spoken_text_for_tts(expected, "Italian")

    assert spoken == "Buongiorno, mi chiamo Marco e il mio codice ordine è quattro, otto, due, nove. Ricordalo per favore."
    transcript = "Buongiorno, mi chiamo Marco e il mio codice ordine è 4829. Ricordalo per favore."
    assert calculate_wer(expected, transcript) == 0.0
    assert contains_italian_digit_sequence(transcript, "4829")


def test_all_long_numeric_codes_are_pronounced_digit_by_digit():
    expected = "Prima ti ho detto 4829 e il codice era 9999, corretto?"

    assert _spoken_text_for_tts(expected, "it") == (
        "Prima ti ho detto quattro, otto, due, nove e il codice era "
        "nove, nove, nove, nove, corretto?"
    )


def test_pretest_accepts_correct_code_split_into_asr_digit_tokens():
    expected = "Buongiorno, mi chiamo Marco e il mio codice ordine è 4829. Ricordalo per favore."
    transcript = "Buongiorno, mi chiamo Marco e il mio codice ordine è 4 8 2 9. Ricordalo per favore."
    raw_wer = calculate_wer(expected, transcript)
    normalized_wer = calculate_wer(_numeric_normalized_text(expected), _numeric_normalized_text(transcript))

    assert raw_wer == 0.2857
    assert normalized_wer == 0.0
    assert _pretest_passed({
        "numeric_normalized_wer": normalized_wer,
        "contains_marco": True,
        "contains_code_4829": True,
        "asr_quality_passed": True,
    })


def test_pretest_rejects_wrong_code_even_when_normalized_wer_is_low():
    assert not _pretest_passed({
        "numeric_normalized_wer": 0.0714,
        "contains_marco": True,
        "contains_code_4829": False,
        "asr_quality_passed": False,
    })


def test_numeric_normalized_wer_equates_digitwise_spelling_and_asr_numeric_output():
    reference = "Buongiorno, mi chiamo Marco. Codice quattro, otto, due, nove."
    hypothesis = "Buongiorno, mi chiamo Marco. Codice 4829."

    assert _numeric_normalized_text(reference) == "buongiorno mi chiamo marco codice 4829"
    assert calculate_wer(_numeric_normalized_text(reference), _numeric_normalized_text(hypothesis)) == 0.0
    assert calculate_wer(reference, hypothesis) > 0.0


def test_numeric_normalization_preserves_spoken_times_as_distinct_numbers():
    reference = "Siamo in quattro persone per le venti e trenta."
    hypothesis = "Siamo in 4 persone per le 20 e 30."

    assert _numeric_normalized_text(reference) == _numeric_normalized_text(hypothesis)


def test_snapshot_search_prefers_the_first_cached_model_family(tmp_path):
    preferred = tmp_path / "preferred" / "snapshots" / "local-revision"
    fallback = tmp_path / "fallback" / "snapshots" / "local-revision"
    preferred.mkdir(parents=True)
    fallback.mkdir(parents=True)

    snapshot = _find_local_snapshot(
        [str(preferred.parent / "*"), str(fallback.parent / "*")],
        "test model",
    )

    assert snapshot == preferred


def _valid_audio_clip(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    expected = "Ciao, mi chiamo Marco."
    audio_path = tmp_path / "clip.wav"
    pcm = (0.1 * np.sin(np.linspace(0.0, 20.0, 16000, dtype=np.float32))).astype(np.float32)
    sf.write(audio_path, pcm, 16_000, subtype="PCM_16")
    saved_pcm, sample_rate = sf.read(audio_path, dtype="float32")
    metrics = compute_physical_audio_metrics(saved_pcm, sample_rate=sample_rate)
    transcript = expected
    clip = {
        "filename": audio_path.name,
        "scenario_id": "test_scenario",
        "step_id": "turn_1_user",
        "text": expected,
        "spoken_text": expected,
        "language": "Italian",
        "sample_rate_hz": sample_rate,
        "asr_engine": "faster-whisper",
        "asr_language": "Italian",
        "asr_transcript": transcript,
        "wer": calculate_wer(expected, transcript),
        "numeric_normalized_wer": 0.0,
        "contains_marco": True,
        "expected_numeric_codes": {},
        "contains_code_4829": False,
        "asr_quality_passed": True,
        "audio_metrics": metrics,
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
    }
    return clip, audio_path, expected


def test_proof_can_force_huggingface_model_loading_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    _force_offline_model_loading()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_snapshot_uses_verified_bytes_and_is_independent_of_later_cache_changes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    wav_name = "input.wav"
    original = b"verified wav bytes"
    (source / wav_name).write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    snapshot = _snapshot_verified_audio_cache(
        source,
        tmp_path / "run" / "input_audio_cache",
        {wav_name: digest},
    )

    (source / wav_name).write_bytes(b"replacement bytes")

    assert (snapshot / wav_name).read_bytes() == original
    assert not (snapshot / wav_name).stat().st_mode & 0o222


def test_snapshot_rejects_cache_mutated_after_preflight_hashing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    wav_name = "input.wav"
    original = b"verified wav bytes"
    (source / wav_name).write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    (source / wav_name).write_bytes(b"changed after preflight")
    snapshot_path = tmp_path / "run" / "input_audio_cache"

    with pytest.raises(RuntimeError, match="cache changed after verification"):
        _snapshot_verified_audio_cache(source, snapshot_path, {wav_name: digest})
    assert not snapshot_path.exists()


def test_clip_validator_rejects_manifest_transcript_not_supported_by_fresh_audio_asr(tmp_path):
    clip, audio_path, expected = _valid_audio_clip(tmp_path)

    errors = _validate_cache_clip(
        clip,
        expected_text=expected,
        audio_path=audio_path,
        independent_transcript="rumore incomprensibile",
    )

    assert any("fresh independent ASR" in error for error in errors)


def test_mode_b_asr_gate_requires_spoken_4829_even_when_normalized_wer_is_below_limit(tmp_path):
    expected_text = (
        "Buongiorno, mi chiamo Marco. "
        "Questa frase lunga serve a verificare con accuratezza i controlli della trascrizione. "
        "Il mio codice ordine è quattro, otto, due, nove."
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_path / "kickoff.wav"
    pcm = (0.1 * np.sin(np.linspace(0.0, 20.0, 16_000, dtype=np.float32))).astype(np.float32)
    sf.write(wav_path, pcm, 16_000, subtype="PCM_16")
    wrong_transcript = expected_text.replace("nove", "cinque")

    class FakeASR:
        def transcribe(self, *_args, **_kwargs):
            return [SimpleNamespace(text=wrong_transcript)], None

    clip = _inspect_clip(
        asr=FakeASR(),
        out_dir=tmp_path,
        filename=wav_path.name,
        scenario_id="mode_b",
        step_id="kickoff",
        expected_text=expected_text,
        spoken_text=expected_text,
        language="Italian",
    )

    assert clip["numeric_normalized_wer"] < 0.15
    assert clip["contains_code_4829"] is False
    assert clip["asr_quality_passed"] is False


def test_complete_manifest_requires_final_status_and_all_cache_gates():
    def record(filename, scenario_id, step_id):
        return {
            "filename": filename,
            "scenario_id": scenario_id,
            "step_id": step_id,
            "text": "canonical input",
            "spoken_text": "spoken input",
            "language": "Italian",
            "asr_engine": "faster-whisper",
            "asr_language": "Italian",
            "asr_transcript": "canonical input",
            "audio_sha256": "a" * 64,
            "sample_rate_hz": 16_000,
            "asr_quality_passed": True,
            "audio_metrics": {"finite": True},
        }

    manifest = {
        "schema_version": 2,
        "generator": "Qwen3-TTS-12Hz-1.7B-Base",
        "model_snapshot": "/local/qwen-snapshot",
        "asr_model_snapshot": "/local/whisper-snapshot",
        "asr_engine": "faster-whisper",
        "sample_rate_hz": 16_000,
        "mode_b_kickoff_text": "Buongiorno, codice 4829.",
        "status": "complete",
        "pretest_passed": True,
        "cache_complete": True,
        "all_audio_asr_passed": True,
        "clips": [
            record(f"scenario_{index}__turn_1_user.wav", f"scenario_{index}", "turn_1_user")
            for index in range(11)
        ],
        "mode_b_kickoff": record("mode_b_kickoff.wav", "mode_b", "kickoff"),
    }
    assert _manifest_is_complete(manifest)

    manifest["clips"].pop()
    assert not _manifest_is_complete(manifest)
    manifest["clips"].append(record("scenario_10__turn_1_user.wav", "scenario_10", "turn_1_user"))
    manifest["status"] = "generation_in_progress"
    assert not _manifest_is_complete(manifest)


def test_clip_validator_checks_audio_hash_transcript_and_finite_metrics(tmp_path):
    clip, audio_path, expected = _valid_audio_clip(tmp_path)

    assert _validate_cache_clip(clip, expected_text=expected, audio_path=audio_path) == []
    assert _validate_cache_clip(
        clip,
        expected_text=expected,
        audio_path=audio_path,
        independent_transcript=clip["asr_transcript"],
    ) == []

    clip["numeric_normalized_wer"] = math.nan
    errors = _validate_cache_clip(clip, expected_text=expected, audio_path=audio_path)
    assert any("numeric_normalized_wer" in error for error in errors)


def test_clip_validator_checks_scenario_and_step_identity(tmp_path):
    clip, audio_path, expected = _valid_audio_clip(tmp_path)

    errors = _validate_cache_clip(
        clip,
        expected_text=expected,
        audio_path=audio_path,
        expected_scenario_id="different_scenario",
        expected_step_id="turn_9_user",
    )

    assert any("scenario_id" in error for error in errors)
    assert any("step_id" in error for error in errors)


def test_output_directory_setup_rejects_symlink_before_manifest_writes(tmp_path):
    outside = tmp_path / "external"
    outside.mkdir()
    link = tmp_path / "cache-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted in this environment")

    with pytest.raises(ValueError, match="symlink"):
        _prepare_output_dir(link)
    assert not (outside / "scenario_audio_manifest.json").exists()


def test_clip_validator_rejects_manifest_metric_tampering(tmp_path):
    clip, audio_path, expected = _valid_audio_clip(tmp_path)
    clip["audio_metrics"]["rms"] *= 2.0
    clip["audio_metrics"]["crest_factor"] = math.nan

    errors = _validate_cache_clip(clip, expected_text=expected, audio_path=audio_path)

    assert any("audio_metrics.rms" in error for error in errors)
    assert any("audio_metrics.crest_factor" in error for error in errors)


def test_clip_validator_rejects_nan_raw_wer_and_manifest_rms(tmp_path):
    clip, audio_path, expected = _valid_audio_clip(tmp_path)
    clip["wer"] = math.nan
    clip["audio_metrics"]["rms"] = math.nan

    errors = _validate_cache_clip(clip, expected_text=expected, audio_path=audio_path)

    assert any("raw WER" in error for error in errors)
    assert any("manifest RMS" in error for error in errors)


def test_clip_validator_rejects_changed_audio_and_wrong_sample_rate(tmp_path):
    clip, audio_path, expected = _valid_audio_clip(tmp_path)
    audio_path.write_bytes(audio_path.read_bytes() + b"tampered")

    errors = _validate_cache_clip(clip, expected_text=expected, audio_path=audio_path)
    assert any("SHA-256" in error for error in errors)

    clip, audio_path, expected = _valid_audio_clip(tmp_path / "second")
    clip["sample_rate_hz"] = 8_000
    errors = _validate_cache_clip(clip, expected_text=expected, audio_path=audio_path)
    assert any("sample rate" in error for error in errors)


def test_english_text_and_short_numbers_are_not_rewritten():
    text = "Can you confirm code 4829 at 20:30?"

    assert _spoken_text_for_tts(text, "English") == text
    assert _spoken_text_for_tts("Siamo in quattro alle 20 e trenta.", "Italian") == "Siamo in quattro alle 20 e trenta."
