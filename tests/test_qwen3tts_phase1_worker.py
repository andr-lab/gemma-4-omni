import os

import numpy as np
import pytest
import soundfile as sf

from tools.generate_scenario_audio_qwen3tts_phase1_worker import (
    _open_safe_output_dir_fd,
    _safe_output_dir,
    _safe_output_path,
    _write_pcm_atomically,
    _to_mono_16khz_pcm,
    _validate_jobs,
)


def test_validate_jobs_preserves_valid_text_and_language():
    jobs = [{"filename": "scenario__turn_1.wav", "text": "Ciao.", "language": "Italian"}]

    assert _validate_jobs(jobs) == jobs


@pytest.mark.parametrize("filename", ["../escape.wav", "nested/file.wav", r"nested\file.wav", "not-a-wave.txt"])
def test_validate_jobs_rejects_unsafe_output_names(filename):
    jobs = [{"filename": filename, "text": "Ciao.", "language": "Italian"}]

    with pytest.raises(ValueError, match="unsafe or invalid"):
        _validate_jobs(jobs)


def test_validate_jobs_rejects_duplicate_output_names():
    job = {"filename": "same.wav", "text": "Ciao.", "language": "Italian"}

    with pytest.raises(ValueError, match="duplicate"):
        _validate_jobs([job, job])


def test_wave_conversion_rejects_invalid_source_sample_rate():
    with pytest.raises(ValueError, match="Invalid source sample rate"):
        _to_mono_16khz_pcm(np.zeros(8, dtype=np.float32), source_rate=0)


def test_wave_conversion_rejects_overflow_created_by_resampling():
    source = np.full(240, np.finfo(np.float32).max, dtype=np.float32)

    with pytest.raises(RuntimeError, match="non-finite values after resampling"):
        _to_mono_16khz_pcm(source, source_rate=24_000)


def test_safe_output_path_rejects_preexisting_symlink(tmp_path):
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"do not modify")
    link = output_dir / "escape.wav"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted in this environment")

    with pytest.raises(ValueError, match="symlink"):
        _safe_output_path(output_dir, "escape.wav")
    assert outside.read_bytes() == b"do not modify"


def test_atomic_writer_creates_readable_mono_pcm16_wave(tmp_path):
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    target = _safe_output_path(output_dir, "audio.wav")
    pcm = _to_mono_16khz_pcm(np.sin(np.linspace(0, 8, 240, dtype=np.float32)), source_rate=24_000)

    _write_pcm_atomically(target, pcm)

    saved, sample_rate = sf.read(target, dtype="float32")
    assert sample_rate == 16_000
    assert saved.ndim == 1 and np.isfinite(saved).all()
    assert not list(output_dir.glob(".audio_*.tmp.wav"))


def test_safe_output_directory_rejects_symlink_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "cache-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted in this environment")

    with pytest.raises(ValueError, match="output directory contains a symlink"):
        _safe_output_dir(link)


@pytest.mark.skipif(os.name != "posix", reason="directory-fd anchoring is POSIX-specific")
def test_atomic_writer_stays_inside_open_directory_after_path_is_replaced(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    moved_cache = tmp_path / "moved-cache"
    output_dir_fd = _open_safe_output_dir_fd(cache)
    if output_dir_fd is None:
        pytest.skip("openat-style directory descriptors are unavailable")
    target = _safe_output_path(cache, "race.wav")
    pcm = _to_mono_16khz_pcm(np.sin(np.linspace(0, 8, 240, dtype=np.float32)), source_rate=24_000)
    try:
        cache.rename(moved_cache)
        cache.symlink_to(outside, target_is_directory=True)
        _write_pcm_atomically(target, pcm, output_dir_fd=output_dir_fd)
    finally:
        os.close(output_dir_fd)

    saved, sample_rate = sf.read(moved_cache / "race.wav", dtype="float32")
    assert sample_rate == 16_000 and np.isfinite(saved).all()
    assert not (outside / "race.wav").exists()


def test_wave_conversion_resamples_mono_and_caps_peak_without_clipping():
    source = np.full(240, 2.0, dtype=np.float32)

    pcm = _to_mono_16khz_pcm(source, source_rate=24_000)

    assert pcm.shape == (160,)
    assert np.isfinite(pcm).all()
    assert np.max(np.abs(pcm)) <= 0.95001
