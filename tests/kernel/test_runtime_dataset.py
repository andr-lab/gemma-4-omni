from __future__ import annotations

import pytest

from tools.kernel_gates.runtime_dataset import aligned_validation_start, attested_validation_start


def test_source_attestation_covers_runtime_split_dependency_and_regression_test() -> None:
    from tools.kernel_gates.source_attestation import DEFAULT_FILES

    assert "tools/extract_runtime_cache.py" in DEFAULT_FILES
    assert "tools/kernel_gates/representation_parity.py" in DEFAULT_FILES
    assert "tools/kernel_gates/runtime_dataset.py" in DEFAULT_FILES
    assert "tests/kernel/test_runtime_dataset.py" in DEFAULT_FILES


def test_validation_start_aligns_to_a_complete_text_window_after_segment_boundary() -> None:
    start = aligned_validation_start(
        total_frames=1_834_067,
        validation_fraction=0.10,
        segment_ends=(1_650_667,),
        window_frames=64,
    )

    assert start == 1_650_688
    assert start % 64 == 0
    assert start >= 1_650_667


def test_attested_validation_start_rejects_legacy_mid_window_cache_offset() -> None:
    kwargs = {
        "total_frames": 1_834_067,
        "validation_fraction": 0.10,
        "segment_ends": (1_650_667,),
        "window_frames": 64,
    }
    assert attested_validation_start(cache_start_frame=1_650_688, **kwargs) == 1_650_688
    with pytest.raises(ValueError, match="non coincide"):
        attested_validation_start(cache_start_frame=1_650_667, **kwargs)


def test_validation_start_uses_nominal_boundary_when_no_later_segment_exists() -> None:
    assert aligned_validation_start(
        total_frames=1_000,
        validation_fraction=0.10,
        segment_ends=(),
        window_frames=64,
    ) == 960


@pytest.mark.parametrize(
    ("total_frames", "validation_fraction", "window_frames"),
    ((0, 0.10, 64), (100, 0.0, 64), (100, 1.0, 64), (100, 0.10, 0)),
)
def test_validation_start_rejects_invalid_ranges(
    total_frames: int,
    validation_fraction: float,
    window_frames: int,
) -> None:
    with pytest.raises(ValueError):
        aligned_validation_start(
            total_frames=total_frames,
            validation_fraction=validation_fraction,
            segment_ends=(),
            window_frames=window_frames,
        )
