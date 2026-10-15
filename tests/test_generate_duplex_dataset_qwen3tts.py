from __future__ import annotations

import numpy as np
import pytest

from tools.generate_duplex_dataset_qwen3tts import (
    ACTIONS,
    FRAME_SAMPLES,
    SCENARIOS,
    build_frame_annotations,
    compose_stereo_timeline,
    ensure_noncanonical_output_path,
    _resolve_timeline,
    validate_scenarios,
)


def test_scenario_library_covers_all_canonical_actions_and_required_user_events() -> None:
    action_coverage, event_coverage = validate_scenarios(SCENARIOS)

    assert action_coverage == set(ACTIONS)
    assert {
        "backchannel",
        "interrupt",
        "correction",
        "false_start",
        "yield",
        "semantic_replan",
        "assistant_backchannel",
        "floor_holding",
        "overlap",
        "silence",
    } <= event_coverage
    assert len(SCENARIOS) >= 5


def test_frame_annotations_are_causal_80ms_and_preserve_overlap() -> None:
    segments = [
        {
            "segment_id": "user-0",
            "speaker_id": "user",
            "text": "Sto spiegando.",
            "start_sample": 0,
            "end_sample": 2 * FRAME_SAMPLES,
        },
        {
            "segment_id": "assistant-0",
            "speaker_id": "assistant",
            "text": "Sì, chiaro.",
            "start_sample": FRAME_SAMPLES,
            "end_sample": 3 * FRAME_SAMPLES,
        },
    ]
    events = [
        {
            "event_id": "event-0",
            "event_type": "assistant_backchannel",
            "target_action": "BACKCHANNEL",
            "start_sample": FRAME_SAMPLES,
            "end_sample": 2 * FRAME_SAMPLES,
        }
    ]

    frames = build_frame_annotations("dialogue-1", 3 * FRAME_SAMPLES, segments, events)

    assert len(frames) == 3
    assert [frame["timestamp_ms"] for frame in frames] == [0, 80, 160]
    assert frames[0]["user_speaking"] is True
    assert frames[0]["assistant_speaking"] is False
    assert frames[0]["target_action"] == "LISTEN"
    assert frames[1]["user_speaking"] is True
    assert frames[1]["assistant_speaking"] is True
    assert frames[1]["target_action"] == "BACKCHANNEL"
    assert frames[1]["transcript_spans"] == ["user-0", "assistant-0"]
    assert all(frame["available_until_frame"] == frame["frame_index"] for frame in frames)


def test_stereo_composition_keeps_speakers_on_separate_channels_and_exact_offsets() -> None:
    clips = {
        "user-0": np.full(2 * FRAME_SAMPLES, 0.25, dtype=np.float32),
        "assistant-0": np.full(2 * FRAME_SAMPLES, -0.5, dtype=np.float32),
    }
    segments = [
        {"segment_id": "user-0", "speaker_id": "user", "start_sample": 0, "end_sample": 2 * FRAME_SAMPLES},
        {"segment_id": "assistant-0", "speaker_id": "assistant", "start_sample": FRAME_SAMPLES, "end_sample": 3 * FRAME_SAMPLES},
    ]

    stereo = compose_stereo_timeline(clips, segments)

    assert stereo.shape == (3 * FRAME_SAMPLES, 2)
    np.testing.assert_array_equal(stereo[: 2 * FRAME_SAMPLES, 0], clips["user-0"])
    np.testing.assert_array_equal(stereo[FRAME_SAMPLES :, 1], clips["assistant-0"])
    assert np.all(stereo[2 * FRAME_SAMPLES :, 0] == 0.0)
    assert np.all(stereo[:FRAME_SAMPLES, 1] == 0.0)


def test_stereo_composition_rejects_nonfinite_or_clipping_input() -> None:
    segments = [{"segment_id": "user-0", "speaker_id": "user", "start_sample": 0, "end_sample": 2}]

    with pytest.raises(ValueError, match="finite"):
        compose_stereo_timeline({"user-0": np.array([0.1, np.nan], dtype=np.float32)}, segments)
    with pytest.raises(ValueError, match="peak"):
        compose_stereo_timeline({"user-0": np.array([0.1, 1.01], dtype=np.float32)}, segments)


def test_interrupt_truncation_snaps_to_word_boundary_and_preserves_one_frame_overlap() -> None:
    scenario = {
        "scenario_id": "interrupt-test",
        "segments": [
            {"segment_id": "assistant", "speaker_id": "assistant", "at_s": 0.0, "text": "ciao"},
            {
                "segment_id": "user",
                "speaker_id": "user",
                "during": "assistant",
                "offset_s": 0.75,
                "align_to_word_boundary": True,
                "text": "aspetta",
            },
        ],
        "events": [
            {
                "event_type": "interrupt",
                "target_action": "YIELD",
                "segment_id": "user",
                "truncates_segment_id": "assistant",
                "truncate_delay_s": 0.08,
            }
        ],
    }
    clip_info = {
        "assistant": {
            "duration_samples": 2 * 16_000,
            "words": [
                {"text": "ciao", "start_s": 0.1, "end_s": 0.5},
                {"text": "ancora", "start_s": 0.9, "end_s": 1.2},
            ],
        },
        "user": {"duration_samples": 4_000, "words": []},
    }

    segments, events = _resolve_timeline(scenario, clip_info)

    assistant, user = segments
    assert user["start_sample"] == round(0.9 * 16_000)
    assert assistant["end_sample"] == round(0.98 * 16_000)
    assert assistant["truncated"] is True
    overlap = next(event for event in events if event["event_type"] == "overlap")
    assert overlap["end_sample"] - overlap["start_sample"] == FRAME_SAMPLES


def test_short_event_limit_truncates_the_audible_segment() -> None:
    scenario = {
        "scenario_id": "short-event-test",
        "segments": [
            {"segment_id": "backchannel", "speaker_id": "user", "at_s": 0.0, "text": "sì"}
        ],
        "events": [
            {
                "event_type": "backchannel",
                "target_action": "CONTINUE",
                "segment_id": "backchannel",
                "max_duration_s": 1.0,
            }
        ],
    }

    segments, _events = _resolve_timeline(scenario, {"backchannel": {"duration_samples": 20_000, "words": []}})

    assert segments[0]["end_sample"] == 16_000
    assert segments[0]["truncated"] is True


def test_synthetic_output_cannot_be_written_into_canonical_benchmark_tree(tmp_path) -> None:
    canonical = tmp_path / "tools" / "training_gates" / "benchmarks" / "benchmark_data" / "DuplexIT-Eval-v1"
    allowed = tmp_path / "artifacts" / "duplex_qwen3tts_synthetic_candidate"

    ensure_noncanonical_output_path(allowed, repository_root=tmp_path)
    with pytest.raises(ValueError, match="canonical benchmark"):
        ensure_noncanonical_output_path(canonical, repository_root=tmp_path)
