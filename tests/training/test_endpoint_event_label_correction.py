from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.duplex_qlora.event_label_correction import (
    assistant_active_in_previous_tick,
    event_run_offset,
    relabel_event_action,
)


def test_event_run_offset_is_scoped_to_same_action_and_contiguous_ticks():
    frames = [
        {"frame_id": 8, "target_action": "REPLAN", "action_mask": True},
        {"frame_id": 9, "target_action": "REPLAN", "action_mask": True},
        {"frame_id": 10, "target_action": "REPLAN", "action_mask": True},
        {"frame_id": 12, "target_action": "REPLAN", "action_mask": True},
    ]

    assert event_run_offset(frames, frame_id=8, action="REPLAN") == 0
    assert event_run_offset(frames, frame_id=9, action="REPLAN") == 1
    assert event_run_offset(frames, frame_id=10, action="REPLAN") == 2
    assert event_run_offset(frames, frame_id=12, action="REPLAN") == 0
    assert event_run_offset(frames, frame_id=11, action="REPLAN") is None
    assert event_run_offset(
        frames,
        frame_id=10,
        action="REPLAN",
        segment_ids={8: "utterance-a", 9: "utterance-a", 10: "utterance-b"},
    ) == 0


def test_previous_assistant_audio_uses_only_the_preceding_tick():
    segments = [
        {
            "role": "assistant",
            "channel_index": 1,
            "start_ms": 80,
            "end_ms": 240,
            "time_mask": True,
        }
    ]
    channel_map = {"0": "user", "1": "assistant"}

    assert assistant_active_in_previous_tick(
        frame_start_ms=240, tick_ms=80, segments=segments, channel_map=channel_map
    ) is True
    assert assistant_active_in_previous_tick(
        frame_start_ms=320, tick_ms=80, segments=segments, channel_map=channel_map
    ) is False
    segments.append(
        {
            "role": "assistant",
            "channel_index": 1,
            "start_ms": 800,
            "end_ms": 960,
            "time_mask": False,
        }
    )
    assert assistant_active_in_previous_tick(
        frame_start_ms=320, tick_ms=80, segments=segments, channel_map=channel_map
    ) is False


def test_backchannel_and_replan_interiors_become_continue_only_with_prior_audio():
    assert relabel_event_action("BACKCHANNEL", offset_ticks=0, assistant_prev_active=False) == (
        "BACKCHANNEL",
        True,
        "event_onset",
    )
    assert relabel_event_action("BACKCHANNEL", offset_ticks=2, assistant_prev_active=True) == (
        "CONTINUE",
        True,
        "interior_assistant_active",
    )
    assert relabel_event_action("REPLAN", offset_ticks=1, assistant_prev_active=False) == (
        None,
        False,
        "interior_without_prior_assistant_audio",
    )


def test_yield_interiors_become_listen_only_without_prior_assistant_audio():
    assert relabel_event_action("YIELD", offset_ticks=0, assistant_prev_active=True) == (
        "YIELD",
        True,
        "event_onset",
    )
    assert relabel_event_action("YIELD", offset_ticks=1, assistant_prev_active=False) == (
        "LISTEN",
        True,
        "interior_assistant_inactive",
    )
    assert relabel_event_action("YIELD", offset_ticks=1, assistant_prev_active=True) == (
        None,
        False,
        "interior_assistant_active",
    )


def test_missing_source_timing_fails_closed():
    assert relabel_event_action("BACKCHANNEL", offset_ticks=None, assistant_prev_active=None) == (
        None,
        False,
        "missing_event_or_assistant_timing",
    )


if __name__ == "__main__":
    test_functions = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test_function in test_functions:
        test_function()
        print(f"PASS {test_function.__name__}")
    print(f"{len(test_functions)} focused assertions passed")
