from __future__ import annotations

import unittest

from training.duplex_qlora.endpoint_causal_data import derive_native_endpoint_labels


def _frame(frame_id: int, action: str | None = None) -> dict:
    return {
        "frame_id": frame_id,
        "start_ms": frame_id * 80,
        "end_ms": (frame_id + 1) * 80,
        "valid_ms": 80,
        "valid_samples": 3840,
        "time_mask": True,
        "audio_mask": False,
        "events": [],
        "target_action": action,
        "action_mask": action is not None,
    }


def _segment(role: str, start_ms: float, end_ms: float, state: str | None) -> dict:
    channel = 1 if role == "user" else 0
    return {
        "role": role,
        "perspective_role": role,
        "channel_index": channel,
        "audio_channel_verified": True,
        "time_mask": True,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "native_turn_state": state,
        "segment_id": f"{role}-{start_ms}",
    }


def _row(user_segments: list[dict], assistant_segments: list[dict] | None = None) -> dict:
    return {
        "conversation_id": "fixture",
        "source_group_id": "fixture-group",
        "source": "SmoothConv",
        "synthetic": False,
        "split": "train",
        "duration_ms": 4000,
        "sample_rate_hz": 48000,
        "tick_ms": 80,
        "channel_map": {"0": "assistant", "1": "user"},
        "segments": [*(user_segments or []), *(assistant_segments or [])],
        "frames": [_frame(12), _frame(13, "SPEAK"), _frame(14), _frame(15)],
    }


class EndpointCausalLabelTests(unittest.TestCase):
    def test_complete_user_turn_becomes_speak_at_first_tick_end_with_160ms_pause(self):
        row = _row([_segment("user", 200, 1000, "complete")])
        labeled, report = derive_native_endpoint_labels(row)
        by_id = {frame["frame_id"]: frame for frame in labeled["frames"]}
        self.assertEqual(by_id[13]["target_action"], "LISTEN")
        self.assertEqual(by_id[14]["target_action"], "SPEAK")
        self.assertGreaterEqual(by_id[14]["end_ms"] - 1000, 160)
        self.assertEqual(by_id[14]["assistant_reference_frame_id"], 13)
        self.assertEqual(report["speak_endpoints_created"], 1)
        self.assertEqual(report["speak_endpoint_rule"], "first_tick_end_at_or_after_user_end_plus_160ms")

    def test_incomplete_user_turn_never_becomes_speak_and_pause_remains_listen(self):
        row = _row([_segment("user", 200, 1000, "incomplete")])
        labeled, report = derive_native_endpoint_labels(row)
        by_id = {frame["frame_id"]: frame for frame in labeled["frames"]}
        self.assertNotIn("SPEAK", [frame["target_action"] for frame in labeled["frames"]])
        self.assertEqual(by_id[14]["target_action"], "LISTEN")
        self.assertEqual(report["incomplete_user_listen_frames"], 4)

    def test_user_backchannel_cannot_create_speak(self):
        row = _row([_segment("user", 200, 1000, "backchannel")])
        labeled, _ = derive_native_endpoint_labels(row)
        self.assertNotIn("SPEAK", [frame["target_action"] for frame in labeled["frames"]])

    def test_no_speak_when_another_user_is_active_at_target_boundary(self):
        row = _row([
            _segment("user", 200, 1000, "complete"),
            _segment("user", 1100, 1400, "incomplete"),
        ])
        labeled, report = derive_native_endpoint_labels(row)
        by_id = {frame["frame_id"]: frame for frame in labeled["frames"]}
        self.assertNotIn("SPEAK", [frame["target_action"] for frame in labeled["frames"]])
        self.assertEqual(by_id[14]["target_action"], "LISTEN")
        self.assertGreater(report["endpoint_rejected_user_active"], 0)

    def test_no_speak_when_assistant_audio_was_active_in_t_minus_1(self):
        row = _row(
            [_segment("user", 200, 1000, "complete")],
            [_segment("assistant", 1100, 1500, "complete")],
        )
        labeled, report = derive_native_endpoint_labels(row)
        self.assertNotIn("SPEAK", [frame["target_action"] for frame in labeled["frames"]])
        self.assertGreater(report["endpoint_rejected_assistant_t_minus_1_active"], 0)

    def test_backchannel_and_continue_are_kept_only_with_native_assistant_links(self):
        row = _row(
            [],
            [
                _segment("assistant", 960, 1030, "backchannel"),
                _segment("assistant", 1040, 2000, "complete"),
            ],
        )
        row["frames"] = [
            _frame(12, "BACKCHANNEL"),
            _frame(13, "CONTINUE"),
            _frame(14, "BACKCHANNEL"),
            _frame(15, "CONTINUE"),
        ]
        labeled, report = derive_native_endpoint_labels(row)
        by_id = {frame["frame_id"]: frame for frame in labeled["frames"]}
        self.assertEqual(by_id[12]["target_action"], "BACKCHANNEL")
        self.assertEqual(by_id[13]["target_action"], "CONTINUE")
        self.assertFalse(by_id[14]["action_mask"])
        self.assertEqual(by_id[15]["target_action"], "CONTINUE")
        self.assertEqual(report["backchannel_kept"], 1)
        self.assertEqual(report["continue_kept"], 2)


if __name__ == "__main__":
    unittest.main()
