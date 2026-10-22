from __future__ import annotations

import json
from pathlib import Path

from tools.derive_prepared_timestamp_action_labels import (
    apply_native1540_pcm_temporal_targets,
    derive_prepared_dataset,
)
from training.duplex_qlora.stage2 import _causal_action_targets


def _row(conversation_id: str, split: str, group: str) -> dict:
    def segment(segment_id: str, role: str, channel: int, start: int, end: int, state: str) -> dict:
        return {
            "segment_id": segment_id,
            "role": role,
            "channel_index": channel,
            "start_ms": start,
            "end_ms": end,
            "audio_channel_verified": True,
            "time_mask": True,
            "native_turn_state": state,
        }

    return {
        "conversation_id": conversation_id,
        "source_group_id": group,
        "source": "DuplexConv",
        "split": split,
        "duration_ms": 640,
        "sample_rate_hz": 16000,
        "channels": 2,
        "audio_path": f"{conversation_id}.wav",
        "channel_map": {"0": "assistant", "1": "user"},
        "segments": [
            segment(f"{conversation_id}-user", "user", 1, 80, 220, "complete"),
            segment(f"{conversation_id}-assistant", "assistant", 0, 300, 380, "complete"),
        ],
        "frames": [{
            "frame_id": 3,
            "target_action": "SPEAK",
            "action_mask": True,
            "target_action_source": "timestamp_derived_assistant_onset_after_peer_termination",
            "target_action_evidence": {"local_segment_id": f"{conversation_id}-assistant"},
        }],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_endpoint_supplement_contains_only_exact_missing_frame_keys(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [_row("train-one", "train", "train-group")])
    _write_jsonl(prepared / "validation.jsonl", [_row("val-one", "validation", "val-group")])

    features = tmp_path / "old-features"
    features.mkdir()
    (features / "train.json").write_text(
        json.dumps([{"conversation_id": "train-one", "frame_id": 2, "action_mask": True}]),
        encoding="utf-8",
    )
    (features / "validation.json").write_text("[]", encoding="utf-8")

    derived = tmp_path / "derived"
    supplement = tmp_path / "supplement"
    report = derive_prepared_dataset(
        prepared,
        derived,
        supplement_output_dir=supplement,
        existing_feature_dir=features,
    )

    train_row = json.loads((derived / "train.jsonl").read_text(encoding="utf-8"))
    train_frames = {frame["frame_id"]: frame for frame in train_row["frames"]}
    assert train_frames[2]["target_action"] == "SPEAK"
    assert train_frames[2]["target_action_source"] == "timestamp_derived_last_vocal_complete_user_turn"
    # The later own onset is retained as a trace row but excluded from loss.
    assert train_frames[3]["action_mask"] is False

    assert report["timestamp_speak_endpoint_targets_by_split"]["train"]["feature_rows_missing_exactly"] == 0
    assert report["timestamp_speak_endpoint_targets_by_split"]["validation"]["feature_rows_missing_exactly"] == 1
    val_supplement = [
        json.loads(line)
        for line in (supplement / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(val_supplement) == 1
    assert val_supplement[0]["conversation_id"] == "val-one"
    assert [frame["frame_id"] for frame in val_supplement[0]["frames"]] == [2]
    assert val_supplement[0]["frames"][0]["target_action"] == "SPEAK"
    assert val_supplement[0]["timestamp_speak_endpoint_supplement_only"] is True
    assert (supplement / "preparation_manifest.json").is_file()


def _temporal_real_row(*, segments: list[dict], frames: list[dict] | None = None) -> dict:
    return {
        "conversation_id": "temporal-row",
        "source_group_id": "temporal-group",
        "source": "SmoothConv",
        "split": "train",
        "duration_ms": 1200,
        "sample_rate_hz": 16000,
        "channels": 2,
        "audio_path": "not-read-by-pure-view-test.wav",
        "channel_map": {"0": "assistant", "1": "user"},
        "segments": segments,
        "frames": frames or [],
    }


def _verified_segment(segment_id: str, role: str, start: float, end: float, state: str) -> dict:
    return {
        "segment_id": segment_id,
        "role": role,
        "perspective_role": role,
        "channel_index": 0 if role == "assistant" else 1,
        "start_ms": start,
        "end_ms": end,
        "native_turn_state": state,
        "audio_channel_verified": True,
        "time_mask": True,
    }


def test_pcm_temporal_speak_uses_trailing_voice_and_masks_late_source_edge() -> None:
    original = {
        "conversation_id": "v5-temporal-row",
        "source_group_id": "temporal-group",
        "source": "authored_synthetic_audio_candidate",
        "profile": "audio_grounded_v5",
        "split": "train",
        "duration_ms": 1200,
        "sample_rate_hz": 16000,
        "channel_map": {"0": "assistant", "1": "user"},
        "utterances": [
            # Deliberately misleading script role: speaker is the timeline role.
            {"utterance_id": "user-1", "speaker": "user", "role": "assistant_start_interruptible",
             "start_sample": 0, "end_sample": 5120},
            {"utterance_id": "assistant-1", "speaker": "assistant", "role": "assistant_start_interruptible",
             "start_sample": 12800, "end_sample": 14080, "target_action": "SPEAK"},
        ],
        "event_schedule": [{
            "action": "SPEAK", "context_utterance_id": "user-1", "response_utterance_id": "assistant-1",
        }],
        "frame_targets": [
            {
                "frame_id": frame_id,
                "input_frame_id": frame_id,
                "target_frame_id": frame_id + 1,
                "input_audio_start_sample": frame_id * 1280,
                "input_audio_end_sample": (frame_id + 1) * 1280,
                "target_action_start_sample": (frame_id + 1) * 1280,
                "target_action_end_sample": (frame_id + 2) * 1280,
                "action": "SPEAK" if frame_id in (9, 10) else "LISTEN",
                "target_action_mask": True,
                "input_text_visible": False,
                "future_audio_excluded": True,
            }
            for frame_id in range(15)
        ],
    }
    rms = [0.01, 0.01, 0.01, 0.0, 0.02, 0.0] + [0.0] * 9

    view, report = apply_native1540_pcm_temporal_targets(original, rms)
    frames = {frame["input_frame_id"]: frame for frame in view["frame_targets"]}

    assert frames[4]["action"] == "SPEAK"  # voice in the 160 ms allowed tail
    assert frames[5]["action"] == "SPEAK"  # first quiet tick after the last voice
    assert frames[9]["target_action_mask"] is False
    assert frames[9]["action_masked_reason"] == "raw_speak_outside_pcm_endpoint_window"
    assert frames[10]["action"] == "CONTINUE"  # held action while own audio is active
    assert frames[9]["native1540_temporal_view_original"]["action"] == "SPEAK"
    assert original["frame_targets"][9]["action"] == "SPEAK"  # source row remains untouched
    assert report["speak_events_qualified"] == 1


def test_pcm_temporal_view_does_not_treat_a_within_turn_pause_as_speak() -> None:
    row = _temporal_real_row(
        segments=[_verified_segment("user-1", "user", 0, 640, "incomplete")],
    )
    rms = [0.02, 0.02, 0.0, 0.0, 0.02, 0.02] + [0.0] * 9

    view, report = apply_native1540_pcm_temporal_targets(row, rms)

    assert not any(frame.get("target_action") == "SPEAK" and frame.get("action_mask") is True for frame in view["frames"])
    assert report["speak_events_qualified"] == 0

    # A raw candidate that the preparer already rejected is not promoted just
    # because its nearby role intervals look like a complete turn.
    masked_seed = _temporal_real_row(
        segments=[
            _verified_segment("user-1", "user", 0, 640, "complete"),
            _verified_segment("assistant-1", "assistant", 800, 880, "complete"),
        ],
        frames=[{
            "frame_id": 4,
            "target_action": "SPEAK",
            "action_mask": False,
            "target_action_source": "timestamp_derived_assistant_onset_after_peer_termination",
            "target_action_evidence": {"peer_segment_id": "user-1", "local_segment_id": "assistant-1"},
            "action_masked_reason": "assistant_onset_without_qualified_user_endpoint",
        }],
    )
    masked_view, _ = apply_native1540_pcm_temporal_targets(masked_seed, rms)
    assert masked_view["frames"][0]["target_action"] == "SPEAK"
    assert masked_view["frames"][0]["action_mask"] is False
    assert masked_view["frames"][0]["target_action_source"] == "timestamp_derived_assistant_onset_after_peer_termination"


def test_pcm_temporal_yield_starts_at_voice_and_protects_reviewed_conflict() -> None:
    row = _temporal_real_row(
        segments=[
            _verified_segment("assistant-1", "assistant", 0, 500, "incomplete"),
            _verified_segment("user-1", "user", 200, 800, "complete"),
        ],
        frames=[
            {
                "frame_id": 2,
                "target_action": "YIELD",
                "action_mask": True,
                "target_action_source": "timestamp_derived_peer_onset_confirmed_local_stop",
                "target_action_evidence": {
                    "local_segment_id": "assistant-1",
                    "peer_segment_id": "user-1",
                    "peer_onset_ms": 200,
                    "local_stop_ms": 500,
                },
            },
            {
                "frame_id": 3,
                "target_action": "CONTINUE",
                "action_mask": True,
                "target_action_source": "timestamp_derived_local_speech_active",
            },
            {
                "frame_id": 4,
                "target_action": "BACKCHANNEL",
                "action_mask": True,
                "target_action_source": "explicit_source_action_reviewed",
            },
        ],
    )
    rms = [0.0, 0.0, 0.0005, 0.01, 0.01, 0.01, 0.01] + [0.0] * 8

    view, report = apply_native1540_pcm_temporal_targets(row, rms)
    frames = {frame["frame_id"]: frame for frame in view["frames"]}

    assert frames[2]["target_action"] == "CONTINUE"
    assert frames[3]["target_action"] == "YIELD"
    assert frames[4]["target_action"] == "BACKCHANNEL"
    assert frames[4]["target_action_source"] == "explicit_source_action_reviewed"
    assert frames[5]["target_action"] == "YIELD"
    assert frames[6]["target_action"] == "YIELD"  # partial final own-active tick
    assert report["yield_events_qualified"] == 1
    assert report["reviewed_conflicts_protected"] == 1


def test_native1540_missing_feature_supplement_selects_only_exact_missing_targets(tmp_path: Path) -> None:
    temporal_view = tmp_path / "temporal-view"
    temporal_view.mkdir()
    row = _temporal_real_row(
        segments=[_verified_segment("user-1", "user", 0, 640, "complete")],
        frames=[
            {"frame_id": 1, "target_action": "LISTEN", "action_mask": True},
            {"frame_id": 2, "target_action": "SPEAK", "action_mask": True},
            {"frame_id": 9, "target_action": "SPEAK", "action_mask": False,
             "target_action_source": "native1540_temporal_late_speak_mask"},
        ],
    )
    row["native1540_temporal_supervision_view"] = {
        "schema": "native1540_pcm_temporal_supervision_v1"
    }
    _write_jsonl(temporal_view / "train.jsonl", [row])
    _write_jsonl(temporal_view / "validation.jsonl", [])
    (temporal_view / "preparation_manifest.json").write_text(
        json.dumps({"schema": "native1540_pcm_temporal_supervision_view_v1"}), encoding="utf-8"
    )

    feature_dir = tmp_path / "existing-features"
    feature_dir.mkdir()
    (feature_dir / "train.json").write_text(
        json.dumps([
            {"conversation_id": "temporal-row", "frame_id": 1},
            {"conversation_id": "temporal-row", "frame_id": 9},
        ]), encoding="utf-8"
    )
    (feature_dir / "validation.json").write_text("[]", encoding="utf-8")
    supplement = tmp_path / "missing-supplement"
    report = derive_prepared_dataset(
        temporal_view,
        supplement,
        existing_feature_dir=feature_dir,
        native1540_missing_feature_supplement=True,
    )

    copied = json.loads((supplement / "train.jsonl").read_text(encoding="utf-8"))
    target_map = {frame["frame_id"]: frame for frame in copied["frames"]}
    assert [target["frame_id"] for target in _causal_action_targets(copied)[0]] == [2]
    assert target_map[1]["action_mask"] is False
    assert target_map[1]["native1540_supplement_original_action_mask"] is True
    assert target_map[2]["action_mask"] is True
    assert target_map[9]["action_mask"] is False
    assert len(copied["segments"]) == len(row["segments"])
    assert [frame["action_mask"] for frame in row["frames"]] == [True, True, False]
    train = report["splits"]["train"]
    assert train["missing_exact_keys_selected"] == 1
    assert train["causal_input_frame_steps"] == 3
    assert train["obsolete_cached_keys_excluded_from_new_view"] == 1
