from __future__ import annotations

import json
from pathlib import Path

from tools.prepare_stage2_candidate_e_yield import FRAME_SAMPLES, prepare_candidate_e


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _natural(conversation_id: str, group: str, split: str) -> dict:
    return {
        "conversation_id": conversation_id,
        "source_group_id": group,
        "source": "SmoothConv",
        "split": split,
        "synthetic": False,
        "audio_path": "/workspace/audio.wav",
        "channel_map": {"0": "user", "1": "assistant"},
        "frames": [
            {"frame_id": 0, "target_action": "LISTEN", "action_mask": True},
            {
                "frame_id": 1,
                "target_action": "BACKCHANNEL",
                "action_mask": False,
                "action_masked_reason": "natural_backchannel_without_verified_remote_overlap_at_source_onset",
            },
            {"frame_id": 2, "target_action": "SPEAK", "action_mask": True},
        ],
    }


def _v5_parent(conversation_id: str, group: str, split: str) -> dict:
    frames = []
    for input_id, action in ((1, "CONTINUE"), (3, "REPLAN")):
        frames.append({
            "frame_id": input_id,
            "input_frame_id": input_id,
            "target_frame_id": input_id + 1,
            "input_audio_start_sample": input_id * FRAME_SAMPLES,
            "input_audio_end_sample": (input_id + 1) * FRAME_SAMPLES,
            "target_action_start_sample": (input_id + 1) * FRAME_SAMPLES,
            "target_action_end_sample": (input_id + 2) * FRAME_SAMPLES,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": action,
            "target_action_mask": True,
        })
    return {
        "conversation_id": conversation_id,
        "source_group_id": group,
        "source": "authored_synthetic_audio_candidate",
        "profile": "audio_grounded_v5",
        "split": split,
        "synthetic": True,
        "audio_path": "/workspace/v5.wav",
        "frame_targets": frames,
        "event_schedule": [],
    }


def _yield_variant(parent: str, conversation_id: str, group: str, split: str) -> dict:
    correction = 1 * FRAME_SAMPLES
    cut = 5 * FRAME_SAMPLES
    frames = []
    for input_id in range(10):
        target_start = (input_id + 1) * FRAME_SAMPLES
        action = "YIELD" if input_id == 4 else "LISTEN"
        mask = input_id == 4 or input_id in (6, 8)
        # Speech correction before the cut is deliberately ambiguous and masked.
        if target_start < cut and (target_start + FRAME_SAMPLES) > correction:
            mask = False
        frames.append({
            "frame_id": input_id,
            "input_frame_id": input_id,
            "target_frame_id": input_id + 1,
            "input_audio_start_sample": input_id * FRAME_SAMPLES,
            "input_audio_end_sample": (input_id + 1) * FRAME_SAMPLES,
            "target_action_start_sample": target_start,
            "target_action_end_sample": target_start + FRAME_SAMPLES,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": action,
            "target_action_mask": mask,
        })
    return {
        "conversation_id": conversation_id,
        "parent_conversation_id": parent,
        "source": "authored_synthetic_audio_candidate",
        "profile": "audio_grounded_v5",
        "split": split,
        "synthetic": True,
        "audio_path": "/workspace/yield.wav",
        "yield_cut_delay_ms": 320,
        "frame_targets": frames,
        "event_schedule": [{
            "action": "YIELD",
            "start_sample": cut,
            "end_sample": cut + FRAME_SAMPLES,
            "correction_onset_sample": correction,
            "assistant_cut_sample": cut,
            "max_delay_ms": 320,
        }],
    }


def _feature_row(conversation: dict, frame_id: int, action: str, mask: bool = True) -> dict:
    return {
        "conversation_id": conversation["conversation_id"],
        "source_group_id": conversation["source_group_id"],
        "frame_id": frame_id,
        "input_frame_id": frame_id,
        "target_frame_id": frame_id + 1 if conversation.get("profile") == "audio_grounded_v5" else frame_id,
        "profile": conversation.get("profile"),
        "action": action if mask else None,
        "action_mask": mask,
        "action_masked_reason": "natural_backchannel_without_verified_remote_overlap_at_source_onset" if not mask else None,
    }


def test_candidate_e_materializer_preserves_c_selection_and_causal_yield_targets(tmp_path: Path):
    base = tmp_path / "base"
    cap = tmp_path / "cap"
    features = tmp_path / "candidate_c"
    natural_train = _natural("natural-train", "natural-group-train", "train")
    masked_only_train = _natural("masked-only-train", "masked-only-group-train", "train")
    masked_only_train["frames"] = [{
        "frame_id": 1,
        "target_action": "BACKCHANNEL",
        "action_mask": False,
        "action_masked_reason": "natural_backchannel_without_verified_remote_overlap_at_source_onset",
    }]
    natural_val = _natural("natural-val", "natural-group-val", "validation")
    parent_train = _v5_parent("v5-train", "v5-group-train", "train")
    parent_val = _v5_parent("v5-val", "v5-group-val", "validation")
    _write_jsonl(base / "train.jsonl", [natural_train, masked_only_train])
    _write_jsonl(base / "validation.jsonl", [natural_val])
    _write_jsonl(cap / "train.jsonl", [parent_train])
    _write_jsonl(cap / "validation.jsonl", [parent_val])
    train_features = [
        _feature_row(natural_train, 0, "LISTEN"),
        _feature_row(natural_train, 1, "BACKCHANNEL", mask=False),
        _feature_row(masked_only_train, 1, "BACKCHANNEL", mask=False),
        _feature_row(parent_train, 1, "CONTINUE"),
    ]
    val_features = [
        _feature_row(natural_val, 0, "LISTEN"),
        _feature_row(parent_val, 1, "CONTINUE"),
    ]
    (features / "train.json").parent.mkdir(parents=True)
    (features / "train.json").write_text(json.dumps(train_features), encoding="utf-8")
    (features / "validation.json").write_text(json.dumps(val_features), encoding="utf-8")
    remix = [
        _yield_variant("v5-train", f"v5-train-yield-{index}", "v5-group-train", "train")
        for index in range(96)
    ] + [
        _yield_variant("v5-val", f"v5-val-yield-{index}", "v5-group-val", "validation")
        for index in range(24)
    ]
    remix_path = tmp_path / "yield.jsonl"
    _write_jsonl(remix_path, remix)

    output = tmp_path / "candidate_e"
    smoke = tmp_path / "candidate_e_smoke"
    result = prepare_candidate_e(
        base_prepared_dir=base,
        traincap_prepared_dir=cap,
        candidate_c_feature_dir=features,
        yield_manifest=remix_path,
        output_dir=output,
        smoke_output_dir=smoke,
    )

    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    natural_out = next(row for row in train_rows if row["conversation_id"] == "natural-train")
    assert [(frame["frame_id"], frame["action_mask"]) for frame in natural_out["frames"]] == [
        (0, True), (1, False), (2, False)
    ]
    assert natural_out["frames"][1]["action_masked_reason"] == "natural_backchannel_without_verified_remote_overlap_at_source_onset"
    variant = next(row for row in train_rows if row["conversation_id"] == "v5-train-yield-0")
    selected = [frame for frame in variant["frame_targets"] if frame["target_action_mask"]]
    assert [(frame["input_frame_id"], frame["target_frame_id"], frame["action"]) for frame in selected] == [
        (4, 5, "YIELD"), (6, 7, "LISTEN"), (8, 9, "LISTEN")
    ]
    assert variant["source_group_id"] == "v5-group-train"
    assert all(row["conversation_id"] != "masked-only-train" for row in train_rows)
    assert result["candidate_c_selection"]["train"]["masked_only_conversations_not_embedded"] == 1
    assert result["splits"]["train"]["supervised_feature_targets"] == 1 + 1 + 96 * 3
    assert result["splits"]["train"]["causal_input_frame_steps_from_frame_zero"] == 1 + 2 + 96 * 9
    assert result["splits"]["validation"]["supervised_feature_targets"] == 1 + 1 + 24 * 3
    assert result["splits"]["validation"]["causal_input_frame_steps_from_frame_zero"] == 1 + 2 + 24 * 9
    smoke_rows = [json.loads(line) for line in (smoke / "train.jsonl").read_text().splitlines()]
    assert len(smoke_rows) == 1
    assert sum(frame["action_mask"] for frame in smoke_rows[0]["frames"]) == 1
