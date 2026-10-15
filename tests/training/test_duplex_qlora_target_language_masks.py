from __future__ import annotations

import json

from training.duplex_qlora.prepare import prepare_splits


def _read_split(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_masks_only_bad_synthetic_assistant_target_and_preserves_user_context(tmp_path):
    row = {
        "conversation_id": "synthetic-it-t6-00000003",
        "source_group_id": "lesson_delivery",
        "source": "synthetic_it_t6_v4",
        "synthetic": True,
        "template_family_id": "lesson_delivery",
        "split": "train",
        "duration_ms": 1000,
        "segments": [
            {"speaker": "user", "role": "user", "script_role": "user_correction",
             "text": "dobbiamo partire da il consumo", "start_ms": 0, "end_ms": 300,
             "text_mask": True, "time_mask": True},
            {"speaker": "assistant", "role": "assistant", "script_role": "assistant_replan",
             "text": "iniziamo da il consumo", "start_ms": 400, "end_ms": 800,
             "text_mask": True, "time_mask": True},
            {"speaker": "assistant", "role": "assistant", "script_role": "assistant_start",
             "text": "iniziamo verificando il consumo", "start_ms": 810, "end_ms": 950,
             "text_mask": True, "time_mask": True},
        ],
        "frames": [],
    }

    report = prepare_splits([row], tmp_path, seed=1)
    prepared = _read_split(tmp_path / "train.jsonl")[0]
    segments = prepared["segments"]

    assert segments[0]["text_mask"] is True
    assert segments[0]["target_mask"] is True
    assert segments[1]["text_mask"] is True
    assert segments[1]["target_mask"] is False
    assert segments[1]["target_mask_reason"] == "uncontracted_preposition_article"
    assert segments[2]["target_mask"] is True
    assert report["synthetic_target_text_masks"] == {"uncontracted_preposition_article": 1}


def test_prepare_masks_each_bad_validation_target_but_keeps_good_target(tmp_path):
    row = {
        "conversation_id": "synthetic-it-t6-00000003",
        "source_group_id": "lesson_delivery",
        "source": "synthetic_it_t6_v4",
        "synthetic": True,
        "template_family_id": "lesson_delivery",
        "split": "validation",
        "duration_ms": 1000,
        "segments": [
            {"speaker": "assistant", "role": "assistant", "script_role": "assistant_replan",
             "text": "dobbiamo partire da il consumo d'acqua", "start_ms": 0, "end_ms": 300,
             "text_mask": True, "time_mask": True},
            {"speaker": "assistant", "role": "assistant", "script_role": "assistant_replan",
             "text": "iniziamo da la scelta degli orari", "start_ms": 400, "end_ms": 700,
             "text_mask": True, "time_mask": True},
            {"speaker": "assistant", "role": "assistant", "script_role": "assistant_start",
             "text": "iniziamo verificando il consumo d'acqua", "start_ms": 710, "end_ms": 950,
             "text_mask": True, "time_mask": True},
        ],
        "frames": [],
    }

    report = prepare_splits([row], tmp_path, seed=1)
    prepared = _read_split(tmp_path / "validation.jsonl")[0]

    assert [segment["target_mask"] for segment in prepared["segments"]] == [False, False, True]
    assert [segment.get("target_mask_reason") for segment in prepared["segments"]] == [
        "uncontracted_preposition_article", "uncontracted_preposition_article", None
    ]
    assert report["synthetic_target_text_masks"] == {"uncontracted_preposition_article": 2}
