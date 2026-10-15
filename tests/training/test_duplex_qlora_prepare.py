from __future__ import annotations

import json

from training.duplex_qlora.prepare import prepare_splits


def test_prepare_preserves_synthetic_family_split_and_keeps_aux_group_with_it(tmp_path):
    rows = [
        {
            "conversation_id": "syn-1",
            "source_group_id": "family-1",
            "source": "synthetic_it_v3",
            "split": "validation",
            "frames": [],
        },
        {
            "conversation_id": "aux-1",
            "source_group_id": "family-1",
            "source": "SmoothConv",
            "frames": [],
        },
        {
            "conversation_id": "syn-2",
            "source_group_id": "family-2",
            "source": "synthetic_it_v3",
            "split": "train",
            "frames": [],
        },
    ]

    report = prepare_splits(rows, tmp_path, seed=1)

    validation = [json.loads(line) for line in (tmp_path / "validation.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert {row["conversation_id"] for row in validation} == {"syn-1", "aux-1"}
    assert {row["conversation_id"] for row in train} == {"syn-2"}
    assert report["preserved_source_splits"] == 2
    assert report["quarantined_group_count"] == 0


def test_prepare_quarantines_connected_group_with_conflicting_official_splits(tmp_path):
    rows = [
        {"conversation_id": "a", "source_group_id": "same", "source": "synthetic_it_v3", "split": "train", "frames": []},
        {"conversation_id": "b", "source_group_id": "same", "source": "synthetic_it_v3", "split": "test", "frames": []},
        {"conversation_id": "c", "source_group_id": "other", "source": "synthetic_it_v3", "split": "train", "frames": []},
    ]

    report = prepare_splits(rows, tmp_path)

    standard = [
        json.loads(line)
        for name in ("train", "validation", "test")
        for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()
    ]
    quarantined = [json.loads(line) for line in (tmp_path / "quarantine.jsonl").read_text().splitlines()]
    assert {row["conversation_id"] for row in standard} == {"c"}
    assert {row["conversation_id"] for row in quarantined} == {"a", "b"}
    assert report["quarantined_group_count"] == 1
    assert report["quarantined_conversation_count"] == 2
