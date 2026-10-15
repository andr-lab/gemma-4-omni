from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_listener_stage2_v5 import prepare
from training.duplex_qlora.audio_grounded_v5 import validate_audio_grounded_v5_record
from training.duplex_qlora.schema import ACTIONS


def _record(root, split, ordinal):
    audio_path = root / f"{split}-{ordinal}.wav"
    samples = 1280 * 7
    audio = np.zeros((samples, 2), dtype=np.float32)
    audio[:, 0] = np.float32(0.01 + ordinal / 100)
    audio[:, 1] = np.float32(-0.01 - ordinal / 100)
    sf.write(audio_path, audio, 16000, subtype="PCM_16")
    frames = []
    for frame_id, action in enumerate(ACTIONS):
        start = frame_id * 1280
        end = start + 1280
        frames.append({
            "frame_id": frame_id,
            "input_frame_id": frame_id,
            "target_frame_id": frame_id + 1,
            "input_audio_start_sample": start,
            "input_audio_end_sample": end,
            "target_action_start_sample": end,
            "target_action_end_sample": end + 1280,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": action,
            "target_action_mask": True,
        })
    record = {
        "conversation_id": f"{split}-{ordinal}",
        "topic_group_id": f"group-{split}-{ordinal}",
        "split": split,
        "source": "authored_synthetic_audio_candidate",
        "profile": "audio_grounded_v5",
        "profile_version": "synthetic-it-t6-audio-grounded-v5-plan.1",
        "audio_path": str(audio_path),
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "sample_rate": 16000,
        "channels": 2,
        "channel_map": {"0": "user", "1": "assistant"},
        "duration_s": samples / 16000,
        "frame_ms": 80,
        "frame_targets": frames,
        "event_schedule": [{
            "action": "YIELD",
            "context_utterance_id": "assistant-0",
            "trigger_utterance_id": "user-0",
            "response_utterance_id": None,
        }],
        "utterances": [
            {"utterance_id": "user-0", "speaker": "user"},
            {"utterance_id": "assistant-0", "speaker": "assistant"},
        ],
    }
    return record


def test_v5_contract_uses_pcm_timeline_masks_and_event_links_without_version_gate(tmp_path):
    record = _record(tmp_path, "train", 0)
    record["source"] = "candidate-renamed"
    record["profile_version"] = "future-version"
    record["frame_targets"][3]["target_action_mask"] = False

    normalized = validate_audio_grounded_v5_record(record)

    assert normalized["sample_rate_hz"] == 16000
    assert normalized["source_group_id"] == "group-train-0"
    assert normalized["action_counts_by_class"]["YIELD"] == 0
    assert normalized["frame_targets"][3]["target_action_mask"] is False
    assert normalized["source"] == "candidate-renamed"


def test_v5_materializer_rejects_cross_split_group_leakage_and_reports_counts(tmp_path):
    records = [
        _record(tmp_path, "train", 0),
        _record(tmp_path, "validation", 1),
        _record(tmp_path, "test", 2),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    prepared = tmp_path / "prepared"
    report = prepare(manifest, prepared)

    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["absent_classes_by_split"] == {split: [] for split in ("train", "validation", "test")}
    assert len((prepared / "test.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    leaked = [dict(row) for row in records]
    leaked[1]["topic_group_id"] = leaked[0]["topic_group_id"]
    bad_manifest = tmp_path / "leaked.jsonl"
    bad_manifest.write_text("\n".join(json.dumps(row) for row in leaked) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source groups leak"):
        prepare(bad_manifest, tmp_path / "rejected")


def test_v5_materializer_resolves_repo_relative_audio_from_invocation_root(tmp_path, monkeypatch):
    project_root = tmp_path
    audio_root = project_root / "data" / "audio"
    audio_root.mkdir(parents=True)
    record = _record(audio_root, "train", 0)
    record["audio_path"] = (audio_root / "train-0.wav").relative_to(project_root).as_posix()
    manifest_root = project_root / "data" / "manifests"
    manifest_root.mkdir()
    manifest = manifest_root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.chdir(project_root)

    report = prepare(manifest, project_root / "prepared")

    assert report["split_counts"] == {"train": 1, "validation": 0, "test": 0}
