from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

from training.duplex_qlora.schema import split_conversations
from training.duplex_qlora.sources import (
    read_duplexconv,
    read_smoothconv,
    read_synthetic_v3_clean_asr,
    read_synthetic_v3_manifest,
    summarize_preparation_counts,
    _channel_rms,
)


def _write_stereo(path: Path, *, rate: int, duration_s: float, ranges: dict[int, list[tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(rate * duration_s)
    payload = bytearray(frames * 4)
    for index in range(frames):
        time_s = index / rate
        left = 5000 if any(start <= time_s < end for start, end in ranges.get(0, [])) else 0
        right = 5000 if any(start <= time_s < end for start, end in ranges.get(1, [])) else 0
        struct.pack_into("<hh", payload, index * 4, left, right)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_pcm16(path: Path, channels: list[list[int]], *, rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = [value for frame in zip(*channels) for value in frame]
    payload = struct.pack("<" + "h" * len(interleaved), *interleaved)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(len(channels))
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload)


def _reference_pcm16_rms(values: list[int]) -> float:
    return math.sqrt(sum(float(value) * value for value in values) / len(values)) / 32768.0


def test_channel_rms_matches_reference_for_mono_stereo_and_clamped_spans(tmp_path: Path) -> None:
    mono_values = [0, 16384, -16384, 32767]
    mono_path = tmp_path / "mono.wav"
    _write_pcm16(mono_path, [mono_values])
    mono_cache = {}
    mono = _channel_rms(mono_path, -1000, 2000, pcm_cache=mono_cache)
    assert mono == [_reference_pcm16_rms(mono_values)]
    assert _channel_rms(mono_path, 10000, 11000, pcm_cache=mono_cache) == [0.0]
    assert _channel_rms(mono_path, 5, 4, pcm_cache=mono_cache) is None
    assert len(mono_cache) == 1

    stereo_values = [[1000, -1000, 0, 0], [0, 0, 2000, -2000]]
    stereo_path = tmp_path / "stereo.wav"
    _write_pcm16(stereo_path, stereo_values)
    stereo_cache = {}
    stereo = _channel_rms(stereo_path, -1000, 2000, pcm_cache=stereo_cache)
    assert stereo == [_reference_pcm16_rms(stereo_values[0]), _reference_pcm16_rms(stereo_values[1])]
    assert _channel_rms(stereo_path, 0, 2, pcm_cache=stereo_cache) == [
        _reference_pcm16_rms(stereo_values[0][:2]), _reference_pcm16_rms(stereo_values[1][:2])
    ]
    assert len(stereo_cache) == 1


def test_synthetic_reader_uses_real_jsonl_maps_speaker_and_masks_clipped_text(tmp_path: Path) -> None:
    root = tmp_path / "data" / "synthetic_it_t6_v3"
    audio_rel = "fixtures/media/audio/train/synthetic-it-t6-00000000.wav"
    audio = tmp_path / audio_rel
    _write_stereo(audio, rate=16000, duration_s=0.2, ranges={0: [(0, 0.08)], 1: [(0.08, 0.2)]})
    row = {
        "conversation_id": "synthetic-it-t6-00000000",
        "script_version": "synthetic-it-t6-v4.0.0",
        "split": "train",
        "template_family_id": "shared_space",
        "turn_pattern_id": "bc_yield_replan",
        "sample_rate": 16000,
        "channels": 2,
        "duration_s": 0.2,
        "audio_path": audio_rel,
        "audio_sha256": "a" * 64,
        "channel_map": {"0": "fictional_user", "1": "fictional_assistant"},
        "utterances": [
            {"utterance_id": "u1", "speaker": "user", "role": "ordinary", "text": "ciao", "start_s": 0, "end_s": 0.08, "measured_duration_s": 0.08},
            {"utterance_id": "u2", "speaker": "assistant", "role": "assistant_backchannel", "text": "sì", "start_s": 0.08, "end_s": 0.12, "measured_duration_s": 0.35},
        ],
        "action_segments": [
            {"start_ms": 0, "end_ms": 80, "action": "LISTEN"},
            {"start_ms": 80, "end_ms": 160, "action": "BACKCHANNEL"},
            {"start_ms": 160, "end_ms": 240, "action": "YIELD"},
        ],
    }
    _write_jsonl(root / "manifest.jsonl", [row])

    conversations = read_synthetic_v3_manifest(root)

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation["split"] == "train"
    assert conversation["source_group_id"] == "shared_space"
    assert conversation["language_training_eligible"] is True
    assert conversation["source"] == "synthetic_it_t6"
    assert conversation["script_version"] == "synthetic-it-t6-v4.0.0"
    assert conversation["channel_map"] == {"0": "user", "1": "assistant"}
    assert conversation["audio_path"] == str(audio.resolve())
    assert [(segment["role"], segment["script_role"], segment["channel_index"]) for segment in conversation["segments"]] == [
        ("user", "ordinary", 0),
        ("assistant", "assistant_backchannel", 1),
    ]
    assert conversation["segments"][0]["text_mask"] is True
    assert conversation["segments"][1]["text_mask"] is False
    assert conversation["segments"][1]["measured_duration_s"] == 0.35
    assert conversation["segments"][1]["end_ms"] == 120
    assert conversation["frames"][-1]["target_action"] == "YIELD"
    assert conversation["frames"][-1]["end_ms"] == 240
    assert conversation["frames"][-1]["valid_ms"] == 40
    assert conversation["frames"][-1]["valid_samples"] == 640
    assert conversation["frames"][-1]["action_mask"] is True
    assert conversation["source_action_segments"][-1]["end_ms"] == 240

    legacy_row = {**row, "conversation_id": "legacy-v3", "script_version": "synthetic-it-t6-v3.0.0"}
    _write_jsonl(root / "manifest.jsonl", [legacy_row])
    legacy = read_synthetic_v3_manifest(root)[0]
    assert legacy["language_training_eligible"] is False
    assert legacy["frames"][-1]["target_action"] == "YIELD"

    explicitly_disabled = {**row, "conversation_id": "v4-disabled", "language_training_eligible": False}
    _write_jsonl(root / "manifest.jsonl", [explicitly_disabled])
    disabled = read_synthetic_v3_manifest(root)[0]
    assert disabled["language_training_eligible"] is False

    future = {
        **row,
        "conversation_id": "future-script",
        "script_version": "synthetic-it-t6-v5.1.0",
        "language_training_eligible": True,
    }
    _write_jsonl(root / "manifest.jsonl", [future])
    future_conversation = read_synthetic_v3_manifest(root)[0]
    assert future_conversation["language_training_eligible"] is True


def test_smoothconv_native_manifest_evidence_maps_audio_without_action_promotion(tmp_path: Path) -> None:
    root = tmp_path / "data" / "hf_slices" / "smoothconv_pinned_20260930_full40h"
    audio = root / "smoothconv" / "fixture.wav"
    _write_stereo(audio, rate=48000, duration_s=0.2, ranges={0: [(0, 0.08)], 1: [(0.12, 0.2)]})
    manifest = {
        "dataset": "SmoothConv",
        "source_key": "smooth-key-1",
        "source_key_sha256": "c" * 64,
        "source_group_id": "smooth-group-1",
        "local_path": "smoothconv/fixture.wav",
        "duration_seconds": 0.2,
        "sample_rate_hz": 48000,
        "channel_count": 2,
        "sha256": "b" * 64,
        "project_split": "unassigned",
    }
    evidence = {
        "source_dataset": "SmoothConv",
        "source_key": "smooth-key-1",
        "source_key_sha256": "c" * 64,
        "source_audio_sha256": "b" * 64,
        "source_group_id": "smooth-group-1",
        "project_split": "unassigned",
        "speaker_identity_status": "verified_within_record",
        "source_events": [
            {
                "source_event_id": "event-a",
                "source_channel_index": 0,
                "source_start_ms": 0,
                "source_end_ms": 80,
                "candidate_t6_action": "BACKCHANNEL",
                "target_action": "IGNORE",
                "source_annotation_event": {
                    "channelIndex": 0, "start": 0, "end": 80, "text": "ciao",
                    "attributes": {"turn": "complete", "speaker": "A1", "other_turn": ["interruption"], "speech event": ["echo"]},
                },
            },
            {
                "source_event_id": "event-b",
                "source_channel_index": 1,
                "source_start_ms": 120,
                "source_end_ms": 200,
                "candidate_t6_action": "YIELD",
                "target_action": "IGNORE",
                "source_annotation_event": {"channelIndex": 1, "start": 120, "end": 200, "text": "mondo"},
            },
        ],
    }
    _write_jsonl(root / "smoothconv.slice.jsonl", [manifest])
    evidence_path = tmp_path / "data" / "training_gates" / "evidence.jsonl"
    _write_jsonl(evidence_path, [evidence])

    conversations = read_smoothconv(root, evidence_path=evidence_path)

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation["source"] == "SmoothConv"
    assert conversation["split"] is None
    assert conversation["sample_rate_hz"] == 48000
    assert conversation["duration_ms"] == 200
    assert conversation["channel_map"] == {}
    assert [segment["text"] for segment in conversation["segments"]] == ["ciao", "mondo"]
    assert [segment["channel_index"] for segment in conversation["segments"]] == [0, 1]
    assert all(segment["role"] is None for segment in conversation["segments"])
    assert all(segment["task_kind"] == "asr" for segment in conversation["segments"])
    assert all(segment["audio_channel_verified"] for segment in conversation["segments"])
    assert all(segment["text_mask"] and segment["time_mask"] for segment in conversation["segments"])
    assert all(segment["action_mask"] is False for segment in conversation["segments"])
    assert conversation["segments"][0]["native_turn_state"] == "complete"
    assert conversation["segments"][0]["native_other_turn"] == ["interruption"]
    assert conversation["segments"][0]["native_paralinguistic"] == ["echo"]
    assert conversation["segments"][0]["source_speaker_id"] == "A1"
    assert conversation["segments"][0]["source_other_speaker_id"] is None
    assert conversation["frames"] == []
    counts = summarize_preparation_counts(conversations)
    assert counts["by_source"]["SmoothConv"]["asr_full_text"] == 2
    assert counts["by_source"]["SmoothConv"]["action_supervision"] == 0


def test_duplexconv_nested_annotation_tracks_are_role_free_and_keep_bad_times(tmp_path: Path) -> None:
    root = tmp_path / "data" / "hf_slices" / "duplexconv_pinned_20260930_full75h"
    audio = root / "duplexconv" / "fixture.wav"
    _write_stereo(audio, rate=48000, duration_s=0.2, ranges={0: [(0, 0.1)], 1: [(0.1, 0.2)]})
    manifest = {
        "dataset": "DuplexConv",
        "source_key": "duplex-key-1",
        "source_key_sha256": "d" * 64,
        "source_group_id": "duplex-group-1",
        "local_path": "duplexconv/fixture.wav",
        "duration_seconds": 0.2,
        "sample_rate_hz": 48000,
        "channel_count": 2,
        "sha256": "e" * 64,
        "project_split": "unassigned",
    }
    pairing = {
        "source_dataset": "DuplexConv",
        "source_key": "duplex-key-1",
        "source_key_sha256": "d" * 64,
        "audio_sha256": "e" * 64,
        "audio_local_path": "duplexconv/fixture.wav",
        "annotation_json_path": "annotations/ann.json",
        "annotation_track_count": 2,
        "event_count": 3,
        "event_time_boundary_status": "anomaly",
        "speaker_role_status": "no_user_assistant_role_inferred",
        "source_labels_status": "hypothesis_only_unadjudicated",
    }
    annotation = {
        "nTrack": 2,
        "timeLenInSec": 0.2,
        "fs": 48000,
        "asr": [
            [
                {"startInMs": 0, "endInMs": 80, "LID": "it", "labels": {"txt": "ciao", "paralinguistic": ["laugh"], "emotion": "neutral"}, "speaker": "source-speaker-a", "state": "<|complete|>"},
                {"startInMs": 160, "endInMs": 240, "LID": "it", "labels": {"txt": "fuori"}, "speaker": "source-speaker-a", "state": "<|incomplete|>"},
            ],
            [{"startInMs": 100, "endInMs": 180, "LID": "it", "labels": {"txt": "salve"}, "speaker": "source-speaker-b", "state": "<|backchannel|>"}],
        ],
    }
    _write_jsonl(root / "duplexconv.slice.jsonl", [manifest])
    review_dir = root / "annotation_pairing_review"
    _write_jsonl(review_dir / "pairing_review.jsonl", [pairing])
    (review_dir / "annotations").mkdir(parents=True)
    (review_dir / "annotations" / "ann.json").write_text(json.dumps(annotation), encoding="utf-8")

    conversations = read_duplexconv(root)

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation["sample_rate_hz"] == 48000
    assert conversation["source_group_id"] == "DuplexConv:duplex-group-1"
    assert conversation["channel_map"] == {}
    assert len(conversation["segments"]) == 3
    first, out_of_range, other_track = conversation["segments"]
    assert (first["text"], first["source_track_index"], first["channel_index"]) == ("ciao", 0, 0)
    assert first["role"] is None and first["task_kind"] == "asr"
    assert first["audio_channel_verified"] and first["text_mask"] and first["time_mask"]
    assert first["native_turn_state"] == "<|complete|>"
    assert first["native_paralinguistic"] == ["laugh"]
    assert first["native_labels"] == {"paralinguistic": ["laugh"], "emotion": "neutral"}
    assert first["language"] == "it"
    assert (other_track["text"], other_track["source_track_index"], other_track["channel_index"]) == ("salve", 1, 1)
    assert other_track["role"] is None and other_track["audio_channel_verified"]
    assert out_of_range["start_ms"] == 160 and out_of_range["end_ms"] == 240
    assert out_of_range["time_mask"] is False and out_of_range["text_mask"] is False
    assert all(segment["action_mask"] is False for segment in conversation["segments"])
    assert conversation["frames"] == []
    counts = summarize_preparation_counts(conversations)
    assert counts["by_source"]["DuplexConv"]["asr_full_text"] == 2
    assert counts["by_source"]["DuplexConv"]["asr_out_of_range"] == 1


def _real_conversation_fixture(*, yield_cue: bool = False) -> dict:
    def segment(segment_id, channel, start, end, state, text, speaker, *, other_turn=None, other_speaker=None):
        return {
            "segment_id": segment_id,
            "text": text,
            "start_ms": start,
            "end_ms": end,
            "role": None,
            "task_kind": "asr",
            "channel_index": channel,
            "source_track_index": channel,
            "audio_channel_verified": True,
            "time_mask": True,
            "text_mask": True,
            "action_mask": False,
            "native_turn_state": state,
            "native_other_turn": other_turn or [],
            "native_paralinguistic": [],
            "source_speaker_id": speaker,
            "source_other_speaker_id": other_speaker,
        }

    incomplete = segment(
        "local-incomplete", 0, 0, 160, "incomplete", "local partial", "A",
        other_turn=["interruption"] if yield_cue else None,
        other_speaker="B" if yield_cue else None,
    )
    return {
        "conversation_id": "smooth-source-1",
        "source_group_id": "same-original-session",
        "source": "SmoothConv",
        "split": "train",
        "duration_ms": 400,
        "sample_rate_hz": 16000,
        "channels": 2,
        "audio_path": "fixture.wav",
        "channel_map": {},
        "segments": [
            incomplete,
            segment("remote-complete", 1, 80, 240, "complete", "remote response", "B"),
            segment("local-complete", 0, 240, 320, "complete", "local reply", "A"),
            segment("remote-backchannel", 1, 240, 280, "backchannel", "remote acknowledgement", "B"),
            segment("future-local", 0, 320, 400, "complete", "future local reply", "A"),
        ],
        "frames": [],
    }


def test_real_conversational_views_are_directed_masked_and_causal() -> None:
    from training.duplex_qlora.sources import build_real_conversational_views

    views = build_real_conversational_views([_real_conversation_fixture()])

    assert len(views) == 2
    local_zero, local_one = views
    assert local_zero["view_local_channel"] == 0
    assert local_one["view_local_channel"] == 1
    assert local_zero["conversation_id"] != local_one["conversation_id"]
    assert local_zero["source_group_id"] == local_one["source_group_id"] == "same-original-session"
    assert local_zero["split"] == local_one["split"] == "train"
    assert local_zero["channel_map"] == {"0": "assistant", "1": "user"}
    assert local_one["channel_map"] == {"1": "assistant", "0": "user"}
    local_targets = [segment for segment in local_zero["segments"] if segment["role"] == "assistant"]
    assert [segment["target_mask"] for segment in local_targets] == [False, True, True]
    assert all(segment["task_kind"] == "conversation_response" for segment in local_targets)
    assert local_zero["segments"][0]["source_role"] is None
    assert local_zero["frames"][0]["target_action"] == "SPEAK"
    assert local_zero["frames"][1]["target_action"] == "CONTINUE"
    assert local_zero["frames"][2]["target_action"] == "LISTEN"
    assert local_zero["frames"][3]["target_action"] == "SPEAK"
    assert local_one["frames"][3]["target_action"] == "BACKCHANNEL"
    assert local_one["frames"][2]["target_action"] == "CONTINUE"
    assert all(frame["target_action"] != "YIELD" for view in views for frame in view["frames"])
    # A future local utterance cannot label frames before its observed onset.
    assert local_zero["frames"][0]["target_action"] == "SPEAK"
    assert local_zero["frames"][1]["target_action"] == "CONTINUE"


def test_real_yield_requires_explicit_remote_speaker_link_and_observed_stop() -> None:
    from training.duplex_qlora.sources import build_real_conversational_views

    views = build_real_conversational_views([_real_conversation_fixture(yield_cue=True)])
    local_zero = next(view for view in views if view["view_local_channel"] == 0)

    assert local_zero["frames"][0]["target_action"] == "SPEAK"
    assert local_zero["frames"][1]["target_action"] == "CONTINUE"
    assert local_zero["frames"][2]["target_action"] == "YIELD"

    unlinked = _real_conversation_fixture(yield_cue=True)
    unlinked["segments"][0]["source_other_speaker_id"] = None
    no_remote_continuation = _real_conversation_fixture(yield_cue=True)
    no_remote_continuation["segments"][1]["end_ms"] = 160
    for candidate in (unlinked, no_remote_continuation):
        candidate_views = build_real_conversational_views([candidate])
        assert all(frame["target_action"] != "YIELD" for view in candidate_views for frame in view["frames"])

    unlinked = _real_conversation_fixture(yield_cue=True)
    unlinked["segments"][0]["source_other_speaker_id"] = None
    no_remote_continuation = _real_conversation_fixture(yield_cue=True)
    no_remote_continuation["segments"][1]["end_ms"] = 160
    for candidate in (unlinked, no_remote_continuation):
        candidate_views = build_real_conversational_views([candidate])
        assert all(frame["target_action"] != "YIELD" for view in candidate_views for frame in view["frames"])


def test_real_conversational_views_require_verified_two_channel_pairing() -> None:
    from training.duplex_qlora.sources import build_real_conversational_views

    three_channel = _real_conversation_fixture()
    three_channel["channels"] = 3
    unverified = _real_conversation_fixture()
    for segment in unverified["segments"]:
        if segment["channel_index"] == 0:
            segment["audio_channel_verified"] = False

    assert build_real_conversational_views([three_channel]) == []
    assert build_real_conversational_views([unverified]) == []


def test_real_view_cap_balances_source_and_split_buckets_without_splitting_groups() -> None:
    from training.duplex_qlora.sources import build_real_conversational_views

    conversations = []
    for source in ("SmoothConv", "DuplexConv"):
        for split in ("train", "validation"):
            for index in range(2):
                row = _real_conversation_fixture()
                row["source"] = source
                row["split"] = split
                row["conversation_id"] = f"{source}-{split}-{index}"
                row["source_group_id"] = f"{source}:{split}:{index}"
                conversations.append(row)

    views = build_real_conversational_views(conversations, max_conversations=4)

    assert len(views) == 8
    selected = {(view["source"], view["split"], view["source_conversation_id"]) for view in views}
    selected_physical = {(source, split, conversation_id) for source, split, conversation_id in selected}
    assert len(selected_physical) == 4
    assert {(source, split) for source, split, _conversation_id in selected_physical} == {
        ("SmoothConv", "train"),
        ("SmoothConv", "validation"),
        ("DuplexConv", "train"),
        ("DuplexConv", "validation"),
    }
    for key in selected_physical:
        assert sum(
            (view["source"], view["split"], view["source_conversation_id"]) == key
            for view in views
        ) == 2


def test_duplex_native_backchannel_maps_at_onset_and_unknown_silence_stays_masked() -> None:
    from training.duplex_qlora.sources import build_real_conversational_views

    duplex = _real_conversation_fixture()
    duplex["source"] = "DuplexConv"
    for segment in duplex["segments"]:
        segment["native_turn_state"] = {
            "complete": "<|complete|>",
            "incomplete": "<|incomplete|>",
            "backchannel": "<|backchannel|>",
        }.get(segment["native_turn_state"], "<|unknown|>")
    views = build_real_conversational_views([duplex])
    local_one = next(view for view in views if view["view_local_channel"] == 1)
    backchannel = next(segment for segment in local_one["segments"] if segment["native_turn_state"] == "backchannel")

    assert local_one["frames"][3]["target_action"] == "BACKCHANNEL"
    assert backchannel["native_turn_state_raw"] == "<|backchannel|>"
    assert all(frame["target_action"] != "REPLAN" for view in views for frame in view["frames"])
    unknown = _real_conversation_fixture()
    unknown["segments"][0]["native_turn_state"] = "unknown"
    unknown_views = build_real_conversational_views([unknown])
    local_zero = next(view for view in unknown_views if view["view_local_channel"] == 0)
    assert local_zero["frames"][0]["action_mask"] is False
    assert local_zero["frames"][0]["target_action"] is None


def test_real_views_materialize_sparse_windows_and_keep_source_groups_atomic(tmp_path: Path) -> None:
    from training.duplex_qlora.sources import build_real_conversational_views
    from training.duplex_qlora.prepare import prepare_splits, read_jsonl

    long = _real_conversation_fixture()
    long["duration_ms"] = 10000
    long["segments"] = [
        {**long["segments"][0], "end_ms": 10000, "native_turn_state": "complete"},
        {**long["segments"][1], "start_ms": 0, "end_ms": 10000},
    ]
    views = build_real_conversational_views([long], max_conversations=1)
    assert len(views) == 2
    assert all(len(view["frames"]) <= 3 for view in views)
    assert all([frame["target_action"] for frame in view["frames"]] == ["SPEAK", "CONTINUE", "CONTINUE"] for view in views)
    assert all([frame["frame_id"] for frame in view["frames"]] == sorted(frame["frame_id"] for frame in view["frames"]) for view in views)

    paired = _real_conversation_fixture()
    paired_again = _real_conversation_fixture()
    paired_again["conversation_id"] = "same-session-second-source-view"
    assert build_real_conversational_views([paired, paired_again], max_conversations=1) == []
    paired_views = build_real_conversational_views([paired, paired_again], max_conversations=2)
    assert len(paired_views) == 4
    prepare_splits(paired_views, tmp_path / "prepared")
    split_rows = {
        split: read_jsonl(tmp_path / "prepared" / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    assert sorted(len(rows) for rows in split_rows.values()) == [0, 0, 4]


def test_split_preserves_explicit_synthetic_split_and_rejects_group_conflict() -> None:
    rows = [
        {"conversation_id": "synth-a", "source_group_id": "family-a", "split": "train"},
        {"conversation_id": "raw-a", "source_group_id": "family-a", "split": None},
        {"conversation_id": "synth-b", "source_group_id": "family-b", "split": "validation"},
    ]

    assignments = split_conversations(rows, seed=17)

    assert assignments["synth-a"] == assignments["raw-a"] == "train"
    assert assignments["synth-b"] == "validation"
    conflicted = split_conversations(
        [
            {"conversation_id": "one", "source_group_id": "same", "split": "train"},
            {"conversation_id": "two", "source_group_id": "same", "split": "validation"},
        ]
    )
    assert conflicted["one"] == conflicted["two"]


def test_v3_clean_asr_salvages_only_exact_complete_slot_free_templates(tmp_path: Path) -> None:
    root = tmp_path / "data" / "synthetic_it_t6_v3"
    audio_rel = "fixtures/media/audio/train/synthetic-it-t6-00000000.wav"
    audio = tmp_path / audio_rel
    _write_stereo(audio, rate=16000, duration_s=1.0, ranges={0: [(0.05, 0.2)], 1: [(0.25, 0.4)]})
    row = {
        "conversation_id": "synthetic-it-t6-00000000",
        "script_version": "synthetic-it-t6-v3.0.0",
        "split": "train",
        "template_family_id": "service_recovery",
        "sample_rate": 16000,
        "channels": 2,
        "duration_s": 1.0,
        "audio_path": audio_rel,
        "channel_map": {"0": "fictional_user", "1": "fictional_assistant"},
        "utterances": [
            {"utterance_id": "user_ack", "speaker": "user", "text": "Chiaro, ho registrato la segnalazione.", "start_s": 0.05, "end_s": 0.20, "measured_duration_s": 0.15},
            {"utterance_id": "assistant_backchannel", "speaker": "assistant", "text": "D'accordo.", "start_s": 0.25, "end_s": 0.60, "measured_duration_s": 1.4},
            {"utterance_id": "assistant_yield_response", "speaker": "assistant", "text": "Mi fermo e la.", "start_s": 0.25, "end_s": 0.40, "measured_duration_s": 0.15},
            {"utterance_id": "assistant_yield_context", "speaker": "assistant", "text": "Rispondo su focus.", "start_s": 0.25, "end_s": 0.40, "measured_duration_s": 0.15},
        ],
    }
    _write_jsonl(root / "manifest.jsonl", [row])

    summary: dict = {}
    conversations = read_synthetic_v3_clean_asr(root, summary_out=summary)

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation["source"] == "synthetic_v3_clean_asr"
    assert conversation["synthetic"] is False
    assert conversation["language_training_eligible"] is True
    assert conversation["conversation_id"] == f"synthetic_v3_clean_asr:{row['conversation_id']}"
    assert conversation["original_conversation_id"] == row["conversation_id"]
    assert conversation["source_group_id"] == row["template_family_id"]
    assert conversation["template_family_id"] == row["template_family_id"]
    assert conversation["split"] == "train"
    assert conversation["frames"] == [] and conversation["events"] == []
    assert len(conversation["segments"]) == 1
    segment = conversation["segments"][0]
    assert segment["segment_id"] == "user_ack"
    assert segment["role"] is None and segment["task_kind"] == "asr"
    assert segment["channel_index"] == 0 and segment["audio_channel_verified"] is True
    assert segment["text_mask"] is True and segment["time_mask"] is True
    assert summary["eligible_utterances"] == 1
    assert abs(summary["eligible_speech_seconds"] - 0.15) < 1e-9
    assert abs(summary["eligible_speech_seconds_by_split"]["train"] - 0.15) < 1e-9
    assert summary["quarantined_utterances"] == 3
    assert summary["quarantined_by_reason"]["cropped_utterance"] == 1
    assert summary["quarantined_by_reason"]["template_text_mismatch"] == 1
    assert summary["quarantined_by_reason"]["slot_bearing_or_unlisted_template"] == 1
