from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from duplex_kernel.audio_response_conditioning import build_audio_response_messages, causal_audio_window
from training.duplex_qlora.action_conditioned_data import build_action_conditioned_records
from training.duplex_qlora import cli
from training.duplex_qlora.stage1 import (
    _format_chat,
    _make_action_weighted_schedule,
    _read_audio_for_record,
    _resolve_adapter_dir_for_load,
    _training_config,
    _validate_direct_audio_records,
    build_training_inputs,
)


class _Tokenizer:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        self.messages = [dict(message) for message in messages]
        prompt = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        return prompt + ("\nassistant:" if add_generation_prompt else "")


class _AudioProcessor:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.calls = []

    def __call__(self, *, text, return_tensors, padding, **kwargs):
        self.calls.append({"text": text[0], "keys": set(kwargs), "audio": kwargs.get("audio")})
        prefix = [1, 2, 3] if "audio" in kwargs else [1, 2]
        has_target = "\nassistant:" in text[0] and not text[0].endswith("assistant:")
        input_ids = prefix + ([4, 5] if has_target else [])
        result = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        }
        if "audio" in kwargs:
            result["input_features"] = torch.ones((1, 4, 8), dtype=torch.float32)
            result["input_features_mask"] = torch.ones((1, 4), dtype=torch.long)
        return result


def _row(audio_path: Path) -> dict:
    return {
        "conversation_id": "v4-train-audio",
        "source_group_id": "group-audio-1",
        "source": "synthetic_it_t6",
        "synthetic": True,
        "script_version": "synthetic-it-t6-v4.0.0",
        "split": "train",
        "language": "Italian",
        "audio_path": str(audio_path),
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "sample_rate_hz": 16000,
        "duration_ms": 5000,
        "channel_map": {"0": "user", "1": "assistant"},
        "source_action_segments": [
            {"action": "SPEAK", "start_ms": 1020, "end_ms": 1260},
            {"action": "BACKCHANNEL", "start_ms": 2220, "end_ms": 2780},
            {"action": "REPLAN", "start_ms": 3220, "end_ms": 3460},
        ],
        "segments": [
            {"role": "user", "segment_id": "u0", "start_ms": 0, "end_ms": 800,
             "text": "Prima richiesta completa.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_start", "segment_id": "a0",
             "start_ms": 1000, "end_ms": 1800, "text": "Risposta iniziale.",
             "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "user", "segment_id": "u1", "start_ms": 1900, "end_ms": 2500,
             "text": "Trascrizione utente ancora attiva.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_backchannel", "segment_id": "a1",
             "start_ms": 2220, "end_ms": 2780, "text": "Certo.",
             "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "user", "segment_id": "u2", "start_ms": 2800, "end_ms": 3200,
             "text": "Correzione completa.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_replan", "segment_id": "a2",
             "start_ms": 3220, "end_ms": 4000, "text": "Rivedo il piano.",
             "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "assistant", "script_role": "assistant_continue", "segment_id": "a3",
             "start_ms": 4500, "end_ms": 4800, "text": "Aggiungo un dettaglio utile.",
             "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "user", "segment_id": "u3", "start_ms": 2100, "end_ms": 2180,
             "text": "Completamento dopo il cutoff.", "time_mask": True, "text_mask": True},
        ],
    }


def _wave(path: Path) -> None:
    user = np.full(5000 * 16, 0.2, dtype=np.float32)
    assistant = np.full(5000 * 16, -0.4, dtype=np.float32)
    sf.write(path, np.stack([user, assistant], axis=1), 16000, subtype="FLOAT")


def test_action_targets_use_causal_user_audio_and_generic_prompt(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, report = build_action_conditioned_records([_row(audio_path)], split="train")

    assert [record["action"] for record in records] == ["SPEAK", "BACKCHANNEL", "REPLAN", "UNKNOWN"]
    assert report["records_by_action"] == {"BACKCHANNEL": 1, "REPLAN": 1, "SPEAK": 1, "UNKNOWN": 1}
    assert report["records_by_sampling_stratum"] == {
        "conversational_regular": 1,
        "labelled_speak": 1,
        "labelled_backchannel": 1,
        "labelled_replan": 1,
    }
    assert all(record["conditioning_mode"] == "baselineaudio_response" for record in records)
    assert all(record["audio_mask"] is True for record in records)
    assert all(record["audio_channel_index"] == 0 for record in records)
    assert all(record["sample_rate_hz"] == 16000 for record in records)
    assert "action_instructions" not in report

    backchannel = records[1]
    assert backchannel["target_start_ms"] == 2220
    assert backchannel["input_cutoff_ms"] == 2160
    assert backchannel["input_cutoff_frame_count"] == 27
    assert backchannel["input_cutoff_frame_id"] == 26
    assert (backchannel["audio_start_ms"], backchannel["audio_end_ms"]) == (1900, 2160)
    assert backchannel["action"] == "BACKCHANNEL"  # metadata for sampling/evaluation only
    assert all(message["content"] != "Trascrizione utente ancora attiva." for message in backchannel["history_messages"])
    assert all(message["content"] != "Completamento dopo il cutoff." for message in backchannel["history_messages"])
    assert any(message["content"] == "Risposta iniziale." for message in backchannel["history_messages"])

    pcm = _read_audio_for_record(backchannel)
    assert pcm.shape == ((2160 - 1900) * 16,)
    np.testing.assert_allclose(pcm, 0.2, atol=1e-6)

    messages = build_audio_response_messages(
        language=backchannel["language"],
        history_messages=backchannel["history_messages"],
        audio_token="<|audio|>",
    )
    assert messages[0]["role"] == "system"
    assert "BACKCHANNEL" not in messages[0]["content"]
    assert "REPLAN" not in str(messages)
    assert messages[-1] == {"role": "user", "content": "<|audio|>"}
    assert all(message["content"] != "Trascrizione utente ancora attiva." for message in messages)
    replan = records[2]
    assert replan["input_cutoff_ms"] == 3200
    assert replan["audio_start_ms"] == 2800
    assert "Correzione completa." not in str(replan["history_messages"])
    assert any(message["content"] == "Risposta iniziale." for message in replan["history_messages"])


def test_protocol_v2_targets_include_only_causal_heard_label_and_response(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, report = build_action_conditioned_records(
        [_row(audio_path)], split="train", response_protocol="heard_action_say_v2",
    )

    assert [record["action"] for record in records] == ["SPEAK", "BACKCHANNEL", "REPLAN"]
    assert report["records_by_action"] == {"BACKCHANNEL": 1, "REPLAN": 1, "SPEAK": 1, "UNKNOWN": 0}
    payloads = [json.loads(record["target_text"]) for record in records]
    assert all(list(payload) == ["heard", "action", "say"] for payload in payloads)
    assert payloads[0] == {
        "heard": "Prima richiesta completa.",
        "action": "SPEAK",
        "say": "Risposta iniziale.",
    }
    assert payloads[1] == {
        "heard": "",
        "action": "BACKCHANNEL",
        "say": "Certo.",
    }
    assert payloads[2] == {
        "heard": "Correzione completa.",
        "action": "REPLAN",
        "say": "Rivedo il piano.",
    }
    assert all(record["audio_response_protocol"] == "heard_action_say_v2" for record in records)
    assert all(record.get("current_user_transcript") is None and record.get("current_user_text") is None for record in records)
    current_user_texts = (
        "Prima richiesta completa.",
        "Trascrizione utente ancora attiva.",
        "Correzione completa.",
    )
    for record, current_user_text in zip(records, current_user_texts):
        assert all(message["content"] != current_user_text for message in record["history_messages"])


def test_training_prompt_uses_shared_audio_response_messages_and_real_pcm(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, _ = build_action_conditioned_records([_row(audio_path)], split="train")
    record = records[1]
    processor = _AudioProcessor()

    expected = build_audio_response_messages(
        language=record["language"],
        history_messages=record["history_messages"],
        audio_token=processor.audio_token,
    )
    _format_chat(processor, record, add_target=False)
    assert processor.tokenizer.messages == expected
    inputs = build_training_inputs(processor, record)
    calls_after_first_encoding = len(processor.calls)
    cached_inputs = build_training_inputs(processor, record)

    assert "input_features" in inputs and "input_features_mask" in inputs
    assert len(processor.calls) == calls_after_first_encoding
    assert torch.equal(inputs["input_features"], cached_inputs["input_features"])
    assert all("audio" in call["keys"] for call in processor.calls)
    assert all(call["audio"][0].shape == ((2160 - 1900) * 16,) for call in processor.calls)
    assert all(np.allclose(call["audio"][0], 0.2, atol=1e-6) for call in processor.calls)
    assert int(inputs["labels"].ne(-100).sum()) > 0


def test_ambiguous_user_channel_is_rejected(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    row = _row(audio_path)
    row["channel_map"] = {"0": "user", "1": "user", "2": "assistant"}

    try:
        build_action_conditioned_records([row], split="train")
    except ValueError as exc:
        assert "unico canale fisico user" in str(exc)
    else:
        raise AssertionError("ambiguous user audio channel must fail closed")


def test_known_v4_source_supplies_italian_when_prepared_language_is_missing(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    row = _row(audio_path)
    row.pop("language")
    for segment in row["segments"]:
        segment.pop("language", None)

    records, report = build_action_conditioned_records([row], split="train")

    assert len(records) == 4
    assert all(record["language"] == "it-IT" for record in records)
    assert report["records_by_action"]["UNKNOWN"] == 1


def test_direct_audio_records_are_validated_and_action_balanced(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, _ = build_action_conditioned_records([_row(audio_path)], split="train")

    validated = _validate_direct_audio_records(records, conditioning_mode="baselineaudio_response")
    schedule, report = _make_action_weighted_schedule(validated, max_steps=20, seed=7)
    assert len(schedule) == 20
    assert report["examples_by_sampling_stratum"] == {
        "conversational_regular": 8,
        "labelled_speak": 6,
        "labelled_backchannel": 3,
        "labelled_replan": 3,
    }
    assert all(record["audio_response_contract"] == "audio_response_conditioning_v1" for record in validated)

    leaked = dict(records[2])
    leaked["context_text"] = "Correzione completa."
    try:
        _validate_direct_audio_records([leaked], conditioning_mode="baselineaudio_response")
    except ValueError as exc:
        assert "context_text" in str(exc)
    else:
        raise AssertionError("current-user transcript in direct audio record must fail closed")


def test_known_action_only_dataset_uses_explicit_fallback_schedule(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, _ = build_action_conditioned_records([_row(audio_path)], split="train")
    known_records = _validate_direct_audio_records(
        [record for record in records if record["action"] != "UNKNOWN"],
        conditioning_mode="baselineaudio_response",
    )

    smoke, smoke_report = _make_action_weighted_schedule(known_records, max_steps=4, seed=7)
    full, full_report = _make_action_weighted_schedule(known_records, max_steps=300, seed=7)

    assert len(smoke) == 4
    assert smoke_report["examples_by_sampling_stratum"] == {
        "conversational_regular": 0,
        "labelled_speak": 2,
        "labelled_backchannel": 1,
        "labelled_replan": 1,
    }
    assert full_report["examples_by_sampling_stratum"] == {
        "conversational_regular": 0,
        "labelled_speak": 150,
        "labelled_backchannel": 75,
        "labelled_replan": 75,
    }
    assert full_report["sampling"] == "50/25/25 labelled_speak/labelled_backchannel/labelled_replan; regular unknown-action targets unavailable"


def test_warm_start_adapter_is_distinct_from_resume_checkpoint(tmp_path):
    initial = tmp_path / "initial" / "adapter"
    initial.mkdir(parents=True)
    (initial / "adapter_config.json").write_text("{}", encoding="utf-8")
    (initial / "adapter_model.safetensors").write_bytes(b"weights")
    (initial.parent / "training_manifest.json").write_text(
        json.dumps({"run_id": "source-run-1"}), encoding="utf-8"
    )

    resolved = _resolve_adapter_dir_for_load(resume_from=None, initial_adapter_dir=initial)
    config = _training_config(
        train_jsonl="train.jsonl", validation_jsonl="validation.jsonl",
        model_path="model", max_steps=300, learning_rate=1e-5, seed=7,
        conditioning_mode="baselineaudio_response", initial_adapter_dir=initial,
    )

    assert resolved == initial.resolve()
    assert config["initial_adapter_dir"] == str(initial.resolve())
    assert config["initial_adapter_run_id"] == "source-run-1"
    checkpoint = tmp_path / "step_000004"
    assert _resolve_adapter_dir_for_load(
        resume_from=checkpoint, initial_adapter_dir=initial,
    ) == checkpoint.resolve() / "adapter"


def test_direct_audio_validator_rejects_wrong_causal_cutoff(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, _ = build_action_conditioned_records([_row(audio_path)], split="train")
    corrupted = dict(records[1])
    corrupted["audio_end_ms"] = corrupted["target_start_ms"]
    try:
        _validate_direct_audio_records([corrupted], conditioning_mode="baselineaudio_response")
    except ValueError as exc:
        assert "audio_end_ms" in str(exc)
    else:
        raise AssertionError("non-frame-aligned or future audio cutoff must fail closed")


def test_speak_does_not_fall_back_to_older_user_turn_when_latest_is_still_open(tmp_path):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    row = _row(audio_path)
    row["source_action_segments"] = [{"action": "SPEAK", "start_ms": 1530, "end_ms": 1700}]
    row["segments"] = [
        {"role": "user", "segment_id": "old", "start_ms": 0, "end_ms": 800,
         "text": "Richiesta precedente.", "time_mask": True, "text_mask": True},
        {"role": "user", "segment_id": "current", "start_ms": 900, "end_ms": 1540,
         "text": "La richiesta è ancora in corso.", "time_mask": True, "text_mask": True},
        {"role": "assistant", "script_role": "assistant_start", "segment_id": "answer",
         "start_ms": 1530, "end_ms": 1800, "text": "Risposta prematura.",
         "time_mask": True, "text_mask": True, "target_mask": True},
    ]
    records, report = build_action_conditioned_records([row], split="train")

    assert records == []
    assert report["skipped"]["no_causal_current_user_turn:labelled_speak"] == 1


def test_runtime_backchannel_cutoff_needs_active_flag_not_future_user_end():
    window = causal_audio_window(
        action="BACKCHANNEL",
        target_start_ms=2220,
        current_user_start_ms=1900,
        current_user_end_ms=None,
        current_user_active=True,
    )

    assert window["audio_start_ms"] == 1900
    assert window["audio_end_ms"] == 2160
    assert window["input_cutoff_ms"] == 2160


def test_source_pcm_is_decoded_once_for_multiple_target_windows(tmp_path):
    import soundfile as sf
    from unittest.mock import patch

    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    records, _ = build_action_conditioned_records([_row(audio_path)], split="train")
    with patch("soundfile.read", wraps=sf.read) as read:
        first = _read_audio_for_record(records[0])
        second = _read_audio_for_record(records[1])

    assert first.size > 0 and second.size > 0
    assert read.call_count == 1


def test_prepare_audio_response_cli_writes_train_and_heldout_direct_audio_rows(tmp_path, capsys):
    audio_path = tmp_path / "conversation.wav"
    _wave(audio_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    train_row = _row(audio_path)
    val_row = _row(audio_path)
    val_row["conversation_id"] = "v4-val-audio"
    val_row["source_group_id"] = "group-audio-val"
    val_row["split"] = "validation"
    for split, row in (("train", train_row), ("validation", val_row)):
        (prepared / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = tmp_path / "audio_responses"
    assert cli.main([
        "prepare-audio-responses", "--prepared-dir", str(prepared),
        "--output-dir", str(output), "--max-audio-seconds", "30",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    train_records = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    validation_records = [json.loads(line) for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["audio_response_contract"] == "audio_response_conditioning_v1"
    assert len(train_records) == 4 and len(validation_records) == 4
    assert all(record["audio_mask"] is True for record in train_records + validation_records)
    assert all(record["action_is_prompt_conditioning"] is False for record in train_records + validation_records)
    assert {record["source_group_id"] for record in train_records}.isdisjoint(
        {record["source_group_id"] for record in validation_records}
    )
