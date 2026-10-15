import pytest

from training.duplex_qlora.stage1 import _make_ordered_audio_asr_schedule


def _record(example_id, task_kind, action=None, *, source=None, split="train", group=None):
    return {
        "training_example_id": example_id,
        "task_mix_contract": "audio_asr_mix_v2",
        "split": split,
        "task_kind": task_kind,
        "action": action,
        "source": source or ("synthetic_v3_clean_asr" if task_kind == "asr" else "authored_synthetic_audio_candidate"),
        "conversation_id": f"conversation-{example_id}",
        "source_group_id": group or f"group-{example_id}",
        "synthetic": task_kind != "asr",
        "target_text": f"target-{example_id}",
        "target_variant_id": "focus_concise_1" if action == "SPEAK" else None,
    }


def _entry(record):
    return {key: record.get(key) for key in (
        "training_example_id", "task_mix_contract", "split", "task_kind", "action",
        "source", "conversation_id", "source_group_id", "target_variant_id",
    )}


def test_ordered_schedule_uses_exact_record_ids_and_reports_task_source_counts():
    records = [
        _record("asr-1", "asr"),
        _record("speak-1", "conversation_response", "SPEAK"),
        _record("bc-1", "conversation_response", "BACKCHANNEL"),
        _record("replan-1", "conversation_response", "REPLAN"),
    ]
    schedule_rows = [_entry(record) for record in (records[2], records[0], records[3], records[1])]

    scheduled, report = _make_ordered_audio_asr_schedule(
        records,
        schedule_rows=schedule_rows,
        max_steps=4,
        response_protocol="v2",
        target_token_counts=[2, 3, 4, 5],
    )

    assert [record["training_example_id"] for record in scheduled] == ["bc-1", "asr-1", "replan-1", "speak-1"]
    assert report["schedule_mode"] == "ordered_jsonl"
    assert report["examples_by_task"] == {"asr": 1, "conversation_response": 3}
    assert report["audio_response_by_action"] == {"SPEAK": 1, "BACKCHANNEL": 1, "REPLAN": 1}
    assert report["audio_response_by_action_source"]["SPEAK"] == {"authored_synthetic_audio_candidate": 1}


def test_ordered_schedule_rejects_duplicate_unknown_or_forged_ids():
    records = [_record("asr-1", "asr"), _record("speak-1", "conversation_response", "SPEAK")]
    schedule_rows = [_entry(records[0]), _entry(records[1])]

    with pytest.raises(ValueError, match="duplicate"):
        _make_ordered_audio_asr_schedule(
            records, schedule_rows=[schedule_rows[0], schedule_rows[0]], max_steps=2, response_protocol="v2",
        )
    with pytest.raises(ValueError, match="unknown"):
        _make_ordered_audio_asr_schedule(
            records, schedule_rows=[schedule_rows[0], {**schedule_rows[1], "training_example_id": "forged"}],
            max_steps=2, response_protocol="v2",
        )
    with pytest.raises(ValueError, match="metadata"):
        _make_ordered_audio_asr_schedule(
            records, schedule_rows=[schedule_rows[0], {**schedule_rows[1], "source_group_id": "heldout"}],
            max_steps=2, response_protocol="v2",
        )


def test_ordered_schedule_rejects_nontrain_rows_and_duplicate_dataset_ids():
    asr = _record("asr-1", "asr")
    validation = _record("speak-1", "conversation_response", "SPEAK", split="validation")
    with pytest.raises(ValueError, match="not a train record"):
        _make_ordered_audio_asr_schedule(
            [asr, validation], schedule_rows=[_entry(asr), _entry(validation)], max_steps=2,
            response_protocol="v2",
        )

    duplicate = _record("asr-1", "asr")
    with pytest.raises(ValueError, match="duplicate training_example_id"):
        _make_ordered_audio_asr_schedule(
            [asr, duplicate], schedule_rows=[_entry(asr)], max_steps=1, response_protocol="v2",
        )
