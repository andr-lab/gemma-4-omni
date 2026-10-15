import json

from duplex_kernel.audio_response_protocol import format_audio_response_v2_target
from training.duplex_qlora.audio_response_target_repairs import (
    build_repair_schedule,
    build_train_repair_candidates,
)


def _row(*, action="SPEAK", split="train", source="authored_synthetic_audio_candidate", profile="audio_grounded_v5"):
    heard = (
        "Aspetta, correggo un dato: dobbiamo considerare il limite settimanale d'acqua, "
        "non la disponibilità degli attrezzi."
        if action == "REPLAN"
        else "Vorrei organizzare il progetto; il punto più delicato è la disponibilità degli attrezzi."
    )
    say = (
        "Rivedo la proposta: verifichiamo il limite settimanale d'acqua, poi definiamo le aiuole per giovedì."
        if action == "REPLAN"
        else "Partirei da la distribuzione delle aiuole: possiamo preparare un prospetto e confrontarci giovedì."
    )
    if action == "BACKCHANNEL":
        heard, say = "", "Mhm."
    return {
        "conversation_id": "synthetic-v5-train-0001",
        "source_group_id": "community_garden",
        "source": source,
        "profile": profile,
        "split": split,
        "task_kind": "conversation_response",
        "audio_response_protocol": "heard_action_say_v2",
        "action": action,
        "target_utterance_id": f"assistant_{action.lower()}",
        "target_heard_text": heard,
        "target_say_text": say,
        "target_text": format_audio_response_v2_target(heard=heard, action=action, say=say),
        "history_messages": [{"role": "assistant", "content": "Risposta precedente."}],
        "audio_path": "/workspace/audio.wav",
        "input_cutoff_ms": 1000,
        "current_user_end_ms": 960,
    }


def test_speak_variants_keep_gold_focus_and_drop_unneeded_schedule_details():
    source = _row(action="SPEAK")
    candidates, summary = build_train_repair_candidates([source])

    assert len(candidates) == 2
    focus = "la distribuzione delle aiuole"
    for candidate in candidates:
        payload = json.loads(candidate["target_text"])
        assert payload["action"] == "SPEAK"
        assert payload["heard"] == source["target_heard_text"]
        assert focus in payload["say"]
        assert "giovedì" not in payload["say"]
        assert candidate["history_messages"] == source["history_messages"]
        assert candidate["audio_path"] == source["audio_path"]
        assert candidate["split"] == "train"
        assert candidate["augmentation_only"] is True
    assert summary["variants_by_action"] == {"SPEAK": 2, "REPLAN": 0}
    assert summary["slot_metrics"]["speak_focus_exact_match"] == {"matched": 2, "total": 2}


def test_replan_variants_use_positive_correction_and_exclude_explicitly_negated_slot():
    source = _row(action="REPLAN")
    candidates, summary = build_train_repair_candidates([source])

    assert len(candidates) == 2
    positive = "il limite settimanale d'acqua"
    negated = "la disponibilità degli attrezzi"
    for candidate in candidates:
        payload = json.loads(candidate["target_text"])
        assert payload["action"] == "REPLAN"
        assert payload["heard"] == source["target_heard_text"]
        assert positive in payload["say"]
        assert negated not in payload["say"]
        assert candidate["target_heard_text"] == source["target_heard_text"]
        assert candidate["history_messages"] == source["history_messages"]
    assert summary["slot_metrics"]["replan_positive_correction_exact_match"] == {"matched": 2, "total": 2}
    assert summary["slot_metrics"]["explicitly_negated_replan_slot_absent"] == {"matched": 2, "total": 2}


def test_speak_keeps_weekday_when_it_is_part_of_the_source_focus_slot():
    source = _row(action="SPEAK")
    source["target_say_text"] = (
        "Partirei da il percorso della domenica: possiamo preparare una lista condivisa "
        "e confrontarci domenica."
    )
    source["target_text"] = format_audio_response_v2_target(
        heard=source["target_heard_text"], action="SPEAK", say=source["target_say_text"],
    )

    candidates, _ = build_train_repair_candidates([source])

    assert len(candidates) == 2
    assert all("percorso della domenica" in row["target_say_text"] for row in candidates)


def test_builder_only_augments_v5_training_speak_and_replan_rows():
    rows = [
        _row(action="SPEAK"),
        _row(action="REPLAN"),
        _row(action="BACKCHANNEL"),
        _row(action="SPEAK", split="validation"),
        _row(action="SPEAK", profile="audio_grounded_v4"),
        _row(action="SPEAK", source="synthetic_it_t6"),
    ]

    candidates, summary = build_train_repair_candidates(rows)

    assert len(candidates) == 4
    assert {row["action"] for row in candidates} == {"SPEAK", "REPLAN"}
    assert all(row["split"] == "train" and row["profile"] == "audio_grounded_v5" for row in candidates)
    assert summary["source_rows_by_action"] == {"SPEAK": 1, "REPLAN": 1}
    assert summary["excluded_noneligible_rows"] == 4


def test_repair_schedule_is_unique_ordered_40_30_15_15_with_v5_two_to_one_cycles():
    rows = []
    for index in range(8):
        rows.append({
            "training_example_id": f"asr-{index}", "task_mix_contract": "audio_asr_mix_v2",
            "split": "train", "task_kind": "asr", "action": None,
            "source": "synthetic_v3_clean_asr", "conversation_id": f"asr-conv-{index}",
            "source_group_id": f"asr-group-{index}", "synthetic": False,
        })
    for action, source_count, variant_source_count in (("SPEAK", 4, 2), ("REPLAN", 2, 1)):
        for index in range(source_count):
            original = {
                "training_example_id": f"{action.lower()}-orig-{index}",
                "task_mix_contract": "audio_asr_mix_v2", "split": "train",
                "task_kind": "conversation_response", "action": action,
                "source": "authored_synthetic_audio_candidate", "profile": "audio_grounded_v5",
                "conversation_id": f"{action.lower()}-conv-{index}",
                "target_utterance_id": "assistant_reply", "source_group_id": f"{action.lower()}-group-{index}",
                "target_variant_id": None, "augmentation_only": False, "synthetic": True,
            }
            rows.append(original)
            if index < variant_source_count:
                for variant_index in range(2):
                    rows.append({
                        **original,
                        "training_example_id": f"{action.lower()}-repair-{index}-{variant_index}",
                        "target_variant_id": f"repair-{variant_index}", "augmentation_only": True,
                    })
    for index in range(3):
        rows.append({
            "training_example_id": f"bc-{index}", "task_mix_contract": "audio_asr_mix_v2",
            "split": "train", "task_kind": "conversation_response", "action": "BACKCHANNEL",
            "source": "authored_synthetic_audio_candidate", "profile": "audio_grounded_v5",
            "conversation_id": f"bc-conv-{index}", "target_utterance_id": "assistant_ack",
            "source_group_id": f"bc-group-{index}", "target_variant_id": None,
            "augmentation_only": False, "synthetic": True,
        })

    schedule, summary = build_repair_schedule(rows, updates=20, seed=17)

    assert [row["task_kind"] if row["task_kind"] == "asr" else row["action"] for row in schedule[:4]] == [
        "asr", "SPEAK", "BACKCHANNEL", "REPLAN",
    ]
    assert summary["task_mix"] == {"asr": 8, "SPEAK": 6, "BACKCHANNEL": 3, "REPLAN": 3}
    assert summary["variant_rows_by_action"] == {"SPEAK": 4, "REPLAN": 2}
    assert summary["original_v5_rows_by_action"] == {"SPEAK": 2, "BACKCHANNEL": 3, "REPLAN": 1}
    assert summary["unique_training_example_ids"] == 20
