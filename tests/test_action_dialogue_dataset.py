"""Tests for the text-only native action dialogue dataset builder."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_native_action_dialogue_dataset as builder  # noqa: E402
import train_unsloth_action_qlora as trainer  # noqa: E402


def _segment(
    segment_id: str,
    channel: int,
    start: float,
    end: float,
    text: str,
    *,
    speaker: str = "A1",
    turn: str = "complete",
) -> dict:
    return {
        "id": segment_id,
        "channelIndex": channel,
        "start": start,
        "end": end,
        "text": text,
        "attributes": {"speaker": speaker, "turn": turn},
    }


def test_word_counter_handles_latin_cjk_and_annotated_text() -> None:
    assert builder.word_count("L'idea, per domani: va bene!") == 4
    assert builder.word_count("我们今天讨论社区花园的安排") == 13
    assert builder.word_count("Hello 世界") == 3


def test_speak_length_predicate_is_conservative_and_bounded() -> None:
    assert builder.is_conversational_speak("We can review the schedule and decide what to do next.")
    assert not builder.is_conversational_speak("Yes, okay.")
    assert not builder.is_conversational_speak(" ".join(["word"] * 81))
    assert builder.is_conversational_speak("我们今天讨论社区花园的安排并确认周末可用的工具")


def test_prompt_contains_only_prior_user_assistant_turns_and_exact_prefix() -> None:
    prompt = builder.make_prompt(
        [
            {"role": "user", "text": "We need to plan the event."},
            {"role": "assistant", "text": "Let's first check the available dates."},
            {"role": "system", "text": "must not appear"},
            {"role": "user", "text": "The venue is available on Friday."},
        ]
    )
    assert prompt.endswith("<start_of_turn>assistant\n")
    assert "<start_of_turn>system" not in prompt
    assert "must not appear" not in prompt
    assert "<start_of_turn>user\nWe need to plan the event.<end_of_turn>" in prompt
    assert "<start_of_turn>assistant\nLet's first check the available dates.<end_of_turn>" in prompt


def test_plan_actions_are_explicit_and_yield_requires_valid_preemption_link() -> None:
    plan = {
        "audio_generated": False,
        "dialogues": [
            {
                "conversation_id": "conv-plan-1",
                "topic_group_id": "topic-1",
                "split": "train",
                "utterances": [
                    {
                        "utterance_id": "u1",
                        "speaker": "user",
                        "text": "We should decide how to organize the community garden this week.",
                    },
                    {
                        "utterance_id": "a1",
                        "speaker": "assistant",
                        "assistant_segment_id": "segment-1",
                        "target_action": "SPEAK",
                        "text": "We can list the available tools and assign each person a specific task.",
                    },
                    {
                        "utterance_id": "u2",
                        "speaker": "user",
                        "text": "Wait, the storage room is closed tomorrow, so we need another plan.",
                        "preempts_assistant_segment_id": "segment-1",
                    },
                    {
                        "utterance_id": "a2",
                        "speaker": "assistant",
                        "assistant_segment_id": "segment-2",
                        "target_action": "YIELD",
                        "text": "This must not be emitted as a YIELD target without a link.",
                    },
                ],
            }
        ],
    }

    examples, excluded = builder.extract_plan_examples(plan)
    speak = next(row for row in examples if row["action"] == "SPEAK")
    yield_row = next(row for row in examples if row["action"] == "YIELD")

    assert len(examples) == 2
    assert speak["target_text"].startswith("We can list")
    assert "We can list the available tools" not in speak["prompt"]
    assert yield_row["target_text"] == ""
    assert yield_row["target"] == "<unused92> <end_of_turn>"
    assert "the storage room is closed tomorrow" in yield_row["prompt"]
    assert "This must not be emitted" not in yield_row["prompt"]
    assert excluded["yield_missing_or_invalid_preemption_link"] == 1
    assert all(row["source_timing"]["available"] is False for row in examples)


def test_smoothconv_speak_requires_complete_turn_timing_and_causal_history() -> None:
    segments = [
        _segment("prior-user", 0, 0.0, 1.0, "我们今天讨论社区花园的安排并确认具体任务"),
        _segment("target-assistant", 1, 1.1, 2.0, "我们先确认周末可以使用的工具和场地安排", speaker="B1"),
        _segment("future-user", 0, 3.0, 4.0, "FUTURE TRANSCRIPT MUST NOT LEAK"),
        _segment("incomplete-user", 0, 5.0, 5.5, "unfinished phrase", turn="incomplete"),
    ]
    row = {
        "source_key": "smoothconv-source-1",
        "source_group_id": "smoothconv-group-1",
        "project_split": "train",
        "source_path": "source.json",
        "annotation_local_path": "source.json",
    }
    examples, excluded = builder.build_smoothconv_examples(row, {"instances": segments}, "train")

    assert len(examples) == 1
    example = examples[0]
    assert example["action"] == "SPEAK"
    assert example["target_text"] == "我们先确认周末可以使用的工具和场地安排"
    assert example["source_timing"]["preceding_partner_gap_ms"] == 100
    assert "我们今天讨论社区花园的安排" in example["prompt"]
    assert "FUTURE TRANSCRIPT MUST NOT LEAK" not in example["prompt"]
    assert example["target_text"] not in example["prompt"]
    assert excluded["not_complete_turn"] >= 1


def test_smoothconv_turns_over_500ms_still_count_as_sequential_speak() -> None:
    prior = _segment("prior", 0, 0.0, 1.0, "我们今天讨论社区花园的安排并确认具体任务")
    at_500ms = _segment("at-500", 1, 1.5, 2.0, "我们先确认周末可以使用的工具和场地安排", speaker="B1")
    after_500ms = _segment("after-500", 1, 1.501, 2.1, "这个回答距离刚才的话已经超过规定窗口", speaker="B1")
    unknown_speaker = _segment("unknown", 1, 1.2, 2.0, "这个说话人身份未知所以不能作为样本", speaker="unknown")
    unknown_turn = _segment("unknown-turn", 1, 1.2, 2.0, "这个轮次没有标记为完整不能作为样本", speaker="B1", turn="unknown")
    row = {"source_key": "source", "source_group_id": "group", "project_split": "train"}

    examples, _ = builder.build_smoothconv_examples(
        row, {"instances": [prior, at_500ms, after_500ms, unknown_speaker, unknown_turn]}, "train"
    )
    assert [item["source_metadata"]["target_segment_id"] for item in examples] == ["at-500", "after-500"]


def test_group_split_uses_declared_split_only_when_complete_and_consistent() -> None:
    groups = {
        "declared-train": ["train", "train"],
        "declared-validation": ["validation", "validation"],
        "missing-assignment": ["train", None],
        "conflict": ["train", "validation"],
    }
    assigned, report = builder.assign_group_splits(groups)
    assigned_again, report_again = builder.assign_group_splits(groups)

    assert assigned == assigned_again
    assert report == report_again
    assert all(assigned[group] == builder._stable_split(group) for group in groups)
    assert report["deterministically_assigned_group_count"] == len(groups)


def test_stable_split_uses_the_full_sha256_digest() -> None:
    import hashlib

    group_id = next(
        f"group-{index}"
        for index in range(10000)
        if int.from_bytes(hashlib.sha256(f"group-{index}".encode()).digest(), "big") % 5
        != int.from_bytes(hashlib.sha256(f"group-{index}".encode()).digest()[:8], "big") % 5
    )
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    expected = "validation" if int.from_bytes(digest, "big") % 5 == 0 else "train"

    assert builder._stable_split(group_id) == expected


def test_smoothconv_temporal_rules_emit_all_three_active_actions() -> None:
    segments = [
        _segment("short-user", 0, 0.0, 1.0, "We need to plan the garden schedule for Saturday."),
        _segment("ack", 1, 0.4, 0.8, "Mhm.", speaker="B1"),
        _segment(
            "speak-user",
            0,
            2.0,
            2.5,
            "The volunteers can meet near the entrance after lunch.",
        ),
        _segment(
            "speak",
            1,
            2.7,
            4.0,
            "We can prepare a detailed schedule and assign everyone a clear task.",
            speaker="B1",
        ),
        _segment(
            "long-overlap-user",
            0,
            5.0,
            8.5,
            "We have to reconsider the route because the gate is closed for repairs.",
        ),
        _segment(
            "replan",
            1,
            5.5,
            8.2,
            "We should revise the plan and use the side entrance while repairs continue.",
            speaker="B1",
        ),
    ]
    row = {"source_key": "smooth-actions", "source_group_id": "smooth-actions"}

    examples, _ = builder.build_smoothconv_examples(row, {"instances": segments}, "train")

    assert {item["action"] for item in examples} == {"SPEAK", "BACKCHANNEL", "REPLAN"}
    assert next(item for item in examples if item["action"] == "BACKCHANNEL")["target_text"] == "Mhm."
    assert next(item for item in examples if item["action"] == "REPLAN")["source_timing"]["overlap_ms"] > 1800


def test_duplexconv_temporal_examples_use_track_text_and_conversation_group() -> None:
    row = {
        "source_dataset": "DuplexConv",
        "source_conversation_group_id": "duplex-session-1",
        "source_events": [
            {
                "source_event_ordinal": 0,
                "source_track_index": 0,
                "source_start_ms": 0,
                "source_end_ms": 900,
                "source_state": "<|complete|>",
                "source_annotation_utterance": {"labels": {"txt": "We need to plan the garden schedule."}},
            },
            {
                "source_event_ordinal": 1,
                "source_track_index": 1,
                "source_start_ms": 500,
                "source_end_ms": 1200,
                "source_state": "<|backchannel|>",
                "source_annotation_utterance": {"labels": {"txt": "Mhm."}},
            },
            {
                "source_event_ordinal": 2,
                "source_track_index": 0,
                "source_start_ms": 1800,
                "source_end_ms": 2200,
                "source_state": "<|complete|>",
                "source_annotation_utterance": {
                    "labels": {"txt": "The volunteers can meet near the garden entrance after lunch today."}
                },
            },
            {
                "source_event_ordinal": 3,
                "source_track_index": 1,
                "source_start_ms": 2400,
                "source_end_ms": 4200,
                "source_state": "<|complete|>",
                "source_annotation_utterance": {
                    "labels": {"txt": "We can prepare a detailed schedule and assign everyone a clear task."}
                },
            },
            {
                "source_event_ordinal": 4,
                "source_track_index": 0,
                "source_start_ms": 5000,
                "source_end_ms": 9000,
                "source_state": "<|complete|>",
                "source_annotation_utterance": {
                    "labels": {"txt": "We must reconsider the route because the garden gate is closed for repairs."}
                },
            },
            {
                "source_event_ordinal": 5,
                "source_track_index": 1,
                "source_start_ms": 5500,
                "source_end_ms": 8800,
                "source_state": "<|complete|>",
                "source_annotation_utterance": {
                    "labels": {"txt": "We should revise the plan and use the side entrance while repairs continue."}
                },
            },
        ],
    }

    examples, _ = builder.build_duplexconv_examples(row, "train")

    assert {item["action"] for item in examples} == {"SPEAK", "BACKCHANNEL", "REPLAN"}
    assert {item["source_group_id"] for item in examples} == {"DuplexConv:duplex-session-1"}
    backchannel = next(item for item in examples if item["action"] == "BACKCHANNEL")
    assert backchannel["target_text"] == "Mhm."
    assert backchannel["source_metadata"]["source_state"] == "<|backchannel|>"
    replan = next(item for item in examples if item["action"] == "REPLAN")
    assert replan["source_timing"]["overlap_ms"] > 1800


class _FakeActionTokenizer:
    bos_token_id = 1
    eos_token_id = 102
    pad_token_id = 0

    def __init__(self) -> None:
        self.special = {
            "<|turn>": 100,
            "<turn|>": 102,
            **{f"<unused{index}>": 256000 + index - 88 for index in range(89, 95)},
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        result: list[int] = []
        offset = 0
        choices = sorted(self.special, key=len, reverse=True)
        while offset < len(text):
            match = next((token for token in choices if text.startswith(token, offset)), None)
            if match is None:
                result.append(ord(text[offset]) + 1000)
                offset += 1
            else:
                result.append(self.special[match])
                offset += len(match)
        return result

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special.get(token, 3)

    def add_special_tokens(self, *args: object, **kwargs: object) -> int:
        return 0


def test_training_labels_mask_prompt_and_keep_control_token_and_turn_end() -> None:
    tokenizer = _FakeActionTokenizer()
    processor = type("FakeProcessor", (), {"tokenizer": tokenizer})()
    row = {
        "prompt": "<start_of_turn>user\nLet's plan the garden.<end_of_turn>\n<start_of_turn>assistant\n",
        "target": "<unused93> Mhm. <end_of_turn>",
        "action_token": "<unused93>",
        "token_id": 256005,
    }

    assert trainer.get_text_tokenizer(processor) is tokenizer
    prepared = trainer.encode_record(row, trainer.get_text_tokenizer(processor), max_seq_length=64)

    assert trainer.normalize_for_gemma4(row["prompt"]).endswith("<|turn>model\n")
    assert prepared["prompt_length"] < len(prepared["input_ids"])
    assert prepared["labels"][: prepared["prompt_length"]] == [-100] * prepared["prompt_length"]
    assert prepared["labels"][prepared["prompt_length"]] == 256005
    assert prepared["labels"][-1] == tokenizer.eos_token_id


def test_speak_examples_emitted_by_extractors_always_satisfy_length_predicate() -> None:
    plan = {
        "audio_generated": False,
        "dialogues": [
            {
                "conversation_id": "conv-length",
                "topic_group_id": "topic-length",
                "split": "train",
                "utterances": [
                    {"speaker": "user", "text": "We need to plan the community garden this week."},
                    {
                        "speaker": "assistant",
                        "assistant_segment_id": "long-speak",
                        "target_action": "SPEAK",
                        "text": "We can prepare a detailed schedule and assign each volunteer a clear task for Saturday.",
                    },
                    {
                        "speaker": "assistant",
                        "assistant_segment_id": "short-speak",
                        "target_action": "SPEAK",
                        "text": "Yes, okay.",
                    },
                ],
            }
        ],
    }
    examples, excluded = builder.extract_plan_examples(plan)
    speak = [row for row in examples if row["action"] == "SPEAK"]

    assert len(speak) == 1
    assert all(builder.is_conversational_speak(row["target_text"]) for row in speak)
    assert excluded["speak_length_outside_8_80"] == 1


def test_native_action_tokens_round_trip_with_actual_gemma_tokenizer() -> None:
    import transformers
    from transformers import AutoTokenizer

    assert transformers.__version__.startswith("5.")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    tokenizer.add_special_tokens({"additional_special_tokens": list(builder.TOKEN_ID_TO_STRING.values())})

    native_source = ROOT / "training" / "duplex_qlora" / "native_control6.py"
    tree = ast.parse(native_source.read_text(encoding="utf-8"))
    maps = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"ACTION_TO_TOKEN_ID", "TOKEN_ID_TO_STRING"}:
                maps[node.target.id] = ast.literal_eval(node.value)
    native_action_to_id = maps["ACTION_TO_TOKEN_ID"]
    native_id_to_token = maps["TOKEN_ID_TO_STRING"]

    assert builder.ACTION_TO_TOKEN_ID == native_action_to_id
    assert builder.TOKEN_ID_TO_STRING == native_id_to_token
    for action, expected_id in native_action_to_id.items():
        token = native_id_to_token[expected_id]
        encoded = tokenizer.encode(token, add_special_tokens=False)
        converted_id = tokenizer.convert_tokens_to_ids(token)
        assert tokenizer.unk_token_id != expected_id
        assert encoded != [tokenizer.unk_token_id]
        assert encoded == [expected_id]
        assert converted_id == expected_id
        assert tokenizer.convert_ids_to_tokens(expected_id) == token
        assert tokenizer.decode([expected_id], skip_special_tokens=False) == token


def test_jsonl_rows_expose_auditable_source_and_split_fields() -> None:
    required = {
        "example_id",
        "conversation_id",
        "source_group_id",
        "split",
        "source",
        "action",
        "action_token",
        "token_id",
        "prompt",
        "target",
        "target_text",
        "source_timing",
        "source_metadata",
    }
    plan = {
        "audio_generated": False,
        "dialogues": [
            {
                "conversation_id": "conv-shape",
                "topic_group_id": "topic-shape",
                "split": "train",
                "utterances": [
                    {"speaker": "user", "text": "We need to organize a community garden this week."},
                    {
                        "speaker": "assistant",
                        "assistant_segment_id": "shape-speak",
                        "target_action": "SPEAK",
                        "text": "We can list the available tools and assign each volunteer a clear task.",
                    },
                ],
            }
        ],
    }
    examples, _ = builder.extract_plan_examples(plan)
    row = examples[0]

    assert required <= set(row)
    assert row["split"] == "train"
    assert row["source_group_id"]
    assert row["conversation_id"] == "conv-shape"
    assert json.loads(json.dumps(row, ensure_ascii=False)) == row
