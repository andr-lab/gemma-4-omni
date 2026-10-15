"""Build an auditable candidate dataset for text-conditioned Gemma actions.

Explicit plan labels and temporal labels derived from source annotations keep
their provenance. Derived labels are hypotheses, not human adjudications.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ACTION_TO_TOKEN_ID: dict[str, int] = {
    "LISTEN": 256001,
    "SPEAK": 256002,
    "CONTINUE": 256003,
    "YIELD": 256004,
    "BACKCHANNEL": 256005,
    "REPLAN": 256006,
}
TOKEN_ID_TO_STRING: dict[int, str] = {
    256001: "<unused89>",
    256002: "<unused90>",
    256003: "<unused91>",
    256004: "<unused92>",
    256005: "<unused93>",
    256006: "<unused94>",
}

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "action_dialogue_v1"
DEFAULT_PLAN_CANDIDATES = (
    ROOT / "data" / "synthetic_it_t6_audio_grounded_v5" / "dialogue_plan.json",
    ROOT / "data" / "synthetic_it_t6_audio_grounded_v5_candidate_seed260928" / "dialogue_plan.json",
    ROOT / "data" / "synthetic_it_t6_audio_grounded_v5_smoke_seed260928" / "dialogue_plan.json",
)
DEFAULT_SMOOTHCONV_MANIFEST = (
    ROOT
    / "data"
    / "hf_slices"
    / "smoothconv_pinned_20260930_full40h"
    / "smoothconv.slice.jsonl"
)
DEFAULT_DUPLEXCONV_EVIDENCE = (
    ROOT
    / "data"
    / "training_gates"
    / "chinese_label_evidence_duplexconv_20260929_v1.jsonl"
)
MIN_SPEAK_WORDS = 8
MAX_SPEAK_WORDS = 80
MAX_CONTEXT_TURNS = 12
_ACTION_NAMES = frozenset(ACTION_TO_TOKEN_ID)
_LATIN_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
        or 0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _recover_mojibake(text: str) -> str:
    """Recover common UTF-8-as-Windows-1252 fixtures when it is unambiguous."""
    original_cjk = sum(_is_cjk(char) for char in text)
    best = text
    best_cjk = original_cjk
    for encoding in ("cp1252", "latin-1"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        candidate_cjk = sum(_is_cjk(char) for char in candidate)
        if candidate_cjk > best_cjk:
            best, best_cjk = candidate, candidate_cjk
    return best


def word_count(text: str) -> int:
    """Count Latin words and each CJK character separately."""
    if not isinstance(text, str):
        return 0
    text = _recover_mojibake(text)
    cjk_count = sum(_is_cjk(char) for char in text)
    non_cjk = "".join(" " if _is_cjk(char) else char for char in text)
    latin_words = _LATIN_WORD_RE.findall(non_cjk)
    # Short Latin function words and single-letter fragments are deliberately
    # ignored by this conservative dialogue-length measure.
    return cjk_count + sum(
        1
        for word in latin_words
        if any(len(part) >= 3 for part in re.split(r"['\u2019]", word))
    )


def is_conversational_speak(text: str) -> bool:
    count = word_count(text)
    return MIN_SPEAK_WORDS <= count <= MAX_SPEAK_WORDS


def make_prompt(turns: Iterable[Mapping[str, Any]]) -> str:
    """Render causal user/assistant history with Gemma's exact turn markers."""
    rendered: list[str] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        role = turn.get("role")
        text = turn.get("text")
        if not isinstance(role, str) or role.casefold() not in {"user", "assistant"}:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        rendered.append(
            f"<start_of_turn>{role.casefold()}\n{text.strip()}<end_of_turn>\n"
        )
    return "".join(rendered) + "<start_of_turn>assistant\n"


def _stable_split(group_id: str) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return "validation" if int.from_bytes(digest, "big") % 5 == 0 else "train"


def assign_group_splits(
    groups: Mapping[str, Sequence[str | None]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign every source group by the complete SHA-256 digest modulo five."""
    assigned: dict[str, str] = {}
    for raw_group_id in sorted(groups, key=str):
        group_id = str(raw_group_id).strip()
        if not group_id:
            raise ValueError("group ids must be non-empty strings")
        assigned[group_id] = _stable_split(group_id)

    report = {
        "group_count": len(assigned),
        "deterministically_assigned_group_count": len(assigned),
        "assigned_group_counts": dict(sorted(Counter(assigned.values()).items())),
        "policy": "int.from_bytes(sha256(group_id), big) modulo 5; 0=validation, otherwise train",
    }
    return assigned, report


def _example_id(source: str, conversation_id: str, target_id: str, action: str) -> str:
    material = "\x1f".join((source, conversation_id, target_id, action))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _target(action: str, target_text: str) -> str:
    action_token = TOKEN_ID_TO_STRING[ACTION_TO_TOKEN_ID[action]]
    if target_text:
        return f"{action_token} {target_text} <end_of_turn>"
    return f"{action_token} <end_of_turn>"


def _source_timing_from_plan(
    utterance: Mapping[str, Any], *, audio_generated: bool
) -> dict[str, Any]:
    if not audio_generated:
        return {"available": False}
    start = utterance.get("start_s", utterance.get("start"))
    end = utterance.get("end_s", utterance.get("end"))
    try:
        start_f, end_f = float(start), float(end)
    except (TypeError, ValueError):
        return {"available": False}
    if not math.isfinite(start_f) or not math.isfinite(end_f) or end_f <= start_f:
        return {"available": False}
    return {"available": True, "start_s": start_f, "end_s": end_f}


def _dialogue_group_id(dialogue: Mapping[str, Any]) -> str:
    value = dialogue.get("topic_group_id") or dialogue.get("source_group_id")
    if not isinstance(value, str) or not value.strip():
        value = dialogue.get("conversation_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("plan dialogue is missing a stable conversation/group id")
    return value.strip()


def extract_plan_examples(
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract explicitly labeled assistant actions from a dialogue plan."""
    if not isinstance(plan, Mapping) or not isinstance(plan.get("dialogues"), list):
        raise ValueError("dialogue plan must be an object with a dialogues list")

    excluded: Counter[str] = Counter()
    usable_dialogues: list[Mapping[str, Any]] = []
    group_declarations: dict[str, list[str | None]] = {}
    for ordinal, dialogue in enumerate(plan["dialogues"]):
        if not isinstance(dialogue, Mapping):
            raise ValueError(f"dialogues[{ordinal}] must be an object")
        group_id = _dialogue_group_id(dialogue)
        group_declarations.setdefault(group_id, []).append(None)
        usable_dialogues.append(dialogue)
    group_splits, _ = assign_group_splits(group_declarations)

    examples: list[dict[str, Any]] = []
    for dialogue_ordinal, dialogue in enumerate(usable_dialogues):
        conversation_id = dialogue.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError(f"usable dialogue {dialogue_ordinal} is missing conversation_id")
        conversation_id = conversation_id.strip()
        group_id = _dialogue_group_id(dialogue)
        utterances = dialogue.get("utterances")
        if not isinstance(utterances, list):
            raise ValueError(f"dialogue {conversation_id!r} must contain an utterances list")

        history: list[dict[str, str]] = []
        known_assistant_segments: set[str] = set()
        valid_preemption: tuple[str, str] | None = None
        for utterance_ordinal, utterance in enumerate(utterances):
            if not isinstance(utterance, Mapping):
                raise ValueError(
                    f"dialogue {conversation_id!r} utterance {utterance_ordinal} must be an object"
                )
            speaker_value = utterance.get("speaker")
            speaker = speaker_value.strip().casefold() if isinstance(speaker_value, str) else ""
            text_value = utterance.get("text")
            text = text_value.strip() if isinstance(text_value, str) else ""
            raw_action = utterance.get("target_action")
            action = raw_action.strip().upper() if isinstance(raw_action, str) else ""
            utterance_id_value = utterance.get("utterance_id")
            utterance_id = (
                utterance_id_value.strip()
                if isinstance(utterance_id_value, str) and utterance_id_value.strip()
                else f"utterance-{utterance_ordinal}"
            )

            if speaker == "user":
                preempts = utterance.get("preempts_assistant_segment_id")
                if isinstance(preempts, str) and preempts.strip():
                    linked_segment = preempts.strip()
                    if linked_segment in known_assistant_segments:
                        valid_preemption = (linked_segment, utterance_id)
                    else:
                        valid_preemption = None
                        excluded["invalid_preemption_reference"] += 1

            if speaker == "assistant":
                if raw_action is None or (isinstance(raw_action, str) and not raw_action.strip()):
                    excluded["assistant_missing_action_target"] += 1
                elif action not in _ACTION_NAMES:
                    excluded["unknown_action_target"] += 1
                else:
                    assistant_segment_value = utterance.get("assistant_segment_id")
                    assistant_segment = (
                        assistant_segment_value.strip()
                        if isinstance(assistant_segment_value, str)
                        and assistant_segment_value.strip()
                        else utterance_id
                    )
                    target_text = text
                    link_metadata: dict[str, Any] = {"preemption_link_valid": None}
                    if action == "SPEAK" and not is_conversational_speak(target_text):
                        excluded["speak_length_outside_8_80"] += 1
                    elif action == "YIELD":
                        if valid_preemption is None:
                            excluded["yield_missing_or_invalid_preemption_link"] += 1
                        else:
                            target_text = ""
                            link_metadata = {
                                "preemption_link_valid": True,
                                "preempted_assistant_segment_id": valid_preemption[0],
                                "preemption_utterance_id": valid_preemption[1],
                            }
                            # The YIELD control token is the target; any accompanying
                            # assistant prose is intentionally omitted from supervision.
                            if text:
                                excluded["yield_missing_or_invalid_preemption_link"] += 1
                    if (
                        (action != "YIELD" or valid_preemption is not None)
                        and (action != "SPEAK" or is_conversational_speak(text))
                    ):
                        prompt = make_prompt(history)
                        action_token = TOKEN_ID_TO_STRING[ACTION_TO_TOKEN_ID[action]]
                        examples.append(
                            {
                                "example_id": _example_id(
                                    "plan", conversation_id, utterance_id, action
                                ),
                                "conversation_id": conversation_id,
                                "source_group_id": group_id,
                                "split": group_splits[group_id],
                                "source": str(
                                    dialogue.get("source")
                                    or plan.get("source")
                                    or "synthetic_dialogue_plan"
                                ),
                                "action": action,
                                "action_token": action_token,
                                "token_id": ACTION_TO_TOKEN_ID[action],
                                "prompt": prompt,
                                "target": _target(action, target_text),
                                "target_text": target_text,
                                "source_timing": _source_timing_from_plan(
                                    utterance, audio_generated=plan.get("audio_generated") is True
                                ),
                                "source_metadata": {
                                    "input_kind": "dialogue_plan",
                                    "dialogue_ordinal": dialogue_ordinal,
                                    "utterance_ordinal": utterance_ordinal,
                                    "utterance_id": utterance_id,
                                    "assistant_segment_id": assistant_segment,
                                    "template_id": dialogue.get("template_id"),
                                    "topic_group_id": dialogue.get("topic_group_id"),
                                    "declared_split": dialogue.get("split"),
                                    **link_metadata,
                                },
                            }
                        )
                # A linked user interruption can supervise only the next
                # assistant decision; it must not leak into later turns.
                valid_preemption = None
                segment_value = utterance.get("assistant_segment_id")
                if isinstance(segment_value, str) and segment_value.strip():
                    known_assistant_segments.add(segment_value.strip())

            if speaker in {"user", "assistant"} and text:
                history.append({"role": speaker, "text": text})

    return examples, dict(sorted(excluded.items()))


def _valid_speaker(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().casefold() not in {"unknown", "none", "null", "n/a"}
    )


def _finite_time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _clean_transcript_text(value: Any) -> tuple[str, bool]:
    raw = value.strip() if isinstance(value, str) else ""
    has_pause = "<pause>" in raw.casefold()
    text = re.sub(r"<[^>]*>", " ", raw)
    return re.sub(r"\s+", " ", text).strip(), has_pause


def _reformulates(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    left = previous["text"].casefold()
    right = current["text"].casefold()
    latin_left = {word for word in _LATIN_WORD_RE.findall(left) if len(word) >= 3}
    latin_right = {word for word in _LATIN_WORD_RE.findall(right) if len(word) >= 3}
    if latin_left & latin_right:
        return True
    cjk_left = "".join(char for char in left if _is_cjk(char))
    cjk_right = "".join(char for char in right if _is_cjk(char))
    bigrams_left = {cjk_left[index : index + 2] for index in range(len(cjk_left) - 1)}
    bigrams_right = {cjk_right[index : index + 2] for index in range(len(cjk_right) - 1)}
    return bool(bigrams_left & bigrams_right)


def _make_timed_examples(
    *,
    source: str,
    conversation_id: str,
    group_id: str,
    split: str,
    events: Sequence[Mapping[str, Any]],
    excluded: Counter[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ordered = sorted(events, key=lambda item: (item["start"], item["end"], item["ordinal"]))
    for target in ordered:
        if target["channel"] not in {0, 1}:
            excluded["invalid_channel"] += 1
            continue
        if target["turn"] not in {"complete", "backchannel", "<|complete|>", "<|backchannel|>"}:
            excluded["not_complete_turn"] += 1
            continue
        if not target["text"]:
            excluded["empty_text"] += 1
            continue

        prior = [item for item in ordered if item["end"] <= target["start"] + 1e-9 and item["text"]]
        prior_partner = [item for item in prior if item["channel"] != target["channel"]]
        overlaps = [
            item
            for item in ordered
            if item["channel"] != target["channel"]
            and item["start"] < target["end"] - 1e-9
            and item["end"] > target["start"] + 1e-9
        ]
        overlap_seconds = max(
            (
                min(target["end"], item["end"])
                - max(target["start"], item["start"])
                for item in overlaps
            ),
            default=0.0,
        )
        previous_same = [item for item in prior if item["channel"] == target["channel"]]
        previous_same_event = max(
            previous_same,
            key=lambda item: (item["end"], item["start"], item["ordinal"]),
            default=None,
        )
        reformulation = bool(
            previous_same_event
            and (
                "incomplete" in str(previous_same_event["turn"]).casefold()
                or previous_same_event["has_pause"]
            )
            and _reformulates(previous_same_event, target)
        )
        words = word_count(target["text"])
        source_state = str(target.get("source_state") or target["turn"])
        if reformulation:
            action, reason = "REPLAN", "same_channel_incomplete_or_pause_reformulation"
        elif "backchannel" in source_state.casefold() and 1 <= words <= 4:
            action, reason = "BACKCHANNEL", "source_annotation_backchannel_state"
        elif overlaps and (overlap_seconds <= 1.8 or words <= 4):
            if 1 <= words <= 4:
                action, reason = "BACKCHANNEL", "short_overlap_or_short_utterance"
            else:
                excluded["backchannel_target_outside_1_4_words"] += 1
                continue
        elif overlaps:
            action, reason = "REPLAN", "prolonged_overlap"
        elif prior_partner:
            if is_conversational_speak(target["text"]):
                action, reason = "SPEAK", "sequential_alternating_turn"
            else:
                excluded["speak_length_outside_8_80"] += 1
                continue
        else:
            excluded["missing_causal_partner_history"] += 1
            continue

        history = [
            {
                "role": "assistant" if item["channel"] == target["channel"] else "user",
                "text": item["text"],
            }
            for item in prior[-MAX_CONTEXT_TURNS:]
        ]
        context_ids = [item["id"] for item in prior[-MAX_CONTEXT_TURNS:]]
        action_token = TOKEN_ID_TO_STRING[ACTION_TO_TOKEN_ID[action]]
        target_text = target["text"]
        examples.append(
            {
                "example_id": _example_id(source, conversation_id, target["id"], action),
                "conversation_id": conversation_id,
                "source_group_id": group_id,
                "split": split,
                "source": source,
                "action": action,
                "action_token": action_token,
                "token_id": ACTION_TO_TOKEN_ID[action],
                "prompt": make_prompt(history),
                "target": _target(action, target_text),
                "target_text": target_text,
                "label_provenance": "derived_temporal_heuristic_v1",
                "source_timing": {
                    "available": True,
                    "target_start_s": target["start"],
                    "target_end_s": target["end"],
                    "target_duration_ms": int(round((target["end"] - target["start"]) * 1000)),
                    "overlap_ms": int(round(overlap_seconds * 1000)),
                    "preceding_partner_gap_ms": (
                        int(round((target["start"] - max(item["end"] for item in prior_partner)) * 1000))
                        if prior_partner
                        else None
                    ),
                },
                "source_metadata": {
                    "input_kind": f"{source.casefold()}_dual_channel_annotation",
                    "target_event_id": target["id"],
                    "target_segment_id": target["id"],
                    "target_channel_index": target["channel"],
                    "target_speaker": target.get("speaker"),
                    "target_turn": target["turn"],
                    "source_state": target.get("source_state"),
                    "label_reason": reason,
                    "label_status": "heuristic_unadjudicated",
                    "context_event_ids": context_ids,
                    "source_metadata": target.get("metadata", {}),
                },
            }
        )
    return examples


def build_smoothconv_examples(
    row: Mapping[str, Any],
    annotation: Mapping[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build bidirectional examples using complete SmoothConv timed instances."""
    if split not in {"train", "validation"}:
        raise ValueError(f"SmoothConv split must be train or validation, got {split!r}")
    if not isinstance(annotation, Mapping) or not isinstance(annotation.get("instances"), list):
        raise ValueError("SmoothConv annotation must be an object with an instances list")
    source_key = row.get("source_key")
    group_id = row.get("source_group_id")
    if not isinstance(source_key, str) or not source_key.strip():
        raise ValueError("SmoothConv manifest row is missing source_key")
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("SmoothConv manifest row is missing source_group_id")
    source_key, group_id = source_key.strip(), group_id.strip()

    excluded: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    for ordinal, instance in enumerate(annotation["instances"]):
        if not isinstance(instance, Mapping):
            raise ValueError(f"SmoothConv instances[{ordinal}] must be an object")
        start = _finite_time(instance.get("start"), f"instances[{ordinal}].start")
        end = _finite_time(instance.get("end"), f"instances[{ordinal}].end")
        if start < 0 or end <= start:
            raise ValueError(f"SmoothConv instances[{ordinal}] has invalid timing")
        channel_value = instance.get("channelIndex")
        channel = (
            channel_value
            if isinstance(channel_value, int) and not isinstance(channel_value, bool)
            else None
        )
        attributes = instance.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        speaker = attributes.get("speaker")
        turn = attributes.get("turn")
        text_value = instance.get("text")
        text = text_value.strip() if isinstance(text_value, str) else ""
        segment_id_value = instance.get("id")
        segment_id = (
            segment_id_value.strip()
            if isinstance(segment_id_value, str) and segment_id_value.strip()
            else f"segment-{ordinal}"
        )
        text, has_pause = _clean_transcript_text(text)
        normalized.append(
            {
                "ordinal": ordinal,
                "id": segment_id,
                "channel": channel,
                "start": start,
                "end": end,
                "speaker": speaker.strip() if isinstance(speaker, str) else None,
                "turn": turn.strip().casefold() if isinstance(turn, str) else None,
                "text": text,
                "has_pause": has_pause,
                "source_state": turn,
                "metadata": {
                    "source_key": source_key,
                    "source_path": row.get("source_path"),
                    "annotation_local_path": row.get("annotation_local_path"),
                    "target_speaker": speaker,
                },
            }
        )

    for item in normalized:
        if not _valid_speaker(item["speaker"]):
            item["turn"] = "invalid_speaker"
            excluded["invalid_speaker"] += 1
    examples = _make_timed_examples(
        source="SmoothConv",
        conversation_id=source_key,
        group_id=group_id,
        split=split,
        events=normalized,
        excluded=excluded,
    )
    return examples, dict(sorted(excluded.items()))


def build_duplexconv_examples(
    row: Mapping[str, Any], split: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build auditable heuristic examples from one DuplexConv conversation row."""
    if split not in {"train", "validation"}:
        raise ValueError(f"DuplexConv split must be train or validation, got {split!r}")
    events_value = row.get("source_events")
    if not isinstance(events_value, list):
        raise ValueError("DuplexConv row must contain a source_events list")
    raw_group = row.get("source_conversation_group_id") or row.get("cross_dataset_source_group_id")
    if not isinstance(raw_group, str) or not raw_group.strip():
        return [], {"missing_source_conversation_group_id": 1}
    raw_group = raw_group.strip()
    group_id = f"DuplexConv:{raw_group}"
    conversation_id = raw_group
    excluded: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    for ordinal, event in enumerate(events_value):
        if not isinstance(event, Mapping):
            excluded["invalid_event"] += 1
            continue
        channel = event.get("source_track_index")
        if not isinstance(channel, int) or isinstance(channel, bool) or channel not in {0, 1}:
            excluded["invalid_channel"] += 1
            continue
        try:
            start = _finite_time(event.get("source_start_ms"), "source_start_ms") / 1000.0
            end = _finite_time(event.get("source_end_ms"), "source_end_ms") / 1000.0
        except ValueError:
            excluded["invalid_timing"] += 1
            continue
        if start < 0 or end <= start:
            excluded["invalid_timing"] += 1
            continue
        nested = event.get("source_annotation_utterance")
        nested = nested if isinstance(nested, Mapping) else {}
        labels = nested.get("labels")
        labels = labels if isinstance(labels, Mapping) else {}
        llm_labels = event.get("source_llm_labels")
        llm_labels = llm_labels if isinstance(llm_labels, Mapping) else {}
        text_value = event.get("source_text")
        if not isinstance(text_value, str) or not text_value.strip():
            text_value = labels.get("txt", llm_labels.get("txt"))
        text, has_pause = _clean_transcript_text(text_value)
        state = event.get("source_state") or nested.get("state") or "missing"
        event_id = event.get("source_event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            event_id = f"{raw_group}:{channel}:{ordinal}:{int(start * 1000)}:{int(end * 1000)}"
        normalized.append(
            {
                "ordinal": ordinal,
                "id": event_id.strip(),
                "channel": channel,
                "start": start,
                "end": end,
                "speaker": None,
                "turn": str(state).strip().casefold(),
                "source_state": str(state),
                "text": text,
                "has_pause": has_pause,
                "metadata": {
                    "source_event_ordinal": event.get("source_event_ordinal", ordinal),
                    "source_revision_status": row.get("source_revision_status"),
                    "source_adjudication_status": row.get("source_adjudication_status"),
                    "training_eligible": row.get("training_eligible"),
                    "action_loss_eligible": event.get("action_loss_eligible"),
                    "transcript_field": (
                        "source_text"
                        if isinstance(event.get("source_text"), str) and event["source_text"].strip()
                        else "source_annotation_utterance.labels.txt"
                    ),
                },
            }
        )
    examples = _make_timed_examples(
        source="DuplexConv",
        conversation_id=conversation_id,
        group_id=group_id,
        split=split,
        events=normalized,
        excluded=excluded,
    )
    return examples, dict(sorted(excluded.items()))


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open JSONL input {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(item)
    return rows


def _find_plan() -> Path:
    for path in DEFAULT_PLAN_CANDIDATES:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in DEFAULT_PLAN_CANDIDATES)
    raise FileNotFoundError(f"no dialogue_plan.json found; searched: {searched}")


def _resolve_manifest_asset(manifest_dir: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"SmoothConv manifest row is missing {field}")
    candidate = (manifest_dir / relative).resolve()
    root = manifest_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"SmoothConv asset path escapes manifest directory: {relative!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"SmoothConv manifest asset declared by {field} is missing: {candidate}")
    return candidate


def _heldout_smoothconv_group_ids() -> tuple[set[str], set[str]]:
    """Read existing held-out identities so training rows cannot reuse them."""
    path = ROOT / "data" / "training_gates" / "t1_source_records.jsonl"
    if not path.is_file():
        return set(), set()
    source_keys: set[str] = set()
    base_groups: set[str] = set()
    for row in _read_jsonl(path):
        if row.get("source") != "SmoothConv-Eval-v1":
            continue
        conversation_id = row.get("source_conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            continue
        source_keys.add(conversation_id.strip())
        match = re.match(r"^(.+?)_seg\d+(?:_|$)", conversation_id.strip())
        if match:
            base_groups.add(match.group(1))
    return source_keys, base_groups


def _smoothconv_base_group(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    match = re.match(r"^(.+?)_seg\d+(?:_|$)", value)
    return match.group(1) if match else value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    plan_path = _find_plan()
    if not DEFAULT_SMOOTHCONV_MANIFEST.is_file():
        raise FileNotFoundError(
            f"pinned SmoothConv manifest is missing: {DEFAULT_SMOOTHCONV_MANIFEST}"
        )
    if not DEFAULT_DUPLEXCONV_EVIDENCE.is_file():
        raise FileNotFoundError(
            f"DuplexConv evidence file is missing: {DEFAULT_DUPLEXCONV_EVIDENCE}"
        )
    plan = _read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError(f"dialogue plan must be a JSON object: {plan_path}")
    plan_examples, plan_excluded = extract_plan_examples(plan)

    manifest_path = DEFAULT_SMOOTHCONV_MANIFEST
    smooth_rows = _read_jsonl(manifest_path)
    if not smooth_rows:
        raise ValueError(f"SmoothConv manifest contains no rows: {manifest_path}")
    heldout_keys, heldout_groups = _heldout_smoothconv_group_ids()
    eligible_rows: list[tuple[dict[str, Any], Path]] = []
    excluded_smooth: Counter[str] = Counter()
    for ordinal, row in enumerate(smooth_rows):
        source_key = row.get("source_key")
        source_group = row.get("source_group_id")
        if not isinstance(source_key, str) or not source_key.strip():
            raise ValueError(f"SmoothConv manifest row {ordinal} is missing source_key")
        if not isinstance(source_group, str) or not source_group.strip():
            raise ValueError(f"SmoothConv manifest row {ordinal} is missing source_group_id")
        base_group = _smoothconv_base_group(source_key)
        if source_key in heldout_keys or (base_group and base_group in heldout_groups):
            excluded_smooth["held_out_evaluation_source_group"] += 1
            continue
        if row.get("evaluation_excluded") is True:
            excluded_smooth["manifest_evaluation_exclusion"] += 1
            continue
        annotation_path = _resolve_manifest_asset(
            manifest_path.parent, row.get("annotation_local_path"), "annotation_local_path"
        )
        eligible_rows.append((row, annotation_path))
    if not eligible_rows:
        raise ValueError("no eligible SmoothConv rows remain after held-out source exclusions")

    declarations: dict[str, list[str | None]] = {}
    for row, _annotation_path in eligible_rows:
        group_id = row["source_group_id"].strip()
        declarations.setdefault(group_id, []).append(None)
    duplex_rows = _read_jsonl(DEFAULT_DUPLEXCONV_EVIDENCE)
    excluded_duplex: Counter[str] = Counter()
    for ordinal, row in enumerate(duplex_rows):
        group_value = row.get("source_conversation_group_id") or row.get("cross_dataset_source_group_id")
        if not isinstance(group_value, str) or not group_value.strip():
            excluded_duplex["missing_source_conversation_group_id"] += 1
            continue
        group_id = f"DuplexConv:{group_value.strip()}"
        declarations.setdefault(group_id, []).append(None)
    # Use one assignment pass so every conversation remains in one split.
    plan_dialogues = [
        dialogue
        for dialogue in plan.get("dialogues", [])
        if isinstance(dialogue, Mapping)
    ]
    for dialogue in plan_dialogues:
        group_id = _dialogue_group_id(dialogue)
        declarations.setdefault(group_id, []).append(None)
    group_splits, split_report = assign_group_splits(declarations)

    for row in plan_examples:
        row["split"] = group_splits[row["source_group_id"]]
        row["label_provenance"] = "explicit_dialogue_plan_target"

    smooth_examples: list[dict[str, Any]] = []
    for row, annotation_path in eligible_rows:
        split = group_splits[row["source_group_id"].strip()]
        annotation = _read_json(annotation_path)
        examples, counters = build_smoothconv_examples(row, annotation, split)
        smooth_examples.extend(examples)
        excluded_smooth.update(counters)

    duplex_examples: list[dict[str, Any]] = []
    duplex_source_status_counts: Counter[str] = Counter()
    for row in duplex_rows:
        group_value = row.get("source_conversation_group_id") or row.get("cross_dataset_source_group_id")
        if not isinstance(group_value, str) or not group_value.strip():
            continue
        group_id = f"DuplexConv:{group_value.strip()}"
        duplex_source_status_counts[str(row.get("source_adjudication_status") or "missing")] += 1
        examples, counters = build_duplexconv_examples(row, group_splits[group_id])
        duplex_examples.extend(examples)
        excluded_duplex.update(counters)

    all_examples = plan_examples + smooth_examples + duplex_examples
    all_examples.sort(
        key=lambda row: (
            row["split"],
            row["source"],
            row["conversation_id"],
            row["example_id"],
        )
    )
    train_rows = [row for row in all_examples if row["split"] == "train"]
    validation_rows = [row for row in all_examples if row["split"] == "validation"]
    group_ids_by_split = {
        split: {row["source_group_id"] for row in rows}
        for split, rows in (("train", train_rows), ("validation", validation_rows))
    }
    leaked_groups = group_ids_by_split["train"] & group_ids_by_split["validation"]
    if leaked_groups:
        raise ValueError(f"source groups appear in both splits: {sorted(leaked_groups)[:5]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    validation_path = OUTPUT_DIR / "validation.jsonl"
    summary_path = OUTPUT_DIR / "summary.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    summary = {
        "schema_version": 1,
        "builder": "tools/build_native_action_dialogue_dataset.py",
        "inputs": {
            "dialogue_plan": str(plan_path.relative_to(ROOT)),
            "smoothconv_manifest": str(manifest_path.relative_to(ROOT)),
            "duplexconv_evidence": str(DEFAULT_DUPLEXCONV_EVIDENCE.relative_to(ROOT)),
            "smoothconv_manifest_rows": len(smooth_rows),
            "smoothconv_rows_used": len(eligible_rows),
            "smoothconv_heldout_groups_excluded": excluded_smooth.get(
                "held_out_evaluation_source_group", 0
            ),
            "duplexconv_rows": len(duplex_rows),
            "duplexconv_source_adjudication_status_counts": dict(sorted(duplex_source_status_counts.items())),
        },
        "counts": {
            "total_examples": len(all_examples),
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "examples_by_source": dict(sorted(Counter(row["source"] for row in all_examples).items())),
            "examples_by_action": dict(sorted(Counter(row["action"] for row in all_examples).items())),
            "examples_by_action_and_split": {
                split: dict(sorted(Counter(row["action"] for row in rows).items()))
                for split, rows in (("train", train_rows), ("validation", validation_rows))
            },
        },
        "group_split_report": split_report,
        "dataset_status": "EXPERIMENTAL_DERIVED_LABELS_NOT_HUMAN_ADJUDICATED",
        "all_active_actions_have_at_least_1000_examples": all(
            Counter(row["action"] for row in all_examples).get(action, 0) >= 1000
            for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
        ),
        "source_group_leakage": False,
        "excluded": {
            "dialogue_plan": plan_excluded,
            "smoothconv": dict(sorted(excluded_smooth.items())),
            "duplexconv": dict(sorted(excluded_duplex.items())),
        },
        "outputs": {
            "train": str(train_path.relative_to(ROOT)),
            "validation": str(validation_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
