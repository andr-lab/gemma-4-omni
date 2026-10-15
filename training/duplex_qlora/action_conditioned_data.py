"""Build leakage-resistant text targets conditioned on native spoken actions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from duplex_kernel.audio_response_conditioning import DEFAULT_MAX_AUDIO_MS, TICK_MS, causal_audio_window


_ROLE_TO_ACTION = {
    "assistant_backchannel": "BACKCHANNEL",
    "assistant_start": "SPEAK",
    "assistant_continue": "SPEAK",
    "assistant_replan": "REPLAN",
}
_ACTION_TO_STRATUM = {
    "BACKCHANNEL": "labelled_backchannel",
    "SPEAK": "labelled_speak",
    "REPLAN": "labelled_replan",
}
_EXPECTED_SCRIPT_VERSION = "synthetic-it-t6-v4.0.0"
_V5_PROFILE = "audio_grounded_v5"
_V5_PROFILE_VERSION = "synthetic-it-t6-audio-grounded-v5-plan.1"
_AUDIO_RESPONSE_PROTOCOL_V1 = "audio_response_conditioning_v1"
_AUDIO_RESPONSE_PROTOCOL_V2 = "heard_action_say_v2"
_ACTIONS = {"LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN"}
_BACKCHANNEL_INTERNAL_MARKER = re.compile(r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _prepared_dataset_purpose(prepared_root: Path) -> str:
    manifest_path = prepared_root / "prepared_v5_clean_asr_manifest.json"
    if not manifest_path.is_file():
        return "candidate"
    value = json.loads(manifest_path.read_text(encoding="utf-8")).get("purpose")
    return str(value) if value in {"candidate", "smoke"} else "candidate"


def _segment_text(segment: Mapping[str, Any]) -> str:
    value = segment.get("text", segment.get("script_text", segment.get("transcript", "")))
    return value.strip() if isinstance(value, str) else ""


def _segment_role(segment: Mapping[str, Any]) -> str | None:
    value = segment.get("role", segment.get("speaker"))
    return str(value).strip().lower() if value is not None else None


def _is_italian(language: Any) -> bool:
    value = str(language or "").strip().lower().replace("_", "-")
    return value in {"it", "ita", "italian"} or value.startswith("it-") or "italian" in value


def _is_complete_before(segment: Mapping[str, Any], target_start_ms: float) -> bool:
    start, end = segment.get("start_ms"), segment.get("end_ms")
    if not (_finite(start) and _finite(end) and 0 <= float(start) < float(end) <= target_start_ms):
        return False
    if (segment.get("time_mask", True) is not True
            or segment.get("text_mask", True) is not True
            or segment.get("history_mask", True) is not True):
        return False
    # Native sources may explicitly mark incomplete/nonfinal turns. The v4
    # authored set has no native state; for it, a valid ended span plus intact
    # text mask is the available completion evidence.
    state = segment.get("native_turn_state", segment.get("annotation_state"))
    if state is not None:
        normalized = str(state).strip().lower().replace("<|", "").replace("|>", "")
        if normalized != "complete":
            return False
    return bool(_segment_text(segment))


def _matching_action_event(
    action_segments: Iterable[Mapping[str, Any]], *, action: str, target_start_ms: float
) -> Mapping[str, Any] | None:
    candidates = []
    for event in action_segments:
        if str(event.get("action") or "").strip().upper() != action:
            continue
        start, end = event.get("start_ms"), event.get("end_ms")
        if not (_finite(start) and _finite(end) and float(start) < float(end)):
            continue
        delta = abs(float(start) - target_start_ms)
        if float(start) <= target_start_ms < float(end) or delta <= 80.0:
            candidates.append((delta, float(start), event))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _context_messages(
    conversation: Mapping[str, Any], *, target: Mapping[str, Any],
    current_user_id: str, current_user_start_ms: float, current_user_end_ms: float,
    target_start_ms: float,
) -> tuple[list[dict[str, str]], int]:
    messages: list[tuple[float, float, dict[str, str]]] = []
    overlapping_user_segments = 0
    for segment in conversation.get("segments", ()) or ():
        if segment is target:
            continue
        role = _segment_role(segment)
        start, end = segment.get("start_ms"), segment.get("end_ms")
        if role == "user" and _finite(start) and _finite(end):
            segment_user_id = str(segment.get("utterance_id") or segment.get("segment_id") or "")
            overlaps_current_user = float(start) < current_user_end_ms and float(end) > current_user_start_ms
            if (current_user_id and segment_user_id == current_user_id) or overlaps_current_user:
                overlapping_user_segments += 1
                continue
        if role not in {"user", "assistant"} or not _is_complete_before(segment, target_start_ms):
            continue
        text = _segment_text(segment)
        messages.append((float(start), float(end), {"role": role, "content": text}))
    # Runtime commits an assistant utterance when its audio turn completes,
    # then commits the user's transcript after the user endpoint and ASR.
    # Sorting the gold transcript history by completion reproduces that order
    # for overlaps such as a BC inside the still-open user utterance.
    messages.sort(key=lambda item: (item[1], item[0], item[2]["role"]))
    return [item[2] for item in messages], overlapping_user_segments


def _verified_user_channel(conversation: Mapping[str, Any]) -> int:
    channel_map = conversation.get("channel_map")
    if not isinstance(channel_map, Mapping):
        raise ValueError(f"{conversation.get('conversation_id')}: channel_map user non verificato")
    channels: list[int] = []
    for raw_channel, role in channel_map.items():
        if str(role).strip().lower() != "user":
            continue
        if isinstance(raw_channel, bool) or not str(raw_channel).isdigit():
            raise ValueError(f"{conversation.get('conversation_id')}: indice canale user non fisico")
        channels.append(int(raw_channel))
    if len(channels) != 1:
        raise ValueError(f"{conversation.get('conversation_id')}: richiesto un unico canale fisico user verificato")
    return channels[0]


def _user_segments_before(
    conversation: Mapping[str, Any], *, target_start_ms: float
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for segment in conversation.get("segments", ()) or ():
        if _segment_role(segment) != "user" or segment.get("time_mask", True) is not True:
            continue
        start, end = segment.get("start_ms"), segment.get("end_ms")
        if _finite(start) and _finite(end) and 0 <= float(start) < float(end) and float(start) < target_start_ms:
            candidates.append(segment)
    return candidates


def _complete_user_segments(
    segments: Iterable[Mapping[str, Any]], *, cutoff_ms: float
) -> list[Mapping[str, Any]]:
    complete: list[Mapping[str, Any]] = []
    for segment in segments:
        end = segment.get("end_ms")
        if not _finite(end) or float(end) > cutoff_ms:
            continue
        state = segment.get("native_turn_state", segment.get("native_turn_state_raw", segment.get("annotation_state")))
        if state is not None:
            normalized = str(state).strip().lower().replace("<|", "").replace("|>", "")
            if normalized != "complete":
                continue
        complete.append(segment)
    return complete


def _current_user_segment(
    conversation: Mapping[str, Any], *, action: str, target_start_ms: float, cutoff_ms: float
) -> Mapping[str, Any] | None:
    candidates = _user_segments_before(conversation, target_start_ms=target_start_ms)
    if action == "BACKCHANNEL":
        active = [
            segment for segment in candidates
            if _finite(segment.get("end_ms")) and float(segment["end_ms"]) > target_start_ms
        ]
        if not active:
            return None
        return min(active, key=lambda item: (float(item["start_ms"]), float(item["end_ms"])))
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: (float(item["start_ms"]), float(item["end_ms"])))
    if latest not in _complete_user_segments((latest,), cutoff_ms=cutoff_ms):
        return None
    return latest


def _validate_action_alignment(
    conversation: Mapping[str, Any], *, segment: Mapping[str, Any], action: str
) -> Mapping[str, Any]:
    start = segment.get("start_ms")
    end = segment.get("end_ms")
    if not (_finite(start) and _finite(end) and 0 <= float(start) < float(end) <= float(conversation.get("duration_ms", math.inf))):
        raise ValueError(f"{conversation.get('conversation_id')}: target assistant non temporizzato valido")
    event = _matching_action_event(
        conversation.get("source_action_segments", ()) or (),
        action=action,
        target_start_ms=float(start),
    )
    if event is None:
        raise ValueError(
            f"{conversation.get('conversation_id')}:{segment.get('segment_id')}: "
            f"nessun action_segments {action} allineato all'onset entro 80 ms"
        )
    return event


def _action_annotation(
    conversation: Mapping[str, Any], *, segment: Mapping[str, Any], script_role: str,
) -> tuple[str | None, Mapping[str, Any] | None, str | None]:
    """Use only onset-aligned event evidence; preserve ambiguous targets as unknown."""
    target_start = float(segment["start_ms"])
    aligned = [
        (action, event)
        for action in ("BACKCHANNEL", "SPEAK", "REPLAN")
        if (event := _matching_action_event(
            conversation.get("source_action_segments", ()) or (),
            action=action,
            target_start_ms=target_start,
        )) is not None
    ]
    role_action = _ROLE_TO_ACTION.get(script_role)
    if role_action is not None:
        matching = [event for action, event in aligned if action == role_action]
        if len(matching) == 1 and len(aligned) == 1:
            return role_action, matching[0], None
        if aligned:
            return None, None, "action_annotation_conflict"
        # A scripted role without a corresponding source event is not an action GT.
        return None, None, None
    if len(aligned) == 1:
        return aligned[0][0], aligned[0][1], None
    if len(aligned) > 1:
        return None, None, "ambiguous_action_events"
    return None, None, None


def _audio_grounded_v5_to_causal_conversation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the measured v5 contract and map its waveform timeline to the LM builder schema."""
    conversation_id = str(row.get("conversation_id") or "")
    if row.get("profile") != _V5_PROFILE or row.get("profile_version") != _V5_PROFILE_VERSION:
        raise ValueError(f"{conversation_id or 'v5 row'}: profilo audio_grounded_v5 non verificato")
    if row.get("human_annotated") is not False:
        raise ValueError(f"{conversation_id}: la provenienza v5 sintetica deve restare esplicita")
    if row.get("sample_rate") != 16000 or row.get("channels") != 2 or row.get("frame_ms") != TICK_MS:
        raise ValueError(f"{conversation_id}: v5 richiede WAV stereo 16 kHz e frame da 80 ms")
    if row.get("channel_map") != {"0": "user", "1": "assistant"}:
        raise ValueError(f"{conversation_id}: canali v5 user/assistant non corrispondono ai WAV")
    if not conversation_id or not isinstance(row.get("audio_path"), str) or not row["audio_path"].strip():
        raise ValueError("v5 conversation_id/audio_path mancante")
    audio_sha = row.get("audio_sha256")
    if not isinstance(audio_sha, str) or len(audio_sha) != 64:
        raise ValueError(f"{conversation_id}: audio_sha256 sorgente mancante")
    duration_s = row.get("duration_s")
    if not _finite(duration_s) or float(duration_s) <= 0:
        raise ValueError(f"{conversation_id}: durata v5 non valida")
    group_id = str(row.get("topic_group_id") or "")
    split = str(row.get("split") or "")
    if not group_id or split not in {"train", "validation", "test"}:
        raise ValueError(f"{conversation_id}: topic group/split v5 mancanti")

    frame_samples = 16000 * TICK_MS // 1000
    frame_targets = row.get("frame_targets")
    if not isinstance(frame_targets, list) or not frame_targets:
        raise ValueError(f"{conversation_id}: frame_targets v5 mancanti")
    for index, frame in enumerate(frame_targets):
        input_frame = frame.get("input_frame_id")
        target_frame = frame.get("target_frame_id")
        if (isinstance(input_frame, bool) or not isinstance(input_frame, int) or input_frame < 0
                or target_frame != input_frame + 1 or frame.get("frame_id") != input_frame):
            raise ValueError(f"{conversation_id}: frame_targets[{index}] non rispetta input t → target t+1")
        if (frame.get("input_text_visible") is not False
                or frame.get("future_audio_excluded") is not True
                or type(frame.get("target_action_mask")) is not bool):
            raise ValueError(f"{conversation_id}: frame_targets[{index}] non attesta input audio causale")
        expected = {
            "input_audio_start_sample": input_frame * frame_samples,
            "input_audio_end_sample": (input_frame + 1) * frame_samples,
            "target_action_start_sample": (input_frame + 1) * frame_samples,
            "target_action_end_sample": (input_frame + 2) * frame_samples,
        }
        if any(frame.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{conversation_id}: frame_targets[{index}] ha cutoff/campioni non causali")
        if str(frame.get("action") or "").upper() not in _ACTIONS:
            raise ValueError(f"{conversation_id}: frame_targets[{index}] ha un'azione sconosciuta")

    utterances = row.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError(f"{conversation_id}: utterances v5 mancanti")
    utterance_by_id: dict[str, Mapping[str, Any]] = {}
    segments: list[dict[str, Any]] = []
    duration_ms = float(duration_s) * 1000.0
    for utterance in utterances:
        utterance_id = str(utterance.get("utterance_id") or "")
        speaker = str(utterance.get("speaker") or "").strip().lower()
        text = _segment_text(utterance)
        start_sample, end_sample = utterance.get("start_sample"), utterance.get("end_sample")
        if (not utterance_id or utterance_id in utterance_by_id or speaker not in {"user", "assistant"}
                or not text or isinstance(start_sample, bool) or isinstance(end_sample, bool)
                or not isinstance(start_sample, int) or not isinstance(end_sample, int)
                or start_sample < 0 or end_sample <= start_sample or end_sample > round(duration_s * 16000)):
            raise ValueError(f"{conversation_id}: utterance v5 priva di speaker, testo o intervallo valido")
        utterance_by_id[utterance_id] = utterance
        start_ms, end_ms = start_sample * 1000.0 / 16000.0, end_sample * 1000.0 / 16000.0
        measured_duration = utterance.get("measured_duration_s")
        was_cut = (
            speaker == "assistant"
            and _finite(measured_duration)
            and end_sample < start_sample + round(float(measured_duration) * 16000)
        )
        segments.append({
            "role": speaker,
            "speaker": speaker,
            "script_role": str(utterance.get("role") or ""),
            "channel_index": 0 if speaker == "user" else 1,
            "audio_channel_verified": True,
            "segment_id": utterance_id,
            "utterance_id": utterance_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "target_action": utterance.get("target_action"),
            "target_mask": utterance.get("target_action") in {"SPEAK", "BACKCHANNEL", "REPLAN"},
            "time_mask": True,
            "text_mask": True,
            "history_mask": not was_cut,
            "native_turn_state": "incomplete" if was_cut else "complete",
        })

    events: list[dict[str, Any]] = []
    response_event: dict[str, Mapping[str, Any]] = {}
    for event in row.get("event_schedule", ()) or ():
        action = str(event.get("action") or "").upper()
        if action not in _ACTIONS:
            continue
        start_s, end_s = event.get("start_s"), event.get("end_s")
        if not (_finite(start_s) and _finite(end_s) and float(start_s) < float(end_s)):
            raise ValueError(f"{conversation_id}: event_schedule contiene intervallo non valido")
        canonical = {
            **dict(event),
            "action": action,
            "start_ms": float(start_s) * 1000.0,
            "end_ms": float(end_s) * 1000.0,
        }
        events.append(canonical)
        response_id = event.get("response_utterance_id")
        if action in {"SPEAK", "BACKCHANNEL", "REPLAN"} and response_id is not None:
            response = str(response_id)
            utterance = utterance_by_id.get(response)
            if utterance is None or str(utterance.get("target_action") or "").upper() != action:
                raise ValueError(f"{conversation_id}: evento {action} non corrisponde al target assistant")
            if response in response_event:
                raise ValueError(f"{conversation_id}: più eventi azione per l'utterance {response}")
            event_start_ms = canonical["start_ms"]
            utterance_start_ms = int(utterance["start_sample"]) * 1000.0 / 16000.0
            if abs(event_start_ms - utterance_start_ms) > TICK_MS:
                raise ValueError(f"{conversation_id}: evento {action} non allineato all'onset assistant")
            response_event[response] = canonical
            if action == "BACKCHANNEL":
                user_id = str(event.get("context_utterance_id") or "")
                context = utterance_by_id.get(user_id)
                elapsed = event.get("user_elapsed_before_backchannel_ms")
                if (context is None or context.get("speaker") != "user" or not _finite(elapsed)
                        or not 800.0 <= float(elapsed) <= 1600.0
                        or event_start_ms < int(context["start_sample"]) * 1000.0 / 16000.0 + 800.0
                        or event_start_ms >= int(context["end_sample"]) * 1000.0 / 16000.0):
                    raise ValueError(f"{conversation_id}: BACKCHANNEL v5 senza contenuto causale user sufficiente")
                word_count = len(_BACKCHANNEL_INTERNAL_MARKER.sub(" ", _segment_text(utterance)).split())
                if not 1 <= word_count <= 3:
                    raise ValueError(f"{conversation_id}: BACKCHANNEL v5 deve contenere 1–3 parole")
            elif action == "REPLAN":
                correction_id = str(event.get("context_utterance_id") or "")
                correction = utterance_by_id.get(correction_id)
                correction_end_ms = (
                    int(correction["end_sample"]) * 1000.0 / 16000.0 if correction is not None else math.inf
                )
                wait_ms = float(event.get("wait_after_correction_ms", -1))
                if (correction is None or correction.get("speaker") != "user"
                        or wait_ms < 800.0 or event_start_ms < correction_end_ms + 800.0):
                    raise ValueError(f"{conversation_id}: REPLAN v5 non attende la correzione completa +800 ms")

    for segment in segments:
        if str(segment.get("target_action") or "").upper() in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
            if segment["utterance_id"] not in response_event:
                raise ValueError(f"{conversation_id}: target assistant privo di evento temporizzato v5")

    return {
        **dict(row),
        "source": str(row.get("source") or "authored_synthetic_audio_candidate"),
        "source_group_id": group_id,
        "synthetic": True,
        "sample_rate_hz": 16000,
        "duration_ms": duration_ms,
        "tick_ms": TICK_MS,
        "segments": segments,
        "source_action_segments": events,
        "_audio_grounded_v5": True,
    }


def build_action_conditioned_records(
    conversations: Iterable[Mapping[str, Any]], *, split: str, max_audio_ms: int = DEFAULT_MAX_AUDIO_MS,
    response_protocol: str = _AUDIO_RESPONSE_PROTOCOL_V1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build direct-audio response examples from v4 and measured v5 assistant turns.

    Action labels remain metadata. The Thinker receives prior completed history
    plus the causal physical user-audio window ending before its target.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("action-conditioned data supports train/validation/test splits")
    if isinstance(max_audio_ms, bool) or int(max_audio_ms) <= 0:
        raise ValueError("max_audio_ms deve essere positivo")
    if response_protocol not in {_AUDIO_RESPONSE_PROTOCOL_V1, _AUDIO_RESPONSE_PROTOCOL_V2}:
        raise ValueError(f"protocollo di risposta audio non supportato: {response_protocol!r}")
    records: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str]] = set()
    context_stats: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for source_conversation in conversations:
        if source_conversation.get("profile") == _V5_PROFILE:
            conversation = _audio_grounded_v5_to_causal_conversation(source_conversation)
        else:
            if source_conversation.get("source") != "synthetic_it_t6" or source_conversation.get("script_version") != _EXPECTED_SCRIPT_VERSION:
                continue
            conversation = source_conversation
            if conversation.get("synthetic") is not True:
                raise ValueError(f"{conversation.get('conversation_id')}: v4 source privo di synthetic=true")
        declared_split = str(conversation.get("split") or split)
        if declared_split not in {split, "valid" if split == "validation" else split}:
            raise ValueError(f"{conversation.get('conversation_id')}: split dichiarato {declared_split!r}, atteso {split!r}")
        conversation_id = str(conversation.get("conversation_id") or "")
        group_id = str(conversation.get("source_group_id") or conversation_id)
        if not conversation_id or not group_id:
            raise ValueError("v4 conversation_id/source_group_id mancante")
        audio_path = conversation.get("audio_path")
        audio_sha = conversation.get("audio_sha256")
        if not isinstance(audio_path, str) or not audio_path.strip() or not isinstance(audio_sha, str) or len(audio_sha) != 64:
            raise ValueError(f"{conversation_id}: audio path/SHA256 sorgente mancanti")
        user_channel_index = _verified_user_channel(conversation)
        if conversation.get("language_training_eligible") is False:
            skipped["language_training_ineligible"] += 1
            continue
        for segment in conversation.get("segments", ()) or ():
            if _segment_role(segment) != "assistant":
                continue
            script_role = str(segment.get("script_role") or "")
            if segment.get("target_mask", True) is not True or segment.get("text_mask", True) is not True or segment.get("time_mask", True) is not True:
                skipped["masked:assistant_target"] += 1
                continue
            text = _segment_text(segment)
            if not text:
                skipped["empty_text:assistant_target"] += 1
                continue
            # The prepared Stage1 v4 conversation rows omit their language
            # field. This builder selects only the attested Italian v4 source
            # and script version above, so an absent field has a safe `it-IT`
            # default while any explicit segment/conversation value wins.
            language = str(segment.get("language") or conversation.get("language") or "it-IT")
            if not _is_italian(language):
                skipped["non_italian_target"] += 1
                continue
            if not (_finite(segment.get("start_ms")) and _finite(segment.get("end_ms"))
                    and 0 <= float(segment["start_ms"]) < float(segment["end_ms"])):
                skipped["untimed_assistant_target"] += 1
                continue
            conversation_duration = conversation.get("duration_ms")
            if _finite(conversation_duration) and float(segment["end_ms"]) > float(conversation_duration):
                skipped["assistant_target_beyond_audio_duration"] += 1
                continue
            action, event, annotation_skip = _action_annotation(
                conversation, segment=segment, script_role=script_role,
            )
            if annotation_skip:
                skipped[annotation_skip] += 1
                continue
            action_label = action or "UNKNOWN"
            if response_protocol == _AUDIO_RESPONSE_PROTOCOL_V2 and action_label not in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
                skipped["v2_unknown_action_has_no_protocol_target"] += 1
                continue
            sampling_stratum = _ACTION_TO_STRATUM.get(action_label, "conversational_regular")
            window_action = action or "SPEAK"  # unknown targets use completed-turn audio only, not SPEAK supervision
            start_ms = float(segment["start_ms"])
            cutoff_frame_count = math.floor(start_ms / TICK_MS)
            input_cutoff_ms = cutoff_frame_count * TICK_MS
            current_user = _current_user_segment(
                conversation,
                action=window_action,
                target_start_ms=start_ms,
                cutoff_ms=input_cutoff_ms,
            )
            if current_user is None:
                skipped[f"no_causal_current_user_turn:{sampling_stratum}"] += 1
                continue
            try:
                audio_window = causal_audio_window(
                    action=window_action,
                    target_start_ms=start_ms,
                    current_user_start_ms=float(current_user["start_ms"]),
                    current_user_end_ms=float(current_user["end_ms"]),
                    current_user_active=(window_action == "BACKCHANNEL"),
                    max_audio_ms=int(max_audio_ms),
                )
            except ValueError as exc:
                reason = (
                    "user_turn_exceeds_max_audio_window"
                    if "supera la durata massima" in str(exc)
                    else "noncausal_or_empty_user_audio_window"
                )
                skipped[f"{reason}:{sampling_stratum}"] += 1
                continue
            history, overlapping_user = _context_messages(
                conversation,
                target=segment,
                current_user_id=str(
                    current_user.get("utterance_id") or current_user.get("segment_id") or ""
                ),
                current_user_start_ms=float(current_user["start_ms"]),
                current_user_end_ms=float(current_user["end_ms"]),
                target_start_ms=start_ms,
            )
            context_stats[f"context_messages:{sampling_stratum}"] += len(history)
            context_stats[f"overlapping_user_segments_hidden:{sampling_stratum}"] += overlapping_user
            context_stats[f"current_user_transcripts_hidden:{sampling_stratum}"] += 1
            target_id = str(segment.get("utterance_id") or segment.get("segment_id") or "")
            key = (conversation_id, target_id)
            if not target_id or key in seen_targets:
                raise ValueError(f"target utterance ID mancante o duplicato: {key}")
            seen_targets.add(key)
            context_stats[f"targets_missing_user_latest:{sampling_stratum}"] += int(bool(history) and history[-1]["role"] != "user")
            context_stats[f"targets_without_history:{sampling_stratum}"] += int(not history)
            response_target_text = text
            response_target_sha = hashlib.sha256(
                unicodedata.normalize("NFKC", " ".join(text.casefold().split())).encode("utf-8")
            ).hexdigest()
            protocol_target_fields: dict[str, str] = {}
            if response_protocol == _AUDIO_RESPONSE_PROTOCOL_V2:
                if action_label == "BACKCHANNEL":
                    heard_text = ""
                else:
                    user_end = float(current_user["end_ms"])
                    if user_end > float(audio_window["input_cutoff_ms"]):
                        skipped[f"v2_heard_transcript_after_cutoff:{action_label}"] += 1
                        continue
                    heard_text = _segment_text(current_user)
                    if not heard_text:
                        skipped[f"v2_empty_heard_transcript:{action_label}"] += 1
                        continue
                from duplex_kernel.audio_response_protocol import format_audio_response_v2_target

                response_target_text = format_audio_response_v2_target(
                    heard=heard_text,
                    action=action_label,
                    say=text,
                )
                protocol_target_fields = {
                    "audio_response_protocol": _AUDIO_RESPONSE_PROTOCOL_V2,
                    "target_heard_text": heard_text,
                    "target_say_text": text,
                }
            record = {
                "conversation_id": conversation_id,
                "source_group_id": group_id,
                "source": str(conversation.get("source", "synthetic_it_t6")),
                "synthetic": True,
                "profile": conversation.get("profile"),
                "profile_version": conversation.get("profile_version"),
                "split": split,
                "task_kind": "conversation_response",
                "conditioning_mode": "baselineaudio_response",
                **({"audio_response_contract": _AUDIO_RESPONSE_PROTOCOL_V1}
                   if response_protocol == _AUDIO_RESPONSE_PROTOCOL_V1 else {}),
                "audio_mask": True,
                "max_audio_ms": int(max_audio_ms),
                "action": action_label,
                "action_label": action_label.lower(),
                "sampling_stratum": sampling_stratum,
                "action_label_provenance": (
                    "measured_v5_event_schedule" if action and conversation.get("_audio_grounded_v5")
                    else "onset_aligned_source_event" if action else "unknown_no_aligned_event"
                ),
                "unknown_action_sampling_note": (
                    "regular_response_proxy_only; completed-user-turn audio; no action class supervision"
                    if not action else None
                ),
                "action_is_prompt_conditioning": False,
                "target_script_role": script_role,
                "target_utterance_id": target_id,
                "target_start_ms": start_ms,
                "target_end_ms": float(segment["end_ms"]),
                "action_event_start_ms": float(event["start_ms"]) if event is not None else None,
                "action_event_end_ms": float(event["end_ms"]) if event is not None else None,
                "current_user_utterance_id": str(
                    current_user.get("utterance_id") or current_user.get("segment_id") or ""
                ),
                "current_user_start_ms": float(current_user["start_ms"]),
                "current_user_end_ms": float(current_user["end_ms"]),
                "current_user_active_at_target": window_action == "BACKCHANNEL",
                "audio_path": audio_path,
                "source_audio_sha256": audio_sha,
                "sample_rate_hz": int(conversation.get("sample_rate_hz", 16000)),
                "audio_channel_index": user_channel_index,
                **audio_window,
                "language": language,
                "history_messages": history,
                "context_text": "",
                "target_text": response_target_text,
                "target_text_sha256": response_target_sha,
            }
            record.update(protocol_target_fields)
            records.append(record)
    records.sort(key=lambda row: (row["conversation_id"], row["target_start_ms"], row["target_utterance_id"]))
    counts = Counter(record["action"] for record in records)
    stratum_counts = Counter(record["sampling_stratum"] for record in records)
    unique_texts = defaultdict(set)
    groups = defaultdict(set)
    for record in records:
        unique_texts[record["action"]].add(record["target_text_sha256"])
        groups[record["action"]].add(record["source_group_id"])
    report = {
        "split": split,
        "audio_response_protocol": response_protocol,
        "records": len(records),
        "records_by_action": {action: counts[action] for action in ("BACKCHANNEL", "SPEAK", "REPLAN", "UNKNOWN")},
        "records_by_sampling_stratum": {
            stratum: stratum_counts[stratum]
            for stratum in ("conversational_regular", "labelled_speak", "labelled_backchannel", "labelled_replan")
        },
        "unique_target_texts_by_action": {
            action: len(unique_texts[action]) for action in ("BACKCHANNEL", "SPEAK", "REPLAN", "UNKNOWN")
        },
        "source_groups_by_action": {action: len(groups[action]) for action in ("BACKCHANNEL", "SPEAK", "REPLAN", "UNKNOWN")},
        "context_and_exclusion_counts": dict(sorted(context_stats.items())),
        "skipped": dict(sorted(skipped.items())),
        "audio_input_contract": {
            "source_channel": "one verified physical user channel only",
            "sample_rate_hz": 16000,
            "tick_ms": TICK_MS,
            "cutoff": "floor(target_start_ms / tick_ms) * tick_ms; end-exclusive",
            "start": "onset of the current physical user turn",
            "max_audio_ms": int(max_audio_ms),
            "overlong_turns": "excluded without truncation",
            "current_user_transcript_in_prompt": False,
            "action_in_prompt": False,
            "unknown_response_targets": "loss-supervised assistant targets with no action-class ground truth; sampled as conversational_regular",
        },
        "audio_metadata": {
            "all_targets_have_source_audio_sha256": all(len(record["source_audio_sha256"]) == 64 for record in records),
            "all_targets_have_audio_path": all(bool(record["audio_path"]) for record in records),
            "all_targets_have_audio_mask": all(record["audio_mask"] is True for record in records),
            "all_targets_have_physical_user_channel": all(isinstance(record["audio_channel_index"], int) for record in records),
            "all_targets_have_valid_time": all(record["target_start_ms"] < record["target_end_ms"] for record in records),
            "all_input_cutoffs_are_causal": all(record["audio_end_ms"] <= record["target_start_ms"] for record in records),
            "input_audio_duration_ms_by_bucket": {
                bucket: sum(
                    (lower <= record["input_audio_duration_ms"] < upper)
                    if upper is not None
                    else record["input_audio_duration_ms"] >= lower
                    for record in records
                )
                for bucket, lower, upper in (
                    ("0_5s", 0, 5000),
                    ("5_10s", 5000, 10000),
                    ("10_20s", 10000, 20000),
                    ("20_30s", 20000, 30000),
                    ("30s_plus", 30000, None),
                )
            },
        },
    }
    return records, report


def build_backchannel_lexicon(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive short BC text candidates only from the training split's BC targets."""
    target_rows = [
        record for record in records
        if record.get("action") == "BACKCHANNEL" and record.get("split") == "train"
    ]
    excluded_rows: Counter[str] = Counter()
    excluded_unique: dict[str, set[str]] = defaultdict(set)
    unique_target_keys: set[str] = set()
    candidate_by_key: dict[str, str] = {}
    candidate_rows = 0

    for record in target_rows:
        raw_text = record.get("target_say_text", record.get("target_text"))
        text = unicodedata.normalize("NFKC", raw_text).strip() if isinstance(raw_text, str) else ""
        phrase = " ".join(text.split())
        key = phrase.casefold()
        unique_target_keys.add(key)

        reason: str | None = None
        if not phrase:
            reason = "empty"
        elif any(mark in phrase for mark in ("?", "？", "¿")):
            reason = "question"
        elif _BACKCHANNEL_INTERNAL_MARKER.search(phrase):
            reason = "internal_marker"
        else:
            words = re.findall(r"\w+(?:['’]\w+)*", phrase, flags=re.UNICODE)
            if not words:
                reason = "no_words"
            elif len(words) > 3:
                reason = "over_3_words"

        if reason is not None:
            excluded_rows[reason] += 1
            excluded_unique[reason].add(key)
            continue
        candidate_rows += 1
        candidate_by_key.setdefault(key, phrase)

    candidates = [candidate_by_key[key] for key in sorted(candidate_by_key)]
    if not candidates:
        raise ValueError("training split contains no eligible BACKCHANNEL text candidates")
    total = len(target_rows)
    counts = {
        "target_records": total,
        "unique_target_texts": len(unique_target_keys),
        "candidate_rows": candidate_rows,
        "candidate_count": len(candidates),
        "covered_target_records": candidate_rows,
        "coverage_fraction": round(candidate_rows / total, 6) if total else 0.0,
        "excluded_target_rows_by_reason": dict(sorted(excluded_rows.items())),
        "excluded_unique_target_texts_by_reason": {
            reason: len(keys) for reason, keys in sorted(excluded_unique.items())
        },
    }
    return {
        "schema_version": 1,
        "language": "it",
        "source": {
            "split": "train",
            "field": "action-conditioned BACKCHANNEL target_text",
        },
        "filters": {
            "min_words": 1,
            "max_words": 3,
            "excluded_question_marks": ["?", "？", "¿"],
            "excluded_internal_markers": "angle, square, or curly bracket markers",
            "deduplication": "NFKC, collapsed whitespace, and casefold; target spelling retained",
        },
        "counts": counts,
        "phrases": candidates,
    }


def _write_backchannel_lexicon_payload(lexicon: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".partial", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(lexicon, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_backchannel_lexicon(train_jsonl: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Atomically write the training-only BC lexicon from prepared action-conditioned rows."""
    source_path = Path(train_jsonl).expanduser().resolve()
    rows = [json.loads(line) for line in source_path.open(encoding="utf-8") if line.strip()]
    lexicon = build_backchannel_lexicon(rows)
    _write_backchannel_lexicon_payload(lexicon, Path(output_path).expanduser().resolve())
    return lexicon


def prepare_action_conditioned_dataset(
    prepared_dir: str | Path, output_dir: str | Path, *, max_audio_ms: int = DEFAULT_MAX_AUDIO_MS,
    response_protocol: str = _AUDIO_RESPONSE_PROTOCOL_V1,
) -> dict[str, Any]:
    prepared_root = Path(prepared_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output action-conditioned esistente, non sovrascritto: {output_root}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "direct native audio-conditioned Italian assistant responses; action labels remain metadata",
        "dataset_purpose": _prepared_dataset_purpose(prepared_root),
        "prepared_dir": str(prepared_root),
        "prompt_contract": "shared native-audio prompt builder; current user transcript and action label omitted",
        "assistant_loss": "causal LM labels only after native audio plus prior completed history",
        "max_audio_ms": int(max_audio_ms),
        "whisper_quality_evidence": "not measured by dataset preparation; report ASR retention separately by source",
        "splits": {},
    }
    if response_protocol == _AUDIO_RESPONSE_PROTOCOL_V1:
        result["audio_response_contract"] = _AUDIO_RESPONSE_PROTOCOL_V1
    elif response_protocol == _AUDIO_RESPONSE_PROTOCOL_V2:
        result["audio_response_protocol"] = _AUDIO_RESPONSE_PROTOCOL_V2
        result["target_protocol"] = '{"heard":"...","action":"SPEAK|BACKCHANNEL|REPLAN","say":"..."}'
        result["heard_label_policy"] = "SPEAK/REPLAN full current utterance only when ended by causal cutoff; BACKCHANNEL heard is empty"
    else:
        raise ValueError(f"protocollo di risposta audio non supportato: {response_protocol!r}")
    split_groups: dict[str, set[str]] = {}
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        source_path = prepared_root / f"{split}.jsonl"
        if split == "test" and not source_path.is_file():
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"split audio-response mancante: {source_path}")
        rows = [json.loads(line) for line in source_path.open(encoding="utf-8") if line.strip()]
        records, report = build_action_conditioned_records(
            rows, split=split, max_audio_ms=max_audio_ms, response_protocol=response_protocol,
        )
        split_groups[split] = {record["source_group_id"] for record in records}
        if not records:
            raise ValueError(f"nessun target v4 per split {split}")
        split_records[split] = records
        report["source_conversations"] = len({record["conversation_id"] for record in records})
        report["source_audio_files"] = len({(record["audio_path"], record["source_audio_sha256"]) for record in records})
        result["splits"][split] = report
    group_owner: dict[str, str] = {}
    for split, groups_in_split in split_groups.items():
        for group in groups_in_split:
            previous = group_owner.setdefault(group, split)
            if previous != split:
                raise ValueError(f"source_group_id attraversa {previous}/{split}: {group}")
    result["group_split_disjoint"] = True
    backchannel_lexicon = build_backchannel_lexicon(split_records["train"])
    result["backchannel_lexicon"] = {
        "file": "backchannel_lexicon.json",
        "counts": backchannel_lexicon["counts"],
    }
    output_root.mkdir(parents=True)
    for split, records in split_records.items():
        with (output_root / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    lexicon_path = output_root / "backchannel_lexicon.json"
    _write_backchannel_lexicon_payload(backchannel_lexicon, lexicon_path)
    manifest_path = output_root / "action_conditioned_manifest.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def prepare_audio_asr_mix_dataset(
    prepared_dir: str | Path,
    audio_response_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_asr_examples: int = 32,
    seed: int = 20261002,
    response_protocol: str = "v1",
) -> dict[str, Any]:
    """Join direct response targets and clean ASR rows without mixing their prompts."""
    if validation_asr_examples <= 0:
        raise ValueError("validation_asr_examples deve essere positivo")
    if response_protocol not in {"v1", "v2"}:
        raise ValueError(f"response_protocol non supportato: {response_protocol!r}")
    protocol_marker = _AUDIO_RESPONSE_PROTOCOL_V1 if response_protocol == "v1" else _AUDIO_RESPONSE_PROTOCOL_V2
    mix_contract = f"audio_asr_mix_{response_protocol}"
    prepared_root = Path(prepared_dir).expanduser().resolve()
    response_root = Path(audio_response_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output audio/ASR mix esistente, non sovrascritto: {output_root}")

    # Reuse Stage1's established converter so clean ASR keeps its native audio
    # span and ASR-specific prompt contract.
    from .stage1 import _training_records

    splits: dict[str, list[dict[str, Any]]] = {}
    split_reports: dict[str, Any] = {}
    groups_by_split: dict[str, set[tuple[str, str]]] = {}
    for split in ("train", "validation", "test"):
        response_path = response_root / f"{split}.jsonl"
        prepared_path = prepared_root / f"{split}.jsonl"
        if split == "test" and (not response_path.is_file() or not prepared_path.is_file()):
            continue
        if not response_path.is_file() or not prepared_path.is_file():
            raise FileNotFoundError(
                f"audio/ASR mix richiede gli input dello split {split}: {response_path}, {prepared_path}"
            )
        response_rows = [json.loads(line) for line in response_path.open(encoding="utf-8") if line.strip()]
        prepared_rows = [json.loads(line) for line in prepared_path.open(encoding="utf-8") if line.strip()]
        direct = [
            row for row in response_rows
            if (row.get("audio_response_contract") == protocol_marker if response_protocol == "v1"
                else row.get("audio_response_protocol") == protocol_marker)
            and row.get("action") in {"SPEAK", "BACKCHANNEL", "REPLAN"}
            and not (
                row.get("action") == "BACKCHANNEL"
                and row.get("action_label_provenance") != "measured_v5_event_schedule"
            )
        ]
        asr = [
            row for row in _training_records(prepared_rows, conditioning_mode="baselineaudio_response")
            if row.get("task_kind") == "asr"
            and row.get("source") == "synthetic_v3_clean_asr"
            and row.get("audio_mask") is True
        ]
        if split == "validation":
            by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in asr:
                by_conversation[str(row.get("conversation_id") or "")].append(row)
            if "" in by_conversation or len(by_conversation) < validation_asr_examples:
                raise ValueError(
                    f"validation ASR richiede {validation_asr_examples} conversation_id distinti; trovati {len(by_conversation)}"
                )
            rng = random.Random(int(seed))
            conversation_ids = sorted(by_conversation)
            rng.shuffle(conversation_ids)
            asr = [rng.choice(by_conversation[conversation]) for conversation in conversation_ids[:validation_asr_examples]]
        if split == "train" and not asr:
            raise ValueError("train audio/ASR mix privo di record clean ASR")
        if not direct:
            raise ValueError(f"split {split} senza SPEAK/BACKCHANNEL v5/REPLAN causali")

        combined = [
            {**row, "task_mix_contract": mix_contract} for row in direct
        ] + [
            {**row, "task_mix_contract": mix_contract} for row in asr
        ]
        groups = {
            (str(row.get("source", "")), str(row.get("source_group_id") or row.get("conversation_id") or ""))
            for row in combined
        }
        if any(not group[1] for group in groups):
            raise ValueError(f"split {split} contiene un record senza source group")
        groups_by_split[split] = groups
        splits[split] = combined
        by_task = Counter(str(row.get("task_kind", "unknown")) for row in combined)
        by_action = Counter(
            str(row.get("action", "UNKNOWN")) for row in combined
            if row.get("task_kind") == "conversation_response"
        )
        split_reports[split] = {
            "records": len(combined),
            "records_by_task": dict(sorted(by_task.items())),
            "audio_response_by_action": dict(sorted(by_action.items())),
            "asr_validation_examples": len(asr) if split == "validation" else None,
            "asr_validation_unique_conversations": (
                len({str(row.get("conversation_id")) for row in asr}) if split == "validation" else None
            ),
            "asr_validation_source_groups": (
                len({str(row.get("source_group_id")) for row in asr}) if split == "validation" else None
            ),
            "source_groups": len(groups),
        }
        if response_protocol == "v2":
            from .stage1 import _validate_audio_asr_mix_records

            _validate_audio_asr_mix_records(
                combined,
                split=split,
                response_protocol="v2",
                require_asr=split != "test",
            )

    group_owner: dict[tuple[str, str], str] = {}
    for split, groups in groups_by_split.items():
        for group in groups:
            previous = group_owner.setdefault(group, split)
            if previous != split:
                raise ValueError(f"source group attraversa train/validation/test: {group}")
    output_root.mkdir(parents=True)
    for split, rows in splits.items():
        with (output_root / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "task_mix_contract": mix_contract,
        "audio_response_protocol": protocol_marker,
        "purpose": _prepared_dataset_purpose(prepared_root),
        "sampling": "20% clean native-audio ASR, 40% direct SPEAK, 20% direct BACKCHANNEL, 20% direct REPLAN",
        "validation_asr_examples": int(validation_asr_examples),
        "validation_asr_sampling_unit": "unique conversation_id within the prepared validation split; source-group holdout remains at the original split boundary",
        "v4_backchannel_policy": "excluded; only measured v5 event-schedule backchannels are included",
        "direct_audio_prompt": (
            "native audio and prior completed history; current-user transcript and action label omitted"
            if response_protocol == "v1"
            else "generic v2 protocol prompt plus native audio and prior completed history; no current-user transcript or expected action in input"
        ),
        "asr_prompt": "existing native-audio ASR prompt and actual speaker-channel PCM",
        "group_split_disjoint": True,
        "splits": split_reports,
    }
    (output_root / "audio_asr_mix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_audio_grounded_v5_dataset(
    candidate_dir: str | Path,
    base_prepared_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify candidate PCM and add its causal rows to prepared v4 + clean-ASR splits."""
    candidate_root = Path(candidate_dir).expanduser().resolve()
    base_root = Path(base_prepared_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    is_smoke_candidate = "smoke" in candidate_root.name.lower()
    if output_root.exists():
        raise FileExistsError(f"output prepared v5 esistente, non sovrascritto: {output_root}")
    manifest_path = candidate_root / "manifest_audio_grounded_v5.jsonl"
    plan_path = candidate_root / "dialogue_plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError(f"candidate v5 incompleto: servono {manifest_path} e {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("profile") != _V5_PROFILE:
        raise ValueError("dialogue_plan non dichiara audio_grounded_v5")
    expected_counts = plan.get("split_metadata", {}).get("dialogue_counts")
    if not isinstance(expected_counts, Mapping):
        raise ValueError("dialogue_plan v5 privo dei conteggi per split")
    planned_dialogues = plan.get("dialogues")
    if not isinstance(planned_dialogues, list) or not planned_dialogues:
        raise ValueError("dialogue_plan v5 privo delle conversation pianificate")
    planned_by_id = {
        str(row.get("conversation_id") or ""): row
        for row in planned_dialogues
        if isinstance(row, Mapping) and row.get("conversation_id")
    }
    if len(planned_by_id) != len(planned_dialogues):
        raise ValueError("dialogue_plan v5 contiene ID mancanti o duplicati")
    v5_rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
    actual_counts = Counter(str(row.get("split")) for row in v5_rows)
    planned_counts = Counter(str(row.get("split")) for row in planned_dialogues)
    required_splits = ("train", "validation", "test")
    if any(planned_counts.get(split, 0) != int(expected_counts.get(split, -1)) for split in required_splits):
        raise ValueError("dialogue_plan v5: IDs pianificati e conteggi per split non coincidono")
    if is_smoke_candidate:
        counts_invalid = any(
            actual_counts.get(split, 0) <= 0
            or actual_counts.get(split, 0) > int(expected_counts.get(split, -1))
            for split in required_splits
        )
    else:
        counts_invalid = any(
            actual_counts.get(split, 0) != int(expected_counts.get(split, -1))
            for split in required_splits
        )
    if counts_invalid or set(actual_counts) - set(required_splits):
        raise ValueError(f"candidate v5 incompleto o contaminato: conteggi manifest {dict(actual_counts)} != piano {dict(expected_counts)}")
    seen_manifest_ids: set[str] = set()
    for row in v5_rows:
        conversation_id = str(row.get("conversation_id") or "")
        if not conversation_id or conversation_id in seen_manifest_ids:
            raise ValueError("manifest v5 contiene ID mancanti o duplicati")
        seen_manifest_ids.add(conversation_id)
        planned = planned_by_id.get(conversation_id)
        if planned is None:
            raise ValueError(f"manifest v5 include conversation_id non presente nel piano: {conversation_id}")
        if str(planned.get("split")) != str(row.get("split")):
            raise ValueError(f"{conversation_id}: split del manifest diverso dal piano")
        planned_group = str(planned.get("topic_group_id") or "")
        if planned_group and str(row.get("topic_group_id") or "") != planned_group:
            raise ValueError(f"{conversation_id}: topic_group_id del manifest diverso dal piano")
    if not is_smoke_candidate and seen_manifest_ids != set(planned_by_id):
        raise ValueError("manifest v5 completo ma non contiene tutti gli ID del piano")

    import numpy as np
    import soundfile as sf

    prepared_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "validation", "test")}
    sources_by_split: dict[str, Counter[str]] = {split: Counter() for split in prepared_by_split}
    seen_conversations: set[str] = set()
    seen_v5_audio_hashes: set[str] = set()
    for index, row in enumerate(v5_rows):
        if str(row.get("conversation_id") or "") in seen_conversations:
            raise ValueError(f"manifest v5 riga {index}: conversation_id duplicato")
        seen_conversations.add(str(row["conversation_id"]))
        raw_path = Path(str(row.get("audio_path") or ""))
        if not raw_path.is_absolute():
            raw_path = candidate_root / raw_path
        if not raw_path.is_file():
            raw_path = candidate_root / "audio_grounded_v5" / "audio" / str(row.get("split")) / f"{row.get('conversation_id')}.wav"
        if not raw_path.is_file():
            raise FileNotFoundError(f"v5 waveform mancante per {row.get('conversation_id')}: {raw_path}")
        audio_path = raw_path.resolve()
        info = sf.info(str(audio_path))
        if info.format != "WAV" or info.subtype != "PCM_16" or info.samplerate != 16000 or info.channels != 2:
            raise ValueError(f"{row['conversation_id']}: waveform v5 deve essere WAV PCM16 stereo 16 kHz")
        expected_frames = round(float(row.get("duration_s", 0.0)) * 16000)
        if info.frames != expected_frames:
            raise ValueError(f"{row['conversation_id']}: durata dichiarata diversa dal WAV")
        decoded, decoded_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if decoded_rate != 16000 or decoded.shape != (info.frames, 2) or not np.isfinite(decoded).all():
            raise ValueError(f"{row['conversation_id']}: decodifica WAV non valida")
        digest = hashlib.sha256()
        with audio_path.open("rb") as wav_handle:
            for block in iter(lambda: wav_handle.read(1024 * 1024), b""):
                digest.update(block)
        actual_sha = digest.hexdigest()
        if actual_sha != str(row.get("audio_sha256", "")).lower():
            raise ValueError(f"{row['conversation_id']}: SHA256 manifest non coincide con il WAV")
        if actual_sha in seen_v5_audio_hashes:
            raise ValueError(f"{row['conversation_id']}: WAV duplicato nel candidate")
        seen_v5_audio_hashes.add(actual_sha)
        normalized = _audio_grounded_v5_to_causal_conversation({**row, "audio_path": str(audio_path)})
        split = str(normalized["split"])
        prepared_by_split[split].append(normalized)
        sources_by_split[split][str(normalized.get("source"))] += 1

    base_groups: dict[tuple[str, str], str] = {}
    for split in prepared_by_split:
        path = base_root / f"{split}.jsonl"
        if split == "test" and not path.is_file():
            continue
        if not path.is_file():
            if split in {"train", "validation"}:
                raise FileNotFoundError(f"prepared v4/clean-ASR split mancante: {path}")
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        for row in rows:
            source = str(row.get("source") or "")
            if source not in {"synthetic_it_t6", "synthetic_v3_clean_asr"}:
                raise ValueError(f"prepared base include source non ammessa: {source}")
            if str(row.get("split")) != split:
                raise ValueError(f"prepared base row split diverso dal file {split}")
            group = str(row.get("source_group_id") or row.get("conversation_id") or "")
            if not group:
                raise ValueError(f"prepared base row senza source_group_id: {row.get('conversation_id')}")
            key = (source, group)
            prior_split = base_groups.setdefault(key, split)
            if prior_split != split:
                raise ValueError(f"prepared base source group attraversa split: {key}")
            if row.get("audio_path") is None:
                raise ValueError(f"prepared base row senza audio_path: {row.get('conversation_id')}")
            prepared_by_split[split].append(row)
            sources_by_split[split][source] += 1

    if any(
        sources_by_split[split]["synthetic_v3_clean_asr"] == 0
        for split in ("train", "validation")
    ):
        raise ValueError("base prepared deve contenere clean ASR sia in train sia in validation")

    merged_groups: dict[tuple[str, str], str] = {}
    for split, rows in prepared_by_split.items():
        row_ids = [str(row.get("conversation_id") or "") for row in rows]
        if any(not value for value in row_ids) or len(row_ids) != len(set(row_ids)):
            raise ValueError(f"prepared output {split}: conversation_id mancanti/duplicati")
        for row in rows:
            source = str(row.get("source") or "")
            group = str(row.get("source_group_id") or row.get("topic_group_id") or row.get("conversation_id") or "")
            key = (source, group)
            prior_split = merged_groups.setdefault(key, split)
            if prior_split != split:
                raise ValueError(f"prepared v5 source group attraversa split: {key}")

    output_root.mkdir(parents=True)
    split_report: dict[str, Any] = {}
    for split, rows in prepared_by_split.items():
        with (output_root / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        split_report[split] = {
            "conversations": len(rows),
            "conversations_by_source": dict(sorted(sources_by_split[split].items())),
            "source_groups": len({
                (str(row.get("source")), str(row.get("source_group_id") or row.get("topic_group_id") or row.get("conversation_id")))
                for row in rows
            }),
        }
    report = {
        "schema_version": 1,
        "purpose": "smoke" if is_smoke_candidate else "candidate",
        "profile": _V5_PROFILE,
        "profile_version": _V5_PROFILE_VERSION,
        "v5_manifest": str(manifest_path),
        "base_prepared_dir": str(base_root),
        "v5_wav_integrity": "full decode, PCM16/stereo/16k, duration and SHA256 verified",
        "v5_smoke_rows_included": is_smoke_candidate,
        "group_split_disjoint": True,
        "splits": split_report,
    }
    (output_root / "prepared_v5_clean_asr_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "build_action_conditioned_records", "build_backchannel_lexicon",
    "prepare_action_conditioned_dataset", "prepare_audio_asr_mix_dataset",
    "prepare_audio_grounded_v5_dataset", "write_backchannel_lexicon",
]
