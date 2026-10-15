"""Readers for synthetic manifests and native SmoothConv/DuplexConv layouts.

The external corpora provide ASR/alignment evidence only. Their action-like
fields are deliberately not promoted to DuplexControlHead targets.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .schema import ACTIONS, canonicalize_conversation, sha256_file

_VALID_SPLITS = {"train": "train", "training": "train", "validation": "validation", "valid": "validation", "val": "validation", "dev": "validation", "test": "test"}


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON non valido") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: ogni riga deve essere un object")
            yield value


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(_read_jsonl(path))


def _find_jsonl(root: str | Path, names: Sequence[str]) -> Path:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return path
    candidates: list[Path] = []
    for name in names:
        candidates.extend((path / name, path / "data" / name))
    # The CLI accepts either a corpus directory or the repository root. Resolve
    # only the known corpus layouts; never scan media or annotation contents.
    fixed_layouts = {
        "manifest.jsonl": ("data/synthetic_it_t6_v3/manifest.jsonl",),
        "smoothconv.slice.jsonl": (),
        "duplexconv.slice.jsonl": (),
    }
    for name in names:
        candidates.extend(path / relative for relative in fixed_layouts.get(name, ()))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    for name in names:
        prefix = "smoothconv" if name == "smoothconv.slice.jsonl" else "duplexconv" if name == "duplexconv.slice.jsonl" else None
        if prefix is None:
            continue
        hf_slices = path / "data" / "hf_slices"
        if not hf_slices.is_dir() and path.name == "hf_slices":
            hf_slices = path
        matches = sorted(candidate.resolve() for candidate in hf_slices.glob(f"{prefix}*/{name}") if candidate.is_file()) if hf_slices.is_dir() else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"piu directory {prefix} trovate sotto {hf_slices}; passare la root esatta")
    raise FileNotFoundError(f"nessuno tra {', '.join(names)} trovato sotto {path}")


def _resolve_path(raw: Any, *, bases: Sequence[Path], required: bool = False) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        if required:
            raise FileNotFoundError("percorso media mancante")
        return None
    supplied = Path(raw).expanduser()
    candidates = [supplied] if supplied.is_absolute() else [base / supplied for base in bases]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(f"media non trovato: {raw}")
    return None


def _path_bases(manifest_path: Path, root: Path) -> list[Path]:
    bases = [manifest_path.parent, root, Path.cwd()]
    # Manifests often store repository-relative `data/...` paths. Try only a
    # few lexical ancestors; never walk the corpus looking for a basename.
    for ancestor in list(manifest_path.parents)[:5]:
        if ancestor not in bases:
            bases.append(ancestor)
    return bases


def _audio_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.getnframes()
            width = handle.getsampwidth()
        if rate <= 0 or channels <= 0:
            return {}
        return {
            "sample_rate_hz": rate,
            "channels": channels,
            "duration_ms_audio": frames * 1000.0 / rate,
            "sample_width_bytes": width,
        }
    except (OSError, EOFError, wave.Error):
        return {}


def _decode_pcm_wav(path: Path) -> tuple[Any, int, float] | None:
    """Decode one PCM WAV into a conversation-local channel-majority frame view."""
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        rate, channels, width, total = handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes()
        if rate <= 0 or channels <= 0 or width not in (1, 2, 3, 4):
            return None
        raw = handle.readframes(total)
    expected = total * channels * width
    if expected <= 0 or len(raw) != expected:
        return None
    if width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        scale = 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2")
        scale = 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        samples = (samples ^ 0x800000) - 0x800000
        scale = 8388608.0
    else:
        samples = np.frombuffer(raw, dtype="<i4")
        scale = 2147483648.0
    return samples.reshape(total, channels), int(rate), scale


def _channel_rms(
    path: Path | None,
    start_ms: Any,
    end_ms: Any,
    *,
    max_seconds: float = 2.0,
    pcm_cache: dict[Path, tuple[Any, int, float]] | None = None,
) -> list[float] | None:
    """Compute normalized RMS for a span, reusing one decoded conversation WAV."""
    if path is None or not _number(start_ms) or not _number(end_ms) or end_ms <= start_ms:
        return None
    try:
        import numpy as np

        key = path.resolve()
        decoded = pcm_cache.get(key) if pcm_cache is not None else None
        if decoded is None:
            decoded = _decode_pcm_wav(path)
            if decoded is None:
                return None
            if pcm_cache is not None:
                pcm_cache[key] = decoded
        samples, rate, scale = decoded
        total, channels = samples.shape
        start = max(0, min(total, int(float(start_ms) * rate / 1000)))
        end = max(start, min(total, int(float(end_ms) * rate / 1000)))
        cap = int(max_seconds * rate)
        if end - start > cap:
            start += (end - start - cap) // 2
            end = start + cap
        if end <= start:
            return [0.0] * channels
        block = samples[start:end].astype(np.float64, copy=False)
        return (np.sqrt(np.mean(block * block, axis=0)) / scale).tolist()
    except (OSError, EOFError, wave.Error, ValueError, OverflowError):
        return None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _split(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _VALID_SPLITS.get(value.strip().lower())


def _source_group(source: str, row: Mapping[str, Any], fallback: str) -> str:
    explicit_cross = row.get("cross_dataset_source_group_id")
    value = explicit_cross or row.get("source_group_id") or row.get("source_key_sha256") or row.get("source_key") or fallback
    value = str(value)
    if explicit_cross:
        return value
    prefix = source + ":"
    return value if value.startswith(prefix) else prefix + value


def _canonical_raw(row: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    canonical = canonicalize_conversation(row, base_dir=base_dir)
    audio_path = canonical.get("audio_path")
    if audio_path and not Path(audio_path).is_file():
        raise FileNotFoundError(f"media audio non trovato: {audio_path}")
    return canonical


def read_annotated_jsonl(path: str | Path, *, source_name: str | None = None) -> list[dict[str, Any]]:
    """Compatibility reader for already-canonical annotation JSONL."""
    source_path = Path(path).resolve()
    result = []
    for row in _read_jsonl(source_path):
        if source_name is not None:
            row["source"] = source_name
        source = str(row.get("source", source_name or "")).lower()
        if source_name is None and not any(name in source for name in ("smoothconv", "duplexconv")):
            raise ValueError(f"{source_path}: source deve identificare SmoothConv o DuplexConv")
        row.setdefault("source_group_id", row.get("cross_dataset_source_group_id", row.get("conversation_id")))
        info = _audio_info(_resolve_path(row.get("audio_path", row.get("media_path", row.get("wave_path"))), bases=[source_path.parent]))
        if "duration_ms" not in row and "duration_s" not in row:
            if info:
                row["duration_ms"] = info["duration_ms_audio"]
                row.setdefault("sample_rate_hz", info["sample_rate_hz"])
                row.setdefault("channels", info["channels"])
        result.append(canonicalize_conversation(row, base_dir=source_path.parent))
    return result


def _synth_channel_roles(channel_map: Any) -> tuple[dict[str, str], dict[str, int]]:
    normalized: dict[str, str] = {}
    role_to_channel: dict[str, int] = {}
    if isinstance(channel_map, Mapping):
        for raw_index, raw_name in channel_map.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            name = str(raw_name).strip().lower()
            role = "user" if "user" in name else "assistant" if "assistant" in name else name
            normalized[str(index)] = role
            if role in ("user", "assistant"):
                role_to_channel.setdefault(role, index)
    return normalized, role_to_channel


def _synthetic_language_eligible(entry: Mapping[str, Any], utterances: Any) -> bool:
    """Honor a source's explicit semantic flag, with a narrow legacy fallback."""
    explicit = entry.get("language_training_eligible")
    if isinstance(explicit, bool):
        return explicit
    if entry.get("script_version") == "synthetic-it-t6-v3.0.0":
        # This saved v3 corpus has confirmed truncated/character-only topic slots.
        return False
    if not isinstance(utterances, Sequence) or isinstance(utterances, (str, bytes)) or not utterances:
        return False
    return all(
        isinstance(utterance, Mapping)
        and isinstance(utterance.get("text"), str)
        and bool(utterance["text"].strip())
        for utterance in utterances
    )


# Exact, slot-free text in the frozen v3 authored profiles. Backchannels are
# intentionally absent because v3 cropped their audio to 350 ms.
_V3_CLEAN_ASR_TEMPLATES: dict[str, dict[str, str]] = {
    "agenda_negotiation": {
        "user_ack": "Va bene, tengo a mente la scaletta.",
    },
    "service_recovery": {
        "user_ack": "Chiaro, ho registrato la segnalazione.",
        "assistant_yield_response": "Mi interrompo, verifico la segnalazione e riprendo con una proposta aggiornata.",
    },
    "resource_allocation": {
        "user_ack": "Perfetto, applico quel criterio.",
        "assistant_yield_response": "Mi fermo e ricontrollo il limite; non assegno risorse finché il dato non è confermato.",
    },
    "schedule_adjustment": {
        "user_ack": "Va bene, attendo la verifica del calendario.",
        "assistant_cont_1": "Verifichiamo disponibilità, durata e dipendenze prima di aggiornare il calendario.",
        "assistant_yield_response": "Mi fermo sul calendario e controllo il conflitto prima di confermare l'orario.",
    },
    "event_logistics": {
        "user_ack": "Ricevuto, segno la conferma come aperta.",
        "assistant_yield_response": "Mi fermo con l'organizzazione finché non riceviamo la conferma mancante.",
    },
    "shared_space": {
        "user_ack": "Capito, ricontrollo la disponibilità.",
        "assistant_yield_response": "Mi fermo, ricontrollo l'occupazione e propongo un'alternativa appena ho i dati.",
    },
    "lesson_delivery": {
        "user_ack": "Va bene, aspetto la nuova sequenza.",
        "assistant_cont_1": "Dopo la spiegazione raccogliamo domande e scegliamo un esempio adatto al tempo disponibile.",
        "assistant_yield_response": "Mi fermo qui, controllo i materiali e riparto da una sequenza che il gruppo possa seguire.",
    },
    "volunteer_handoff": {
        "user_ack": "Ricevuto, non procedo finché non arriva l'aggiornamento.",
        "assistant_cont_1": "Raccogliamo ciò che è già deciso e separiamolo dalle questioni ancora senza risposta.",
        "assistant_yield_response": "Mi fermo e aggiorno il passaggio di consegne prima che il gruppo riceva indicazioni.",
    },
}


def read_synthetic_v3_clean_asr(
    root_or_path: str | Path,
    *,
    summary_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Recover only complete, exact, slot-free v3 ASR utterances.

    The result keeps the original conversation/family/split identity, but it is
    role-free and has no action frames. The optional summary is populated in
    the same manifest pass for the caller's preparation report.
    """
    manifest_path = _find_jsonl(root_or_path, ("manifest.jsonl",))
    root = Path(root_or_path).resolve() if Path(root_or_path).is_dir() else manifest_path.parent
    bases = _path_bases(manifest_path, root)
    conversations: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "conversations_seen": 0,
        "eligible_conversations": 0,
        "utterances_seen": 0,
        "eligible_utterances": 0,
        "eligible_speech_seconds": 0.0,
        "eligible_speech_seconds_by_split": {},
        "quarantined_utterances": 0,
        "quarantined_by_reason": {},
    }

    def quarantine(reason: str) -> None:
        summary["quarantined_utterances"] += 1
        counts = summary["quarantined_by_reason"]
        counts[reason] = counts.get(reason, 0) + 1

    for index, entry in enumerate(_read_jsonl(manifest_path)):
        summary["conversations_seen"] += 1
        raw_utterances = entry.get("utterances", entry.get("transcript_spans", ())) or ()
        utterances = list(raw_utterances) if isinstance(raw_utterances, Sequence) and not isinstance(raw_utterances, (str, bytes)) else []
        summary["utterances_seen"] += len(utterances)
        conversation_id = str(entry.get("conversation_id") or entry.get("dialogue_id") or f"synthetic-it-v3-{index:08d}")
        if entry.get("script_version") != "synthetic-it-t6-v3.0.0":
            for _ in utterances:
                quarantine("unsupported_script_version")
            continue

        family_id = entry.get("template_family_id")
        expected_templates = _V3_CLEAN_ASR_TEMPLATES.get(str(family_id), {})
        audio_path = _resolve_path(entry.get("audio_path", entry.get("media_path", entry.get("wave_path"))), bases=bases)
        info = _audio_info(audio_path)
        declared_rate = entry.get("sample_rate_hz", entry.get("sample_rate", info.get("sample_rate_hz")))
        declared_channels = entry.get("channels", info.get("channels"))
        audio_valid = bool(
            audio_path is not None
            and info.get("sample_rate_hz", 0) > 0
            and info.get("channels") == 2
            and info.get("duration_ms_audio", 0) > 0
            and _number(declared_rate)
            and int(declared_rate) == info.get("sample_rate_hz")
            and declared_channels == 2
        )
        channel_names, role_to_channel = _synth_channel_roles(entry.get("channel_map", {}))
        channel_map_valid = (
            channel_names == {"0": "user", "1": "assistant"}
            and role_to_channel == {"user": 0, "assistant": 1}
        )
        segments: list[dict[str, Any]] = []

        for utterance in utterances:
            reason = None
            if not isinstance(utterance, Mapping):
                reason = "malformed_utterance"
            elif not audio_valid:
                reason = "invalid_stereo_audio"
            elif not channel_map_valid:
                reason = "unverified_channel_map"
            else:
                utterance_id = str(utterance.get("utterance_id", utterance.get("segment_id", "")))
                speaker = str(utterance.get("speaker", "")).strip().lower()
                start_s, end_s = utterance.get("start_s"), utterance.get("end_s")
                measured_s = utterance.get("measured_duration_s")
                if not (_number(start_s) and _number(end_s) and _number(measured_s) and measured_s > 0 and end_s > start_s):
                    reason = "invalid_time_metadata"
                elif start_s < 0 or end_s * 1000 > float(info["duration_ms_audio"]) + 1e-6:
                    reason = "out_of_range_time"
                elif abs((end_s - start_s - measured_s) * 1000) > 80.0:
                    reason = "cropped_utterance"
                else:
                    expected_speaker = "user" if utterance_id == "user_ack" else "assistant"
                    expected_text = expected_templates.get(utterance_id)
                    text = utterance.get("text")
                    if expected_text is None:
                        reason = "slot_bearing_or_unlisted_template"
                    elif not isinstance(text, str) or text != expected_text:
                        reason = "template_text_mismatch"
                    elif speaker != expected_speaker:
                        reason = "utterance_speaker_mismatch"
                    else:
                        channel_index = role_to_channel[expected_speaker]
                        segments.append(
                            {
                                "segment_id": utterance_id,
                                "source_utterance_id": utterance_id,
                                "text": text,
                                "start_ms": float(start_s) * 1000.0,
                                "end_ms": float(end_s) * 1000.0,
                                "measured_duration_s": float(measured_s),
                                "task_kind": "asr",
                                "role": None,
                                "channel_index": channel_index,
                                "source_track_index": channel_index,
                                "audio_channel_verified": True,
                                "text_mask": True,
                                "time_mask": True,
                            }
                        )
                        summary["eligible_utterances"] += 1
                        span_seconds = float(end_s - start_s)
                        summary["eligible_speech_seconds"] += span_seconds
                        split_name = _split(entry.get("split")) or "unassigned"
                        split_counts = summary["eligible_speech_seconds_by_split"]
                        split_counts[split_name] = split_counts.get(split_name, 0.0) + span_seconds
            if reason is not None:
                quarantine(reason)

        if segments:
            duration_ms = float(info["duration_ms_audio"])
            raw = {
                # v3 role-free ASR can share the original synthetic ID with a
                # newer gold conversation. Keep split/family identity, while
                # namespacing this separate auxiliary task's row identity.
                "conversation_id": f"synthetic_v3_clean_asr:{conversation_id}",
                "original_conversation_id": conversation_id,
                "source_group_id": str(family_id or entry.get("source_group_id") or conversation_id),
                "source": "synthetic_v3_clean_asr",
                "synthetic": False,
                "language_training_eligible": True,
                "script_version": entry.get("script_version"),
                "source_manifest_path": str(manifest_path),
                "split": _split(entry.get("split")),
                "duration_ms": duration_ms,
                "sample_rate_hz": int(info["sample_rate_hz"]),
                "channels": 2,
                "channel_map": channel_names,
                "audio_path": str(audio_path),
                "audio_sha256": entry.get("audio_sha256", entry.get("media_sha256", entry.get("sha256"))),
                "template_family_id": family_id,
                "turn_pattern_id": entry.get("turn_pattern_id"),
                "segments": segments,
                "events": [],
                "frames": [],
            }
            conversations.append(_canonical_raw(raw, base_dir=manifest_path.parent))
            summary["eligible_conversations"] += 1

    if summary_out is not None:
        summary_out.clear()
        summary_out.update(summary)
    return conversations


def read_synthetic_v3_manifest(root_or_path: str | Path) -> list[dict[str, Any]]:
    """Read the generator's JSONL manifest while preserving explicit labels/splits."""
    manifest_path = _find_jsonl(root_or_path, ("manifest.jsonl",))
    root = Path(root_or_path).resolve() if Path(root_or_path).is_dir() else manifest_path.parent
    bases = _path_bases(manifest_path, root)
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(_read_jsonl(manifest_path)):
        conversation_id = str(entry.get("conversation_id") or entry.get("dialogue_id") or f"synthetic-it-v3-{index:08d}")
        audio_path = _resolve_path(entry.get("audio_path", entry.get("media_path", entry.get("wave_path"))), bases=bases)
        info = _audio_info(audio_path)
        sample_rate = int(entry.get("sample_rate_hz", entry.get("sample_rate", info.get("sample_rate_hz", 16000))))
        if _number(entry.get("duration_ms")):
            duration_ms = float(entry["duration_ms"])
        else:
            duration_ms = float(entry.get("duration_s", entry.get("duration_seconds", info.get("duration_ms_audio", 0) / 1000))) * 1000
        if info.get("duration_ms_audio") and abs(duration_ms - info["duration_ms_audio"]) > max(80.0, 1000.0 / max(sample_rate, 1)):
            # Media bounds are authoritative for loss masks, while declared
            # duration remains available for diagnosis.
            declared_duration_ms = duration_ms
            duration_ms = info["duration_ms_audio"]
        else:
            declared_duration_ms = None
        raw_utterances = entry.get("utterances", entry.get("transcript_spans", ())) or ()
        channel_names, role_to_channel = _synth_channel_roles(entry.get("channel_map", {}))
        segments: list[dict[str, Any]] = []
        for utterance_index, utterance in enumerate(raw_utterances):
            if not isinstance(utterance, Mapping):
                continue
            segment = dict(utterance)
            speaker = str(utterance.get("speaker", "")).strip().lower()
            role = speaker if speaker in ("user", "assistant") else None
            start_s, end_s = utterance.get("start_s"), utterance.get("end_s")
            start_ms = float(start_s) * 1000 if _number(start_s) else utterance.get("start_ms")
            end_ms = float(end_s) * 1000 if _number(end_s) else utterance.get("end_ms")
            duration_value = utterance.get("measured_duration_s")
            observed_ms = (float(end_ms) - float(start_ms)) if _number(start_ms) and _number(end_ms) else -1
            full_crop_text = (
                _number(duration_value)
                and observed_ms >= float(duration_value) * 1000 - 80.0
            ) if duration_value is not None else observed_ms > 0
            channel = role_to_channel.get(role) if role else None
            segment.update(
                {
                    "segment_id": str(utterance.get("utterance_id", utterance.get("segment_id", utterance_index))),
                    "speaker": utterance.get("speaker"),
                    "script_role": utterance.get("role"),
                    "role": role,
                    "channel_index": channel,
                    "task_kind": "chat",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "measured_duration_s": duration_value,
                    "text_mask": bool(utterance.get("text_mask", True) and full_crop_text and role is not None),
                    "time_mask": bool(utterance.get("time_mask", True)),
                    "action_mask": False,
                }
            )
            segments.append(segment)

        action_segments = entry.get("action_segments", ()) or ()
        frame_count = int(math.ceil(duration_ms / 80.0))
        frames: list[dict[str, Any]] = []
        for frame_id in range(frame_count):
            start_ms = frame_id * 80.0
            end_ms = start_ms + 80.0
            labels = set()
            raw_labels = []
            for action_segment in action_segments:
                if not isinstance(action_segment, Mapping):
                    continue
                a_start = action_segment.get("start_ms")
                a_end = action_segment.get("end_ms")
                if a_start is None and _number(action_segment.get("start_sample")):
                    a_start = float(action_segment["start_sample"]) * 1000 / sample_rate
                if a_end is None and _number(action_segment.get("end_sample")):
                    a_end = float(action_segment["end_sample"]) * 1000 / sample_rate
                if not _number(a_start) or not _number(a_end) or a_start >= end_ms or a_end <= start_ms:
                    continue
                label = action_segment.get("action", action_segment.get("target_action"))
                raw_labels.append(label)
                if label in ACTIONS:
                    labels.add(label)
            action = next(iter(labels)) if len(labels) == 1 and len(raw_labels) == 1 else None
            text_parts = []
            for segment in segments:
                text = segment.get("text")
                if isinstance(text, str) and _number(segment.get("start_ms")) and _number(segment.get("end_ms")) and segment["start_ms"] < end_ms and segment["end_ms"] > start_ms:
                    text_parts.append(text.strip())
            frames.append(
                {
                    "frame_id": frame_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "valid_ms": max(0.0, min(end_ms, duration_ms) - start_ms),
                    "valid_samples": int(round(max(0.0, min(end_ms, duration_ms) - start_ms) * sample_rate / 1000)),
                    "target_action": action,
                    "raw_target_action": raw_labels[0] if len(raw_labels) == 1 else raw_labels,
                    "action_mask": action is not None,
                    "text": " ".join(part for part in text_parts if part),
                }
            )
        raw = {
            "conversation_id": conversation_id,
            "source_group_id": str(entry.get("template_family_id", entry.get("source_group_id", conversation_id))),
            "source": str(entry.get("dataset_id") or "synthetic_it_t6"),
            "script_version": entry.get("script_version"),
            "language_training_eligible": _synthetic_language_eligible(entry, raw_utterances),
            "synthetic": True,
            "source_manifest_path": str(manifest_path),
            "split": _split(entry.get("split")),
            "duration_ms": duration_ms,
            "declared_duration_ms": declared_duration_ms,
            "sample_rate_hz": sample_rate,
            "channels": int(entry.get("channels", info.get("channels", 1))),
            "channel_map": channel_names,
            "audio_path": str(audio_path) if audio_path else None,
            "audio_sha256": entry.get("audio_sha256", entry.get("media_sha256", entry.get("sha256"))),
            "template_family_id": entry.get("template_family_id"),
            "turn_pattern_id": entry.get("turn_pattern_id"),
            "source_action_segments": [dict(segment) for segment in action_segments if isinstance(segment, Mapping)],
            "segments": segments,
            "frames": frames,
        }
        rows.append(_canonical_raw(raw, base_dir=manifest_path.parent))
    return rows


def _smooth_evidence_path(root: Path) -> Path | None:
    candidates = [
        root / "data/training_gates/chinese_label_evidence_smoothconv_20260930_full40h.jsonl",
        root / "training_gates/chinese_label_evidence_smoothconv_20260930_full40h.jsonl",
        root.parent.parent / "training_gates/chinese_label_evidence_smoothconv_20260930_full40h.jsonl",
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _index_rows(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value):
                index.setdefault(str(value), []).append(row)
    return index


def _match_row(row: Mapping[str, Any], index: Mapping[str, list[Mapping[str, Any]]], keys: Sequence[str]) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        for match in index.get(str(value), ()):
            if id(match) not in seen:
                seen.add(id(match))
                found.append(match)
    return found


def _merge_group_id(source: str, row: Mapping[str, Any], fallback: str) -> str:
    return _source_group(source, row, fallback)


def _event_overlap(start: float, end: float, others: Sequence[tuple[float, float]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in others)


def _infer_channel_map(
    path: Path | None,
    tracks: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    channel_count: int,
    start_key: str,
    end_key: str,
    other_time_units_ms: bool = True,
    pcm_cache: dict[Path, tuple[Any, int, float]] | None = None,
) -> dict[int, int]:
    """Infer track to physical channel only from clean timed spans and RMS."""
    if path is None or channel_count < 1 or len(tracks) > channel_count:
        return {}
    audio_duration_ms = _audio_info(path).get("duration_ms_audio")
    all_spans: list[tuple[int, float, float]] = []
    for track, events in tracks.items():
        for event in events:
            start, end = event.get(start_key), event.get(end_key)
            if _number(start) and _number(end) and end > start:
                start, end = float(start), float(end)
                if audio_duration_ms is not None:
                    start, end = max(0.0, start), min(float(audio_duration_ms), end)
                if end > start:
                    all_spans.append((track, start, end))
    scores: dict[int, list[float]] = {track: [0.0] * channel_count for track in tracks}
    usable_windows: dict[int, int] = {track: 0 for track in tracks}
    for track, events in tracks.items():
        for event in events:
            start, end = event.get(start_key), event.get(end_key)
            if not _number(start) or not _number(end) or end <= start:
                continue
            clean_start, clean_end = float(start), float(end)
            if audio_duration_ms is not None:
                clean_start, clean_end = max(0.0, clean_start), min(float(audio_duration_ms), clean_end)
            if clean_end <= clean_start:
                continue
            # Subtract only the portions overlapped by a different track. A
            # clipped/out-of-range annotation elsewhere must not erase clean
            # evidence from the valid part of this event.
            overlaps = sorted(
                (max(clean_start, other_start), min(clean_end, other_end))
                for other_track, other_start, other_end in all_spans
                if other_track != track and clean_start < other_end and clean_end > other_start
            )
            clean_spans: list[tuple[float, float]] = []
            cursor = clean_start
            for overlap_start, overlap_end in overlaps:
                if overlap_start - cursor >= 10.0:
                    clean_spans.append((cursor, overlap_start))
                cursor = max(cursor, overlap_end)
            if clean_end - cursor >= 10.0:
                clean_spans.append((cursor, clean_end))
            if not overlaps and clean_end - clean_start >= 10.0:
                clean_spans.append((clean_start, clean_end))
            for span_start, span_end in clean_spans:
                rms = _channel_rms(path, span_start, span_end, pcm_cache=pcm_cache)
                if rms is None or len(rms) != channel_count:
                    continue
                max_rms = max(rms, default=0.0)
                if max_rms < 1e-4:
                    continue
                for channel, value in enumerate(rms):
                    scores[track][channel] += value * value
                usable_windows[track] += 1
    candidates: dict[int, int] = {}
    for track, values in scores.items():
        ordered = sorted(range(channel_count), key=lambda ch: values[ch], reverse=True)
        first = values[ordered[0]] if ordered else 0.0
        second = values[ordered[1]] if len(ordered) > 1 else 0.0
        if usable_windows[track] and first > 0 and (second == 0 or first >= second * 1.44):
            candidates[track] = ordered[0]
    # Conflicting tracks are ambiguous; do not force a one-to-one assignment.
    inverse: dict[int, list[int]] = {}
    for track, channel in candidates.items():
        inverse.setdefault(channel, []).append(track)
    return {track: channel for track, channel in candidates.items() if len(inverse[channel]) == 1}


def _duration_and_audio(row: Mapping[str, Any], audio_path: Path | None, duration_keys: Sequence[str]) -> tuple[float, dict[str, Any]]:
    info = _audio_info(audio_path)
    duration_key = next((key for key in duration_keys if _number(row.get(key))), None)
    if duration_key is not None:
        multiplier = 1000 if duration_key.endswith("_s") or duration_key.endswith("seconds") else 1
        duration_ms = float(row[duration_key]) * multiplier
    else:
        duration_ms = float(info.get("duration_ms_audio", 0))
    if info.get("duration_ms_audio"):
        duration_ms = float(info["duration_ms_audio"])
    return duration_ms, info


def _native_state(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized.startswith("<|") and normalized.endswith("|>"):
        normalized = normalized[2:-2]
    return normalized if normalized in {"complete", "incomplete", "backchannel", "wait"} else None


def _language_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"cn", "zh", "zh-cn", "chinese"}:
        return "Chinese"
    if normalized in {"en", "english"}:
        return "English"
    if normalized in {"it", "italian", "italiano"}:
        return "Italian"
    if normalized in {"unknown", "und", "n/a"}:
        return None
    return value.strip()


def _has_native_interruption(value: Any) -> bool:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return any(isinstance(item, str) and item.strip().lower() == "interruption" for item in values)


def _physical_pair(conversation: Mapping[str, Any]) -> dict[int, int] | None:
    """Return source-track -> physical-channel only for a verified stereo pair."""
    if conversation.get("channels") != 2:
        return None
    segments = conversation.get("segments", ()) or ()
    track_channels: dict[int, set[int]] = {}
    verified_tracks: set[int] = set()
    verified_channels: set[int] = set()
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        track = segment.get("source_track_index")
        channel = segment.get("channel_index")
        if isinstance(track, int) and not isinstance(track, bool) and isinstance(channel, int) and not isinstance(channel, bool):
            track_channels.setdefault(track, set()).add(channel)
        if segment.get("audio_channel_verified") is True:
            if not (isinstance(track, int) and not isinstance(track, bool) and isinstance(channel, int) and not isinstance(channel, bool)):
                return None
            if segment.get("time_mask") is not True:
                continue
            verified_tracks.add(track)
            verified_channels.add(channel)
    if len(track_channels) != 2 or verified_tracks != set(track_channels) or verified_channels != {0, 1}:
        return None
    if any(len(channels) != 1 for channels in track_channels.values()):
        return None
    mapping = {track: next(iter(channels)) for track, channels in track_channels.items()}
    if set(mapping.values()) != {0, 1}:
        return None
    return mapping


def build_real_conversational_views(
    conversations: Sequence[Mapping[str, Any]],
    *,
    tick_ms: int = 80,
    max_conversations: int | None = 128,
) -> list[dict[str, Any]]:
    """Build two conservative, directed views from verified Smooth/Duplex stereo pairs.

    Physical channels are preserved in ``channel_map`` while assistant/user are
    perspective-only labels. Action supervision comes only from observed timing
    and explicit native turn states. Ambiguous gaps, unknown states, and overlap
    by themselves remain masked.
    """
    if isinstance(tick_ms, bool) or not isinstance(tick_ms, int) or tick_ms != 80:
        raise ValueError("le viste conversazionali usano tick da 80 ms")
    if max_conversations is not None and (
        isinstance(max_conversations, bool) or not isinstance(max_conversations, int) or max_conversations <= 0
    ):
        raise ValueError("max_conversations deve essere positivo o None")

    candidate_groups: dict[str, dict[str, Any]] = {}
    for conversation in conversations:
        source = conversation.get("source")
        duration = conversation.get("duration_ms")
        if source not in {"SmoothConv", "DuplexConv"} or not _number(duration) or duration <= 0:
            continue
        pair = _physical_pair(conversation)
        if pair is None:
            continue
        segments = [segment for segment in (conversation.get("segments", ()) or ()) if isinstance(segment, Mapping)]
        complete_by_channel = {0: 0, 1: 0}
        state_counts: set[str] = set()
        backchannels = 0
        verified_text = 0
        for segment in segments:
            if not (segment.get("audio_channel_verified") is True and segment.get("time_mask") is True):
                continue
            channel = segment.get("channel_index")
            if channel not in {0, 1}:
                continue
            state = _native_state(segment.get("native_turn_state")) or _native_state(segment.get("annotation_state"))
            if state:
                state_counts.add(state)
            if segment.get("text_mask") is True and isinstance(segment.get("text"), str) and segment["text"].strip():
                verified_text += 1
                if state == "complete":
                    complete_by_channel[channel] += 1
            if state == "backchannel":
                backchannels += 1
        # Rank by paired complete targets first, then native state coverage,
        # explicit backchannels, and verified timed text. Stable ID breaks ties.
        quality = (
            int(all(complete_by_channel[channel] > 0 for channel in (0, 1))),
            len(state_counts),
            min(backchannels, 16),
            min(sum(complete_by_channel.values()), 32),
            min(verified_text, 64),
        )
        original_id = str(conversation.get("conversation_id", ""))
        group_id = str(conversation.get("source_group_id") or original_id)
        group = candidate_groups.setdefault(group_id, {"rows": [], "quality": [0, 0, 0, 0, 0]})
        group["rows"].append((conversation, pair, quality))
        group["quality"] = [left + right for left, right in zip(group["quality"], quality)]

    ranked_groups = sorted(
        candidate_groups.items(),
        key=lambda item: tuple(-value for value in item[1]["quality"]) + (item[0],),
    )
    groups_by_source_split: dict[tuple[str, str], list[str]] = {}
    for group_id, group in candidate_groups.items():
        buckets = {
            (str(conversation.get("source", "unknown")), str(_split(conversation.get("split")) or "unknown"))
            for conversation, _pair, _quality in group["rows"]
        }
        for bucket in buckets:
            groups_by_source_split.setdefault(bucket, []).append(group_id)
    for group_ids in groups_by_source_split.values():
        group_ids.sort(
            key=lambda group_id: tuple(-value for value in candidate_groups[group_id]["quality"]) + (group_id,)
        )
    selected_group_ids: list[str] = []
    if max_conversations is None:
        selected_group_ids = [group_id for group_id, _group in ranked_groups]
    else:
        cursors = {bucket: 0 for bucket in groups_by_source_split}
        bucket_order = sorted(groups_by_source_split)
        selected_ids: set[str] = set()
        selected_count = 0
        while bucket_order and selected_count < max_conversations:
            added = False
            for bucket in bucket_order:
                group_ids = groups_by_source_split[bucket]
                while cursors[bucket] < len(group_ids) and group_ids[cursors[bucket]] in selected_ids:
                    cursors[bucket] += 1
                if cursors[bucket] >= len(group_ids):
                    continue
                group_id = group_ids[cursors[bucket]]
                cursors[bucket] += 1
                group_size = len(candidate_groups[group_id]["rows"])
                if selected_count + group_size > max_conversations:
                    continue
                selected_ids.add(group_id)
                selected_group_ids.append(group_id)
                selected_count += group_size
                added = True
                if selected_count >= max_conversations:
                    break
            if not added:
                break
    selected: list[tuple[Mapping[str, Any], dict[int, int], str, int, tuple[int, ...]]] = []
    for group_rank, group_id in enumerate(selected_group_ids, start=1):
        group = candidate_groups[group_id]
        group_rows = group["rows"]
        quality = tuple(int(value) for value in group["quality"])
        selected.extend((conversation, pair, group_id, group_rank, quality) for conversation, pair, _ in group_rows)

    output: list[dict[str, Any]] = []
    eligible_source_conversation_count = sum(len(group["rows"]) for group in candidate_groups.values())
    selected_source_conversation_count = len(selected)
    for conversation, track_to_channel, group_id, selection_rank, selection_quality in selected:
        source = conversation.get("source")
        duration = conversation.get("duration_ms")
        segments = [segment for segment in (conversation.get("segments", ()) or ()) if isinstance(segment, Mapping)]
        # A speaker-to-channel link is usable for YIELD only if observed
        # unambiguously on verified physical speech segments.
        speaker_channels: dict[str, set[int]] = {}
        for segment in segments:
            speaker = segment.get("source_speaker_id")
            channel = segment.get("channel_index")
            if (segment.get("audio_channel_verified") is True and segment.get("time_mask") is True
                    and isinstance(speaker, str)
                    and speaker.strip() and speaker.strip().lower() != "unknown"
                    and isinstance(channel, int) and not isinstance(channel, bool) and channel in {0, 1}):
                speaker_channels.setdefault(speaker.strip(), set()).add(channel)

        for local_channel in (0, 1):
            remote_channel = 1 - local_channel
            channel_map = {str(local_channel): "assistant", str(remote_channel): "user"}
            view_segments: list[dict[str, Any]] = []
            local_segments: list[dict[str, Any]] = []
            remote_segments: list[dict[str, Any]] = []
            for original in segments:
                segment = dict(original)
                channel = segment.get("channel_index")
                physically_mapped = (
                    isinstance(channel, int) and not isinstance(channel, bool)
                    and channel in {0, 1}
                    and segment.get("source_track_index") in track_to_channel
                    and track_to_channel[segment["source_track_index"]] == channel
                )
                if physically_mapped:
                    role = "assistant" if channel == local_channel else "user"
                    segment["perspective_role"] = role
                    segment["role"] = role
                    state_raw = segment.get("native_turn_state")
                    if state_raw is None:
                        state_raw = segment.get("annotation_state")
                    segment.setdefault("native_turn_state_raw", state_raw)
                    state = _native_state(state_raw)
                    segment["native_turn_state"] = state
                    if channel == local_channel:
                        segment["task_kind"] = "conversation_response"
                        target_ok = (
                            state == "complete"
                            and segment.get("audio_channel_verified") is True
                            and segment.get("time_mask") is True
                            and segment.get("text_mask") is True
                            and _number(segment.get("start_ms")) and _number(segment.get("end_ms"))
                            and 0 <= segment["start_ms"] < segment["end_ms"] <= duration
                            and isinstance(segment.get("text"), str)
                            and bool(segment["text"].strip())
                        )
                        segment["target_mask"] = bool(target_ok)
                        local_segments.append(segment)
                    else:
                        segment["task_kind"] = "context"
                        segment["target_mask"] = False
                        remote_segments.append(segment)
                else:
                    segment["perspective_role"] = None
                    segment["role"] = None
                    segment["target_mask"] = False
                    segment["task_kind"] = "context"
                # Source roles are annotations, not inferred perspective roles.
                segment.setdefault("source_role", None)
                source_lid = segment.get("language")
                segment.setdefault("source_language_lid", source_lid)
                segment["language"] = _language_name(source_lid) or _language_name(conversation.get("language")) or "Chinese"
                view_segments.append(segment)

            frame_count = int(math.ceil(float(duration) / tick_ms))
            yield_by_frame: dict[int, list[dict[str, Any]]] = {}
            if source == "SmoothConv":
                for local in local_segments:
                    start, end = local.get("start_ms"), local.get("end_ms")
                    local_speaker = local.get("source_speaker_id")
                    other_speaker = local.get("source_other_speaker_id")
                    if not (
                        _native_state(local.get("native_turn_state")) == "incomplete"
                        and _has_native_interruption(local.get("native_other_turn"))
                        and local.get("audio_channel_verified") is True and local.get("time_mask") is True
                        and _number(start) and _number(end) and 0 <= start < end < duration
                        and isinstance(local_speaker, str) and speaker_channels.get(local_speaker.strip()) == {local_channel}
                        and isinstance(other_speaker, str) and speaker_channels.get(other_speaker.strip()) == {remote_channel}
                    ):
                        continue
                    local_still_active = any(
                        other is not local
                        and other.get("audio_channel_verified") is True
                        and _number(other.get("start_ms")) and _number(other.get("end_ms"))
                        and other["start_ms"] <= end < other["end_ms"]
                        for other in local_segments
                    )
                    remote_continues = any(
                        remote.get("audio_channel_verified") is True
                        and remote.get("time_mask") is True
                        and isinstance(remote.get("source_speaker_id"), str)
                        and remote["source_speaker_id"].strip() == other_speaker.strip()
                        and _number(remote.get("start_ms")) and _number(remote.get("end_ms"))
                        and remote["start_ms"] < end < remote["end_ms"]
                        for remote in remote_segments
                    )
                    if local_still_active or not remote_continues:
                        continue
                    frame_id = int(float(end) // tick_ms)
                    if 0 <= frame_id < frame_count:
                        yield_by_frame.setdefault(frame_id, []).append(local)

            candidate_frame_ids: set[int] = set(yield_by_frame)
            for segment in local_segments + remote_segments:
                start_ms, end_ms = segment.get("start_ms"), segment.get("end_ms")
                if (segment.get("audio_channel_verified") is not True or segment.get("time_mask") is not True
                        or not _number(start_ms) or not _number(end_ms) or not 0 <= start_ms < end_ms <= duration):
                    continue
                onset_id = int(float(start_ms) // tick_ms)
                if 0 <= onset_id < frame_count:
                    candidate_frame_ids.add(onset_id)
                interior_start = onset_id + 1
                interior_stop = min(int(math.ceil(float(end_ms) / tick_ms)), frame_count)
                interior_ids = list(range(interior_start, interior_stop))
                if len(interior_ids) <= 2:
                    candidate_frame_ids.update(interior_ids)
                elif interior_ids:
                    first = round((len(interior_ids) - 1) / 3)
                    second = round(2 * (len(interior_ids) - 1) / 3)
                    candidate_frame_ids.add(interior_ids[first])
                    candidate_frame_ids.add(interior_ids[second])

            frames: list[dict[str, Any]] = []
            for frame_id in sorted(candidate_frame_ids):
                start = frame_id * tick_ms
                end = min(float(duration), start + tick_ms)
                def overlaps(segment: Mapping[str, Any]) -> bool:
                    return bool(
                        segment.get("audio_channel_verified") is True
                        and segment.get("time_mask") is True
                        and _number(segment.get("start_ms")) and _number(segment.get("end_ms"))
                        and float(segment["start_ms"]) < end and float(segment["end_ms"]) > start
                    )

                local_active = [segment for segment in local_segments if overlaps(segment)]
                remote_active = [segment for segment in remote_segments if overlaps(segment)]
                action = None
                if len(yield_by_frame.get(frame_id, ())) == 1:
                    action = "YIELD"
                elif not yield_by_frame.get(frame_id):
                    if len(local_active) == 1:
                        active = local_active[0]
                        state = _native_state(active.get("native_turn_state"))
                        onset = start <= float(active["start_ms"]) < end
                        if state == "backchannel" and onset:
                            action = "BACKCHANNEL"
                        elif state in {"complete", "incomplete"}:
                            action = "SPEAK" if onset else "CONTINUE"
                    elif not local_active and len(remote_active) == 1:
                        action = "LISTEN"
                valid_ms = max(0.0, end - start)
                sample_rate = int(conversation.get("sample_rate_hz", 16000))
                frames.append({
                    "frame_id": frame_id,
                    "start_ms": start,
                    "end_ms": end,
                    "valid_ms": valid_ms,
                    "valid_samples": int(round(valid_ms * sample_rate / 1000.0)),
                    "target_action": action,
                    "action_mask": action is not None,
                    "time_mask": valid_ms > 0,
                    "audio_mask": bool(conversation.get("audio_path")),
                    "events": [],
                })

            original_id = str(conversation.get("conversation_id", ""))
            view = dict(conversation)
            view.update({
                "conversation_id": f"{source}:{original_id}:local-channel-{local_channel}",
                "source_group_id": str(conversation.get("source_group_id") or original_id),
                "source": source,
                "source_conversation_id": original_id,
                "view_local_channel": local_channel,
                "view_remote_channel": remote_channel,
                "source_view_selection_rank": selection_rank,
                "source_view_selection_group_quality": list(selection_quality),
                "source_view_max_conversations": max_conversations,
                "source_view_eligible_conversations_seen": eligible_source_conversation_count,
                "source_view_selected_conversations": selected_source_conversation_count,
                "channel_map": channel_map,
                "language": _language_name(conversation.get("language")) or "Chinese",
                "source_language_lid": conversation.get("source_language_lid"),
                "segments": view_segments,
                "frames": frames,
                "tick_ms": tick_ms,
                "control_supervision": "native_turn_state_and_observed_channel_timing",
            })
            output.append(view)
    return output


def read_smoothconv(
    root_or_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    source_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Join SmoothConv slice rows to nested source-event evidence and WAV spans."""
    manifest_path = _find_jsonl(root_or_path, ("smoothconv.slice.jsonl",))
    root = Path(root_or_path).resolve() if Path(root_or_path).is_dir() else manifest_path.parent
    bases = _path_bases(manifest_path, root)
    if evidence_path is None:
        evidence_candidate = _smooth_evidence_path(root)
    else:
        evidence_candidate = Path(evidence_path).expanduser().resolve()
    evidence_rows = _rows(evidence_candidate) if evidence_candidate and evidence_candidate.is_file() else []
    evidence_index = _index_rows(evidence_rows, "source_key", "source_key_sha256", "source_audio_sha256")
    result: list[dict[str, Any]] = []
    for row_index, entry in enumerate(_read_jsonl(manifest_path)):
        key = entry.get("source_key") or entry.get("source_key_sha256") or str(row_index)
        if source_keys is not None and str(key) not in source_keys:
            continue
        matches = _match_row(entry, evidence_index, ("source_key", "source_key_sha256"))
        events_by_id: dict[tuple[Any, ...], dict[str, Any]] = {}
        for match in matches:
            for event in match.get("source_events", ()) or ():
                if not isinstance(event, Mapping):
                    continue
                annotation = event.get("source_annotation_event") or {}
                if not isinstance(annotation, Mapping):
                    annotation = {}
                text = annotation.get("text", annotation.get("transcript", annotation.get("txt", "")))
                start = event.get("source_start_ms", event.get("start_ms", annotation.get("start")))
                end = event.get("source_end_ms", event.get("end_ms", annotation.get("end")))
                channel = event.get("source_channel_index", annotation.get("channelIndex", annotation.get("channel_index")))
                try:
                    channel_index = int(channel)
                except (TypeError, ValueError):
                    channel_index = None
                identity = (event.get("source_event_id"), channel_index, start, end, str(text))
                events_by_id.setdefault(identity, {
                    "segment_id": str(event.get("source_event_id", f"event-{len(events_by_id)}")),
                    "text": str(text) if isinstance(text, str) else "",
                    "start_ms": start,
                    "end_ms": end,
                    "channel_index": channel_index,
                    "source_track_index": channel_index,
                    "source_event_id": event.get("source_event_id"),
                    "candidate_t6_action": event.get("candidate_t6_action"),
                    "source_annotation_event": dict(annotation),
                })
        audio_path = _resolve_path(entry.get("local_path", entry.get("audio_path")), bases=bases)
        duration_ms, info = _duration_and_audio(entry, audio_path, ("duration_seconds", "duration_s", "duration_ms"))
        sample_rate = int(info.get("sample_rate_hz", entry.get("sample_rate_hz", 16000)))
        channel_count = int(info.get("channels", entry.get("channel_count", entry.get("channels", 1))))
        events = list(events_by_id.values())
        tracks: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            track = event.get("source_track_index")
            if isinstance(track, int):
                tracks.setdefault(track, []).append(event)
        pcm_cache: dict[Path, tuple[Any, int, float]] = {}
        channel_map = _infer_channel_map(
            audio_path, tracks, channel_count=channel_count,
            start_key="start_ms", end_key="end_ms", pcm_cache=pcm_cache,
        )
        segments = []
        for event in events:
            raw_channel = event.get("source_track_index")
            mapped = channel_map.get(raw_channel) if isinstance(raw_channel, int) else None
            valid_time = _number(event.get("start_ms")) and _number(event.get("end_ms")) and 0 <= event["start_ms"] < event["end_ms"] <= duration_ms
            event_rms = _channel_rms(
                audio_path, event.get("start_ms"), event.get("end_ms"), pcm_cache=pcm_cache
            ) if mapped is not None else None
            channel_active = bool(event_rms and mapped < len(event_rms) and event_rms[mapped] >= 1e-4)
            segment = dict(event)
            annotation = event.get("source_annotation_event") or {}
            attrs = annotation.get("attributes", {}) if isinstance(annotation, Mapping) else {}
            if not isinstance(attrs, Mapping):
                attrs = {}
            segment.update({
                "native_turn_state": attrs.get("turn"),
                "native_other_turn": attrs.get("other_turn"),
                "native_paralinguistic": attrs.get("speech event"),
                "source_speaker_id": attrs.get("speaker", event.get("source_speaker_ref")),
                "source_other_speaker_id": attrs.get("other_speaker", event.get("source_other_speaker_ref")),
                "source_role": None,
                "language": attrs.get("language", annotation.get("language", entry.get("language"))),
            })
            segment.update({
                "role": None,
                "task_kind": "asr",
                "channel_index": mapped,
                "source_track_index": raw_channel,
                "audio_channel_verified": mapped is not None and channel_active,
                "time_mask": bool(valid_time),
                "text_mask": bool(valid_time and isinstance(event.get("text"), str) and bool(event["text"].strip()) and mapped is not None and channel_active),
                "action_mask": False,
            })
            segments.append(segment)
        split = _split(entry.get("project_split", entry.get("split")))
        canonical = _canonical_raw({
            "conversation_id": str(entry.get("source_key") or entry.get("source_key_sha256") or f"smoothconv-{row_index:08d}"),
            "source_group_id": _merge_group_id("SmoothConv", entry, str(key)),
            "source": "SmoothConv",
            "source_manifest_path": str(manifest_path),
            "split": split,
            "duration_ms": duration_ms,
            "sample_rate_hz": sample_rate,
            "channels": channel_count,
            "channel_map": {},
            "audio_path": str(audio_path) if audio_path else None,
            "audio_sha256": entry.get("sha256", entry.get("source_audio_sha256")),
            "source_key": entry.get("source_key"),
            "source_key_sha256": entry.get("source_key_sha256"),
            "language": _language_name(entry.get("language")) or "Chinese",
            "source_language_lid": entry.get("language"),
            "preparation_warnings": [] if matches else ["evidence_missing"],
            "segments": segments,
            "frames": [],
        }, base_dir=manifest_path.parent)
        result.append(canonical)
    return result


def _resolve_annotation_path(
    entry: Mapping[str, Any],
    pairing: Mapping[str, Any] | None,
    *,
    root: Path,
    bases: Sequence[Path],
    annotation_roots: Sequence[str | Path] | None,
) -> tuple[Path | None, bool]:
    raw_candidates: list[Any] = []
    if pairing:
        raw_candidates.extend((pairing.get("annotation_json_path"), pairing.get("annotation_path"), pairing.get("local_annotation_path")))
    raw_candidates.extend((entry.get("annotation_json_path"), entry.get("annotation_path")))
    search_bases = list(bases)
    search_bases.extend([root / "annotation_pairing_review", root / "annotation_pairing_review" / "annotations", root / "annotations"])
    search_bases.extend(Path(value).expanduser().resolve() for value in (annotation_roots or ()))
    resolved: set[Path] = set()
    absolute_matches: set[Path] = set()
    for raw in raw_candidates:
        if not isinstance(raw, str) or not raw.strip():
            continue
        supplied = Path(raw).expanduser()
        candidates = [supplied] if supplied.is_absolute() else [base / supplied for base in search_bases]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".json":
                resolved.add(candidate.resolve())
                if supplied.is_absolute():
                    absolute_matches.add(candidate.resolve())
    if absolute_matches:
        if len(absolute_matches) == 1:
            return next(iter(absolute_matches)), False
        return None, True
    if resolved:
        if len(resolved) == 1:
            return next(iter(resolved)), False
        return None, True
    basenames = []
    for raw in raw_candidates:
        if isinstance(raw, str) and raw.strip():
            basenames.append(Path(raw).name)
    for base in search_bases:
        for basename in basenames:
            candidate = base / basename
            if candidate.is_file() and candidate not in resolved:
                resolved.add(candidate.resolve())
    # If a supplied root is an annotation directory, don't recursively scan it;
    # its direct basename matches are enough to detect collisions.
    unique = sorted(resolved)
    if len(unique) == 1:
        return unique[0], False
    if len(unique) > 1:
        return None, True
    return None, False


def _annotation_text(event: Mapping[str, Any]) -> str:
    labels = event.get("labels", "")
    if isinstance(labels, Mapping):
        value = labels.get("txt", labels.get("text", ""))
    else:
        value = labels
    return value if isinstance(value, str) else ""


def read_duplexconv(
    root_or_path: str | Path,
    *,
    pairing_path: str | Path | None = None,
    annotation_roots: Sequence[str | Path] | None = None,
    source_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read DuplexConv pairing review and nested per-track ASR annotations."""
    manifest_path = _find_jsonl(root_or_path, ("duplexconv.slice.jsonl",))
    root = Path(root_or_path).resolve() if Path(root_or_path).is_dir() else manifest_path.parent
    bases = _path_bases(manifest_path, root)
    if pairing_path is not None:
        pairing_manifest = Path(pairing_path).expanduser().resolve()
    else:
        candidates = [
            root / "annotation_pairing_review/pairing_review.jsonl",
            root / "pairing_review.jsonl",
            root / "data/hf_slices/annotation_pairing_review/pairing_review.jsonl",
        ]
        pairing_manifest = next((p.resolve() for p in candidates if p.is_file()), None)
        if pairing_manifest is None:
            hf_slices = root / "data" / "hf_slices"
            if not hf_slices.is_dir() and root.name == "hf_slices":
                hf_slices = root
            discovered = sorted(
                candidate.resolve()
                for candidate in hf_slices.glob("duplexconv*/annotation_pairing_review/pairing_review.jsonl")
                if candidate.is_file()
            ) if hf_slices.is_dir() else []
            if len(discovered) == 1:
                pairing_manifest = discovered[0]
            elif len(discovered) > 1:
                raise FileNotFoundError(f"piu pairing_review.jsonl trovati sotto {hf_slices}; passare --duplexconv-pairing")
    pairing_rows = _rows(pairing_manifest) if pairing_manifest else []
    pairing_index = _index_rows(pairing_rows, "source_key", "source_key_sha256", "audio_sha256")
    result: list[dict[str, Any]] = []
    for row_index, entry in enumerate(_read_jsonl(manifest_path)):
        key = entry.get("source_key") or entry.get("source_key_sha256") or str(row_index)
        if source_keys is not None and str(key) not in source_keys:
            continue
        matches = _match_row(entry, pairing_index, ("source_key", "source_key_sha256"))
        pairing = matches[0] if len(matches) == 1 else None
        annotation_path, collision = _resolve_annotation_path(
            entry,
            pairing,
            root=root,
            bases=bases,
            annotation_roots=annotation_roots,
        ) if pairing else (None, False)
        annotation: Mapping[str, Any] = {}
        if annotation_path:
            try:
                loaded = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, Mapping):
                annotation = loaded
        audio_path = _resolve_path(entry.get("local_path", entry.get("audio_local_path", entry.get("audio_path"))), bases=bases)
        duration_ms, info = _duration_and_audio(entry, audio_path, ("duration_seconds", "duration_s", "duration_ms"))
        sample_rate = int(info.get("sample_rate_hz", entry.get("sample_rate_hz", annotation.get("fs", 16000))))
        channel_count = int(info.get("channels", entry.get("channel_count", entry.get("channels", 1))))
        asr = annotation.get("asr", ()) or ()
        tracks: dict[int, list[dict[str, Any]]] = {}
        for track_index, events in enumerate(asr if isinstance(asr, list) else ()):
            if not isinstance(events, list):
                continue
            tracks[track_index] = [event for event in events if isinstance(event, Mapping)]
        pcm_cache: dict[Path, tuple[Any, int, float]] = {}
        channel_map = _infer_channel_map(
            audio_path,
            tracks,
            channel_count=channel_count,
            start_key="startInMs",
            end_key="endInMs",
            pcm_cache=pcm_cache,
        )
        segments: list[dict[str, Any]] = []
        for track_index, events in tracks.items():
            mapped = channel_map.get(track_index)
            for event_index, event in enumerate(events):
                start, end = event.get("startInMs"), event.get("endInMs")
                text = _annotation_text(event)
                labels = event.get("labels", {})
                if not isinstance(labels, Mapping):
                    labels = {}
                valid_time = _number(start) and _number(end) and 0 <= start < end <= duration_ms
                rms = _channel_rms(audio_path, start, end, pcm_cache=pcm_cache) if mapped is not None else None
                active = bool(rms and mapped < len(rms) and rms[mapped] >= 1e-4)
                segments.append({
                    "segment_id": f"track-{track_index}-event-{event_index}",
                    "text": text,
                    "start_ms": start,
                    "end_ms": end,
                    "role": None,
                    "task_kind": "asr",
                    "channel_index": mapped,
                    "source_track_index": track_index,
                    "source_speaker": event.get("speaker"),
                    "source_role": None,
                    "language": event.get("LID"),
                    "annotation_state": event.get("state"),
                    "native_turn_state": event.get("state"),
                    "native_paralinguistic": labels.get("paralinguistic"),
                    "native_labels": {
                        str(name): value for name, value in labels.items()
                        if str(name).strip().lower() not in {"txt", "text", "transcript"}
                    },
                    "audio_channel_verified": mapped is not None and active,
                    "time_mask": bool(valid_time),
                    "text_mask": bool(valid_time and text.strip() and mapped is not None and active),
                    "action_mask": False,
                })
        warnings = []
        if not matches:
            warnings.append("pairing_missing")
        elif len(matches) > 1:
            warnings.append("pairing_ambiguous")
        if pairing and not annotation_path:
            warnings.append("annotation_unresolved")
        if collision:
            warnings.append("annotation_basename_collision")
        group_id = _merge_group_id("DuplexConv", entry, str(key))
        result.append(_canonical_raw({
            "conversation_id": str(entry.get("source_key") or entry.get("source_key_sha256") or f"duplexconv-{row_index:08d}"),
            "source_group_id": group_id,
            "source": "DuplexConv",
            "source_manifest_path": str(manifest_path),
            "annotation_path": str(annotation_path) if annotation_path else None,
            "pairing_review_path": str(pairing_manifest) if pairing_manifest else None,
            "split": _split(entry.get("project_split", entry.get("split"))),
            "duration_ms": duration_ms,
            "sample_rate_hz": sample_rate,
            "channels": channel_count,
            "channel_map": {},
            "audio_path": str(audio_path) if audio_path else None,
            "audio_sha256": entry.get("sha256", entry.get("audio_sha256")),
            "source_key": entry.get("source_key"),
            "source_key_sha256": entry.get("source_key_sha256"),
            "language": "Chinese",
            "preparation_warnings": warnings,
            "segments": segments,
            "frames": [],
        }, base_dir=manifest_path.parent))
    return result


def select_real_conversational_source_keys(
    *,
    smoothconv_root: str | Path | None = None,
    smoothconv_evidence: str | Path | None = None,
    duplexconv_root: str | Path | None = None,
    duplexconv_pairing: str | Path | None = None,
    duplexconv_annotation_roots: Sequence[str | Path] | None = None,
    max_conversations: int = 128,
    shortlist_multiplier: int = 2,
) -> dict[str, Any]:
    """Rank annotation-only candidates before RMS reads and return bounded source-key sets.

    This preselection reads manifests/evidence/annotation JSON only. It never
    opens WAVs; the regular readers subsequently verify the selected channels.
    A source_group_id is selected or skipped as a unit, including cross-source
    rows when the manifests carry the same group identity.
    """
    if isinstance(max_conversations, bool) or not isinstance(max_conversations, int) or max_conversations <= 0:
        raise ValueError("max_conversations deve essere positivo")
    if isinstance(shortlist_multiplier, bool) or not isinstance(shortlist_multiplier, int) or shortlist_multiplier < 1:
        raise ValueError("shortlist_multiplier deve essere almeno 1")
    records_by_source: dict[str, list[dict[str, Any]]] = {"SmoothConv": [], "DuplexConv": []}

    if smoothconv_root is not None:
        manifest_path = _find_jsonl(smoothconv_root, ("smoothconv.slice.jsonl",))
        root = Path(smoothconv_root).resolve() if Path(smoothconv_root).is_dir() else manifest_path.parent
        evidence_path = Path(smoothconv_evidence).expanduser().resolve() if smoothconv_evidence else _smooth_evidence_path(root)
        evidence_rows = _rows(evidence_path) if evidence_path and evidence_path.is_file() else []
        evidence_index = _index_rows(evidence_rows, "source_key", "source_key_sha256", "source_audio_sha256")
        for row_index, entry in enumerate(_read_jsonl(manifest_path)):
            key = str(entry.get("source_key") or entry.get("source_key_sha256") or row_index)
            matches = _match_row(entry, evidence_index, ("source_key", "source_key_sha256"))
            complete_by_channel = {0: 0, 1: 0}
            state_counts: set[str] = set()
            backchannels = 0
            text_events = 0
            valid_events = 0
            for match in matches:
                for event in match.get("source_events", ()) or ():
                    if not isinstance(event, Mapping):
                        continue
                    annotation = event.get("source_annotation_event") or {}
                    if not isinstance(annotation, Mapping):
                        annotation = {}
                    attrs = annotation.get("attributes", {})
                    if not isinstance(attrs, Mapping):
                        attrs = {}
                    raw_channel = event.get("source_channel_index", annotation.get("channelIndex", annotation.get("channel_index")))
                    try:
                        channel = int(raw_channel)
                    except (TypeError, ValueError):
                        continue
                    start = event.get("source_start_ms", event.get("start_ms", annotation.get("start")))
                    end = event.get("source_end_ms", event.get("end_ms", annotation.get("end")))
                    if channel not in {0, 1} or not (_number(start) and _number(end) and 0 <= start < end):
                        continue
                    valid_events += 1
                    text = annotation.get("text", annotation.get("transcript", annotation.get("txt", "")))
                    if isinstance(text, str) and text.strip():
                        text_events += 1
                    state = _native_state(attrs.get("turn"))
                    if state:
                        state_counts.add(state)
                    if state == "complete" and isinstance(text, str) and text.strip():
                        complete_by_channel[channel] += 1
                    if state == "backchannel":
                        backchannels += 1
            if not valid_events:
                continue
            quality = (
                int(all(complete_by_channel[channel] > 0 for channel in (0, 1))),
                len(state_counts), min(backchannels, 16), min(sum(complete_by_channel.values()), 32),
                min(text_events, 64), min(valid_events, 64),
            )
            group_id = _merge_group_id("SmoothConv", entry, key)
            records_by_source["SmoothConv"].append({
                "key": key, "group_id": group_id, "split": _split(entry.get("project_split", entry.get("split"))),
                "quality": quality,
            })

    if duplexconv_root is not None:
        manifest_path = _find_jsonl(duplexconv_root, ("duplexconv.slice.jsonl",))
        root = Path(duplexconv_root).resolve() if Path(duplexconv_root).is_dir() else manifest_path.parent
        if duplexconv_pairing is not None:
            pairing_manifest = Path(duplexconv_pairing).expanduser().resolve()
        else:
            candidates = [
                root / "annotation_pairing_review/pairing_review.jsonl",
                root / "pairing_review.jsonl",
                root / "data/hf_slices/annotation_pairing_review/pairing_review.jsonl",
            ]
            pairing_manifest = next((path.resolve() for path in candidates if path.is_file()), None)
            if pairing_manifest is None:
                hf_slices = root / "data" / "hf_slices"
                if not hf_slices.is_dir() and root.name == "hf_slices":
                    hf_slices = root
                discovered = sorted(
                    candidate.resolve()
                    for candidate in hf_slices.glob("duplexconv*/annotation_pairing_review/pairing_review.jsonl")
                    if candidate.is_file()
                ) if hf_slices.is_dir() else []
                if len(discovered) == 1:
                    pairing_manifest = discovered[0]
                elif len(discovered) > 1:
                    raise FileNotFoundError(f"piu pairing_review.jsonl trovati sotto {hf_slices}; passare --duplexconv-pairing")
        pairing_rows = _rows(pairing_manifest) if pairing_manifest else []
        pairing_index = _index_rows(pairing_rows, "source_key", "source_key_sha256", "audio_sha256")
        bases = _path_bases(manifest_path, root)
        for row_index, entry in enumerate(_read_jsonl(manifest_path)):
            key = str(entry.get("source_key") or entry.get("source_key_sha256") or row_index)
            matches = _match_row(entry, pairing_index, ("source_key", "source_key_sha256"))
            if len(matches) != 1:
                continue
            annotation_path, _collision = _resolve_annotation_path(
                entry, matches[0], root=root, bases=bases, annotation_roots=duplexconv_annotation_roots,
            )
            if annotation_path is None:
                continue
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(annotation, Mapping):
                continue
            asr = annotation.get("asr", ()) or ()
            complete_by_track = {0: 0, 1: 0}
            state_counts: set[str] = set()
            backchannels = 0
            text_events = 0
            valid_events = 0
            for track, events in enumerate(asr if isinstance(asr, list) else ()):
                if track not in {0, 1} or not isinstance(events, list):
                    continue
                for event in events:
                    if not isinstance(event, Mapping):
                        continue
                    start, end = event.get("startInMs"), event.get("endInMs")
                    if not (_number(start) and _number(end) and 0 <= start < end):
                        continue
                    valid_events += 1
                    text = _annotation_text(event)
                    if text.strip():
                        text_events += 1
                    state = _native_state(event.get("state"))
                    if state:
                        state_counts.add(state)
                    if state == "complete" and text.strip():
                        complete_by_track[track] += 1
                    if state == "backchannel":
                        backchannels += 1
            if not valid_events:
                continue
            quality = (
                int(all(complete_by_track[track] > 0 for track in (0, 1))),
                len(state_counts), min(backchannels, 16), min(sum(complete_by_track.values()), 32),
                min(text_events, 64), min(valid_events, 64),
            )
            group_id = _merge_group_id("DuplexConv", entry, key)
            records_by_source["DuplexConv"].append({
                "key": key, "group_id": group_id, "split": _split(entry.get("project_split", entry.get("split"))),
                "quality": quality,
            })

    all_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for source, source_rows in records_by_source.items():
        for record in source_rows:
            all_groups.setdefault(record["group_id"], []).append((source, record))

    # Interleave source/split buckets so a high-count source or split cannot
    # consume the annotation shortlist before the others are considered.
    bucket_rows: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for source, source_rows in records_by_source.items():
        for record in source_rows:
            bucket = (source, str(record["split"] or "unknown"))
            bucket_rows.setdefault(bucket, []).append((source, record))
    for rows in bucket_rows.values():
        rows.sort(key=lambda item: tuple(-value for value in item[1]["quality"]) + (item[1]["key"],))

    budget = max_conversations * shortlist_multiplier
    selected_groups: set[str] = set()
    selected_count = 0
    cursors = {bucket: 0 for bucket in bucket_rows}
    bucket_order = sorted(bucket_rows)
    while bucket_order and selected_count < budget:
        added = False
        for bucket in bucket_order:
            rows = bucket_rows[bucket]
            while cursors[bucket] < len(rows) and rows[cursors[bucket]][1]["group_id"] in selected_groups:
                cursors[bucket] += 1
            if cursors[bucket] >= len(rows):
                continue
            group_id = rows[cursors[bucket]][1]["group_id"]
            group_rows = all_groups[group_id]
            cursors[bucket] += 1
            if selected_count + len(group_rows) > budget:
                continue
            selected_groups.add(group_id)
            selected_count += len(group_rows)
            added = True
            if selected_count >= budget:
                break
        if not added:
            break

    selected_keys = {source: set() for source in records_by_source}
    selected_by_source = {source: 0 for source in records_by_source}
    for group_id in selected_groups:
        for source, record in all_groups[group_id]:
            selected_keys[source].add(record["key"])
            selected_by_source[source] += 1
    return {
        "source_keys": selected_keys,
        "annotation_candidate_conversations_by_source": {source: len(rows) for source, rows in records_by_source.items()},
        "selected_annotation_conversations_by_source": selected_by_source,
        "selected_annotation_conversations": selected_count,
        "selected_source_groups": len(selected_groups),
        "rms_reads_bounded_by_selected_source_keys": True,
        "shortlist_limit": budget,
    }


def summarize_preparation_counts(conversations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize useable transcript/audio/action labels without loading media."""
    by_source: dict[str, dict[str, int]] = {}
    for conversation in conversations:
        source = str(conversation.get("source", "unknown"))
        counts = by_source.setdefault(source, {
            "conversations": 0,
            "asr_full_text": 0,
            "asr_cropped_text": 0,
            "asr_out_of_range": 0,
            "channel_ambiguous": 0,
            "annotation_unresolved": 0,
            "audio_missing": 0,
            "action_supervision": 0,
        })
        counts["conversations"] += 1
        if not conversation.get("audio_path"):
            counts["audio_missing"] += 1
        warnings = set(conversation.get("preparation_warnings", ()))
        if warnings.intersection({"annotation_unresolved", "evidence_missing", "pairing_missing", "pairing_ambiguous"}):
            counts["annotation_unresolved"] += 1
        for segment in conversation.get("segments", ()) or ():
            if segment.get("task_kind") != "asr":
                if segment.get("text_mask") and segment.get("time_mask"):
                    counts["asr_full_text"] += 1
                elif segment.get("time_mask") and segment.get("text"):
                    counts["asr_cropped_text"] += 1
            else:
                if not segment.get("time_mask"):
                    counts["asr_out_of_range"] += 1
                elif segment.get("text_mask") and segment.get("audio_channel_verified"):
                    counts["asr_full_text"] += 1
                elif (
                    segment.get("text")
                    and _number(segment.get("measured_duration_s"))
                    and _number(segment.get("start_ms"))
                    and _number(segment.get("end_ms"))
                    and segment["end_ms"] - segment["start_ms"] < segment["measured_duration_s"] * 1000 - 80
                ):
                    counts["asr_cropped_text"] += 1
                if segment.get("channel_index") is None or not segment.get("audio_channel_verified"):
                    counts["channel_ambiguous"] += 1
        counts["action_supervision"] += sum(
            1 for frame in conversation.get("frames", ()) or () if frame.get("action_mask")
        )
    totals: dict[str, int] = {}
    for counts in by_source.values():
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    return {"totals": totals, "by_source": by_source}


__all__ = [
    "read_annotated_jsonl",
    "read_synthetic_v3_manifest",
    "read_synthetic_v3_clean_asr",
    "read_smoothconv",
    "read_duplexconv",
    "select_real_conversational_source_keys",
    "build_real_conversational_views",
    "summarize_preparation_counts",
]
