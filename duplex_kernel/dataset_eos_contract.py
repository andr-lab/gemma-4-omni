"""Versioned segment-aware dataset/EOS contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DATASET_EOS_CONTRACT_SCHEMA_VERSION = 5
DATASET_EOS_CONTRACT_NAME = "05_dataset_eos_contract"
RUNTIME_CACHE_SCHEMA_VERSION = 3
RUNTIME_CACHE_CONTRACT = "p4d_r_semantic_aligned_segment_eos_v3"
DEFAULT_WINDOW_FRAMES = 64
TERMINAL_QUERY_BUDGET = 1
PAD_TARGET_ID = -100


def _as_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class UtteranceRecord:
    """Authoritative source annotation for one target utterance."""

    utterance_id: str
    speaker_id: str
    frame_start: int
    frame_end: int
    text: str
    text_token_ids: tuple[int, ...] = ()
    mel_provenance: Mapping[str, Any] = field(default_factory=dict)
    rvq_provenance: Mapping[str, Any] = field(default_factory=dict)
    voice_conditioning_signature: str | None = None

    def __post_init__(self) -> None:
        start, end = int(self.frame_start), int(self.frame_end)
        if start < 0 or end <= start:
            raise ValueError("utterance frame range deve essere non vuoto e non negativo")
        if not str(self.utterance_id).strip() or not str(self.speaker_id).strip():
            raise ValueError("utterance_id e speaker_id devono essere non vuoti")
        ids = tuple(int(token) for token in self.text_token_ids)
        if any(token < 0 for token in ids):
            raise ValueError("text_token_ids contiene valori negativi")
        object.__setattr__(self, "utterance_id", str(self.utterance_id))
        object.__setattr__(self, "speaker_id", str(self.speaker_id))
        object.__setattr__(self, "frame_start", start)
        object.__setattr__(self, "frame_end", end)
        object.__setattr__(self, "text", _normalise_text(self.text))
        object.__setattr__(self, "text_token_ids", ids)
        object.__setattr__(self, "mel_provenance", _as_json_value(dict(self.mel_provenance)))
        object.__setattr__(self, "rvq_provenance", _as_json_value(dict(self.rvq_provenance)))
        if self.voice_conditioning_signature is not None:
            object.__setattr__(self, "voice_conditioning_signature", str(self.voice_conditioning_signature))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UtteranceRecord":
        return cls(
            utterance_id=str(value["utterance_id"]),
            speaker_id=str(value.get("speaker_id", "unknown")),
            frame_start=int(value["frame_start"]),
            frame_end=int(value["frame_end"]),
            text=str(value.get("text", value.get("effective_text", ""))),
            text_token_ids=tuple(int(token) for token in value.get("text_token_ids", ())),
            mel_provenance=value.get("mel_provenance", {}),
            rvq_provenance=value.get("rvq_provenance", {}),
            voice_conditioning_signature=value.get("voice_conditioning_signature"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "text": self.text,
            "text_token_ids": list(self.text_token_ids),
            "mel_provenance": _as_json_value(self.mel_provenance),
            "rvq_provenance": _as_json_value(self.rvq_provenance),
            "voice_conditioning_signature": self.voice_conditioning_signature,
        }


@dataclass(frozen=True)
class SegmentWindowRecord:
    utterance_id: str
    speaker_id: str
    frame_start: int
    frame_end: int
    valid_frame_mask: tuple[bool, ...]
    is_utterance_end: bool
    effective_text: str
    text_token_ids: tuple[int, ...]
    input_text_token_ids: tuple[int, ...]
    text_truncated: bool
    semantic_text_valid: bool
    mel_provenance: Mapping[str, Any]
    rvq_provenance: Mapping[str, Any]
    voice_conditioning_signature: str | None
    window_index: int
    source_frame_start: int
    source_frame_end: int

    @property
    def valid_frame_count(self) -> int:
        return sum(self.valid_frame_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "valid_frame_mask": list(self.valid_frame_mask),
            "is_utterance_end": bool(self.is_utterance_end),
            "effective_text": self.effective_text,
            "text_token_ids": list(self.text_token_ids),
            "input_text_token_ids": list(self.input_text_token_ids),
            "text_truncated": bool(self.text_truncated),
            "semantic_text_valid": bool(self.semantic_text_valid),
            "mel_provenance": _as_json_value(self.mel_provenance),
            "rvq_provenance": _as_json_value(self.rvq_provenance),
            "voice_conditioning_signature": self.voice_conditioning_signature,
            "window_index": self.window_index,
            "source_frame_start": self.source_frame_start,
            "source_frame_end": self.source_frame_end,
        }


def coerce_utterances(values: Iterable[UtteranceRecord | Mapping[str, Any]]) -> list[UtteranceRecord]:
    records = [value if isinstance(value, UtteranceRecord) else UtteranceRecord.from_mapping(value) for value in values]
    records.sort(key=lambda item: (item.frame_start, item.frame_end, item.utterance_id))
    previous_end = -1
    for record in records:
        if record.frame_start < previous_end:
            raise ValueError("utterance frame ranges si sovrappongono")
        previous_end = record.frame_end
    return records


def build_segment_windows(
    utterances: Sequence[UtteranceRecord | Mapping[str, Any]],
    *,
    total_frames: int,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    text_block_len: int = 48,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> list[SegmentWindowRecord]:
    total_frames, window_frames, text_block_len = int(total_frames), int(window_frames), int(text_block_len)
    frame_start = int(frame_start)
    frame_end = total_frames if frame_end is None else int(frame_end)
    if total_frames <= 0 or window_frames <= 0 or text_block_len <= 0:
        raise ValueError("total_frames, window_frames e text_block_len devono essere positivi")
    if not 0 <= frame_start < frame_end <= total_frames:
        raise ValueError("frame range dataset non valido")
    max_text_tokens = max(0, text_block_len - 1)  # reserve one slot for BOA/terminal shift
    records = coerce_utterances(utterances)
    if not records:
        records = [UtteranceRecord("unknown-utterance-0", "unknown", 0, total_frames, "")]
    selected: list[UtteranceRecord] = []
    for record in records:
        if record.frame_start < frame_end and record.frame_end > frame_start:
            if record.frame_start < frame_start or record.frame_end > frame_end:
                raise ValueError("split taglia un utterance; usare split per utterance/speaker, non per offset")
            selected.append(record)
    if not selected:
        raise ValueError("nessuna utterance nel range dataset")

    windows: list[SegmentWindowRecord] = []
    window_index = 0
    for utterance in selected:
        ids = tuple(utterance.text_token_ids)
        truncated = len(ids) > max_text_tokens
        input_ids = () if truncated else ids
        semantic_valid = bool(utterance.text) and bool(ids) and not truncated
        for start in range(utterance.frame_start, utterance.frame_end, window_frames):
            end = min(start + window_frames, utterance.frame_end)
            valid_count = end - start
            windows.append(SegmentWindowRecord(
                utterance_id=utterance.utterance_id,
                speaker_id=utterance.speaker_id,
                frame_start=start - frame_start,
                frame_end=end - frame_start,
                valid_frame_mask=tuple([True] * valid_count + [False] * (window_frames - valid_count)),
                is_utterance_end=end == utterance.frame_end,
                effective_text=utterance.text,
                text_token_ids=ids,
                input_text_token_ids=input_ids,
                text_truncated=truncated,
                semantic_text_valid=semantic_valid,
                mel_provenance=utterance.mel_provenance,
                rvq_provenance=utterance.rvq_provenance,
                voice_conditioning_signature=utterance.voice_conditioning_signature,
                window_index=window_index,
                source_frame_start=start,
                source_frame_end=end,
            ))
            window_index += 1
    return windows


def validate_window_records(records: Sequence[SegmentWindowRecord], window_frames: int) -> None:
    expected = int(window_frames)
    if expected <= 0:
        raise ValueError("window_frames deve essere positivo")
    by_utterance: dict[str, list[SegmentWindowRecord]] = {}
    for record in records:
        if len(record.valid_frame_mask) != expected:
            raise ValueError("valid_frame_mask non allineata alla capacità finestra")
        if record.frame_end <= record.frame_start or record.valid_frame_count != record.frame_end - record.frame_start:
            raise ValueError("valid_frame_mask non coincide con frame_start/end")
        if any(not record.valid_frame_mask[index] for index in range(record.valid_frame_count)):
            raise ValueError("valid_frame_mask deve avere frame validi contigui in testa")
        if any(record.valid_frame_mask[index] for index in range(record.valid_frame_count, expected)):
            raise ValueError("valid_frame_mask deve avere padding solo in coda")
        by_utterance.setdefault(record.utterance_id, []).append(record)
    ordered = sorted(records, key=lambda item: (item.frame_start, item.frame_end, item.utterance_id))
    for previous, current in zip(ordered, ordered[1:]):
        if current.frame_start < previous.frame_end:
            raise ValueError("finestre segment-aware sovrapposte tra utterance")

    for utterance_id, group in by_utterance.items():
        group.sort(key=lambda item: item.frame_start)
        for previous, current in zip(group, group[1:]):
            if previous.frame_end != current.frame_start:
                raise ValueError(f"gap/overlap nelle finestre utterance={utterance_id}")
            if previous.is_utterance_end:
                raise ValueError(f"utterance={utterance_id} termina prima dell'ultima finestra")
        if not group[-1].is_utterance_end:
            raise ValueError(f"utterance={utterance_id} senza vera fine utterance")


def split_utterances_by_speaker(
    utterances: Sequence[UtteranceRecord | Mapping[str, Any]], *, validation_fraction: float
) -> tuple[list[UtteranceRecord], list[UtteranceRecord]]:
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction deve essere tra 0 e 1")
    records = coerce_utterances(utterances)
    groups: dict[str, list[UtteranceRecord]] = {}
    for record in records:
        groups.setdefault(record.speaker_id, []).append(record)
    target = max(1, int(round(len(records) * fraction))) if records else 0
    validation: list[UtteranceRecord] = []
    for speaker_id in reversed(list(groups)):
        if validation and len(validation) >= target:
            break
        validation[0:0] = groups[speaker_id]
    validation_ids = {id(record) for record in validation}
    train = [record for record in records if id(record) not in validation_ids]
    if not train and validation:
        return records, []
    return train, validation


def split_utterances_by_time(
    utterances: Sequence[UtteranceRecord | Mapping[str, Any]], *, validation_fraction: float, by_speaker: bool = False
) -> tuple[list[UtteranceRecord], list[UtteranceRecord]]:
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction deve essere tra 0 e 1")
    records = coerce_utterances(utterances)
    if not records:
        return [], []
    if by_speaker:
        return split_utterances_by_speaker(records, validation_fraction=fraction)
    target = max(1, int(round(sum(r.frame_end - r.frame_start for r in records) * fraction)))
    validation: list[UtteranceRecord] = []
    covered = 0
    for record in reversed(records):
        validation.insert(0, record)
        covered += record.frame_end - record.frame_start
        if covered >= target:
            break
    ids = {id(record) for record in validation}
    train = [record for record in records if id(record) not in ids]
    return (records, []) if not train else (train, validation)


def voice_conditioning_signature(voice_conditioning: Mapping[str, Any] | None) -> str | None:
    if voice_conditioning is None:
        return None
    if voice_conditioning.get("signature") is not None:
        return str(voice_conditioning["signature"])
    reference = voice_conditioning.get("reference_wav_sha256")
    embedding = voice_conditioning.get("speaker_embedding_sha256")
    return None if reference is None or embedding is None else f"xvector:{reference}:{embedding}"


def assert_voice_conditioning_compatible(record: SegmentWindowRecord | Mapping[str, Any], voice_conditioning: Mapping[str, Any] | None) -> None:
    is_window = isinstance(record, SegmentWindowRecord)
    expected = record.voice_conditioning_signature if is_window else record.get("voice_conditioning_signature")
    speaker_id = record.speaker_id if is_window else str(record.get("speaker_id", "unknown"))
    actual = voice_conditioning_signature(voice_conditioning)
    if expected is not None and actual != expected:
        raise ValueError(f"speaker conditioning incompatibile con il target: expected={expected!r} actual={actual!r}")
    if speaker_id != "unknown" and actual is None:
        raise ValueError("speaker conditioning mancante per target multispeaker")


def contract_document() -> dict[str, Any]:
    """Return the complete normative contract embedded in every v5 sidecar."""
    return {
        "schema_version": DATASET_EOS_CONTRACT_SCHEMA_VERSION,
        "contract": DATASET_EOS_CONTRACT_NAME,
        "status": "normative",
        "timeline": {"rvq_hz": 12.5, "mel_hz": 100, "rvq_frame_ms": 80, "frame_end": "exclusive", "mel_per_rvq_frame": 8},
        "record_fields": {
            "utterance_id": "non-empty stable source id", "speaker_id": "non-empty source speaker id",
            "frame_start": "inclusive source RVQ frame", "frame_end": "exclusive source RVQ frame",
            "valid_frame_mask": "bool[window_capacity], contiguous true prefix then false padding",
            "is_utterance_end": "true only for the final window of the annotated utterance",
            "effective_text": "complete utterance text, never a window prefix", "text_token_ids": "complete tokenizer output",
            "input_text_token_ids": "complete output only when it fits text capacity minus one target-shift slot; otherwise empty",
            "text_truncated": "true when complete token output exceeds text capacity", "semantic_text_valid": "false for missing or truncated text",
            "mel_provenance": "source/frontend/config/hash metadata", "rvq_provenance": "source/tokenizer/codebook/config/hash metadata",
            "voice_conditioning_signature": "content-addressed speaker conditioning identity",
        },
        "window": {"prefix_capacity_frames": 64, "padding": "valid_frame_mask_only", "window_boundary_is_utterance_boundary": False, "padding_target_id": PAD_TARGET_ID, "padding_never_contributes_to_loss": True, "mixed_utterance_window": "forbidden; split at the annotated boundary"},
        "eos": {"terminal_query_budget": TERMINAL_QUERY_BUDGET, "terminal_query_position": "after the last valid acoustic query", "terminal_query_does_not_consume_acoustic_frame_budget": True, "eos_label_on_intermediate_window": "not_applicable/ignored", "eos_label_on_true_utterance_end": "required", "eos_label_on_budget_exhaustion_without_eos": "not_completed; continuation or explicit budget stop"},
        "text_target_policy": {"terminal_shift_reserve": "one text-block position reserved so the first token and BOA remain reachable", "prefix_only_text_for_rvq": "forbidden", "long_text": "retain complete text and mark semantic_text_valid=false unless an explicit continuation unit is created", "text_loss_for_invalid_text": "ignored", "audio_target_for_invalid_text": "excluded from semantic cache extraction"},
        "speaker_policy": {"fixed_reference_for_multispeaker": "forbidden", "missing_or_mismatched_conditioning": "fail_closed", "signature_algorithm": "canonical signature supplied by TalkerModel.voice_conditioning_signature"},
        "split_policy": {"primary_unit": "utterance", "optional_group": "speaker", "offset_split_through_utterance": "forbidden", "split_manifest_required": True},
        "provenance": ["mel_provenance", "rvq_provenance"],
        "runtime_budget": {"prefix_capacity": 64, "acoustic_frame_budget": 64, "terminal_query_budget": TERMINAL_QUERY_BUDGET, "longer_utterance": "explicit fenced continuation/rollover; do not enlarge cache silently", "causality": "prefix visibility remains bounded by lookahead and available frame", "fencing": "generation_id + segment_id + contiguous local frame ids", "native_backend_terminal_query": "required; backend fails closed until audio-only ABI is available"},
        "cache_policy": {"runtime_cache_schema_version": RUNTIME_CACHE_SCHEMA_VERSION, "runtime_cache_contract": RUNTIME_CACHE_CONTRACT, "segment_sidecar_content_manifest": "required_sha256_and_byte_size_for_each_tensor_file", "historical_cache_overwrite": False, "new_default_name": "cache_ars_runtime_v3_segment_eos.pt"},
        "required_tests": ["intermediate window", "true utterance end", "two contiguous utterances", "padding", "utterance lengths 1/63/64", "long text", "budget exhausted without EOS", "segment rollover beyond 64 frames", "speaker conditioning mismatch", "split by utterance/speaker", "smoke extraction on a few samples"],
        "continuation": {"max_acoustic_frames_per_segment": 64, "rollover_required_for_longer_segments": True, "terminal_query_does_not_consume_acoustic_budget": True},
    }


__all__ = [
    "DATASET_EOS_CONTRACT_NAME", "DATASET_EOS_CONTRACT_SCHEMA_VERSION", "DEFAULT_WINDOW_FRAMES",
    "RUNTIME_CACHE_CONTRACT", "RUNTIME_CACHE_SCHEMA_VERSION", "TERMINAL_QUERY_BUDGET", "PAD_TARGET_ID",
    "SegmentWindowRecord", "UtteranceRecord", "assert_voice_conditioning_compatible", "build_segment_windows",
    "contract_document", "split_utterances_by_speaker", "split_utterances_by_time", "validate_window_records",
    "voice_conditioning_signature",
]
