"""Functional contract for synthetic frame-target audio used by listener Stage2.

PCM metadata is read with soundfile.info, whose documented result exposes
samplerate, channels, and frame count:
https://python-soundfile.readthedocs.io/en/0.13.1/#soundfile.info
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .schema import ACTIONS


SAMPLE_RATE_HZ = 16_000
TICK_MS = 80
FRAME_SAMPLES = SAMPLE_RATE_HZ * TICK_MS // 1000


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_audio_grounded_v5_record(
    source: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate a rendered stereo record before Stage2 ingestion.

    The contract is deliberately functional: it checks readable PCM, sample
    timeline, action masks, utterance links, and split/group provenance. It
    does not require a particular generator source tag or profile version.
    """
    import soundfile as sf

    row = dict(source)
    conversation_id = str(row.get("conversation_id") or "").strip()
    split = str(row.get("split") or "").strip().lower()
    if not conversation_id:
        raise ValueError("audio-grounded record requires conversation_id")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"{conversation_id}: split must be train/validation/test")
    raw_path = row.get("audio_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{conversation_id}: audio_path is required")
    audio_path = Path(raw_path).expanduser()
    if not audio_path.is_absolute() and base_dir is not None:
        audio_path = Path(base_dir) / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"{conversation_id}: rendered PCM is missing: {audio_path}")
    info = sf.info(str(audio_path))
    if int(info.samplerate) != SAMPLE_RATE_HZ or int(info.channels) != 2:
        raise ValueError(
            f"{conversation_id}: expected stereo 16 kHz PCM, got "
            f"{info.channels}ch/{info.samplerate}Hz"
        )
    channel_map = row.get("channel_map") or {}
    if str(channel_map.get("0", channel_map.get(0, ""))).lower() != "user" or str(
        channel_map.get("1", channel_map.get(1, ""))
    ).lower() != "assistant":
        raise ValueError(f"{conversation_id}: channel_map must map user=0 and assistant=1")
    for field in ("sample_rate", "sample_rate_hz"):
        if field in row and (not _integer(row[field]) or row[field] != SAMPLE_RATE_HZ):
            raise ValueError(f"{conversation_id}: {field} must be 16000")
    if "channels" in row and (not _integer(row["channels"]) or row["channels"] != 2):
        raise ValueError(f"{conversation_id}: channels must be 2")
    if not _integer(row.get("frame_ms", TICK_MS)) or row.get("frame_ms", TICK_MS) != TICK_MS:
        raise ValueError(f"{conversation_id}: frame_ms must be 80")
    audio_samples = int(info.frames)
    if audio_samples <= 0:
        raise ValueError(f"{conversation_id}: audio PCM is empty")
    if row.get("duration_s") is not None:
        duration_s = float(row["duration_s"])
        if not math.isfinite(duration_s) or abs(duration_s - audio_samples / SAMPLE_RATE_HZ) > 1.0 / SAMPLE_RATE_HZ:
            raise ValueError(f"{conversation_id}: duration_s disagrees with rendered PCM")

    targets = row.get("frame_targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{conversation_id}: frame_targets must be a non-empty list")
    input_ids: list[int] = []
    action_counts = {action: 0 for action in ACTIONS}
    for index, frame in enumerate(targets):
        if not isinstance(frame, Mapping):
            raise ValueError(f"{conversation_id}: frame_targets[{index}] must be an object")
        input_id = frame.get("input_frame_id")
        target_id = frame.get("target_frame_id")
        if not _integer(input_id) or input_id < 0 or (input_ids and input_id <= input_ids[-1]):
            raise ValueError(f"{conversation_id}: input_frame_id must be unique and increasing")
        if target_id != input_id + 1:
            raise ValueError(f"{conversation_id}: target_frame_id must equal input_frame_id + 1")
        expected = {
            "input_audio_start_sample": input_id * FRAME_SAMPLES,
            "input_audio_end_sample": (input_id + 1) * FRAME_SAMPLES,
            "target_action_start_sample": (input_id + 1) * FRAME_SAMPLES,
            "target_action_end_sample": (input_id + 2) * FRAME_SAMPLES,
        }
        for field, expected_value in expected.items():
            value = frame.get(field)
            if not _integer(value) or value != expected_value:
                raise ValueError(f"{conversation_id}: {field} is not aligned to the 80 ms causal timeline")
        if frame["input_audio_end_sample"] > audio_samples or frame["target_action_end_sample"] > audio_samples:
            raise ValueError(f"{conversation_id}: frame target extends beyond rendered PCM")
        if frame.get("input_text_visible") is not False or frame.get("future_audio_excluded") is not True:
            raise ValueError(f"{conversation_id}: frame target does not attest a causal audio-only cutoff")
        if frame.get("action") not in ACTIONS or type(frame.get("target_action_mask")) is not bool:
            raise ValueError(f"{conversation_id}: action/mask contract is invalid")
        if frame["target_action_mask"]:
            action_counts[str(frame["action"])] += 1
        input_ids.append(input_id)
    if input_ids != list(range(len(input_ids))):
        raise ValueError(f"{conversation_id}: frame_targets must cover the causal timeline from input frame 0")

    utterances = row.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError(f"{conversation_id}: utterances are required for event provenance")
    utterance_ids = {
        str(item.get("utterance_id"))
        for item in utterances
        if isinstance(item, Mapping) and item.get("utterance_id")
    }
    if not utterance_ids:
        raise ValueError(f"{conversation_id}: utterance_id provenance is missing")
    for index, event in enumerate(row.get("event_schedule", ()) or ()):
        if not isinstance(event, Mapping):
            raise ValueError(f"{conversation_id}: event_schedule[{index}] must be an object")
        links = [
            event.get(key)
            for key in ("trigger_utterance_id", "context_utterance_id", "response_utterance_id")
            if event.get(key) not in (None, "")
        ]
        if not links or any(str(link) not in utterance_ids for link in links):
            raise ValueError(f"{conversation_id}: event_schedule[{index}] has no valid utterance link")

    group_id = str(
        row.get("source_group_id")
        or row.get("topic_group_id")
        or row.get("topic_family_id")
        or conversation_id
    ).strip()
    if not group_id:
        raise ValueError(f"{conversation_id}: source group id is required")
    hasher = hashlib.sha256()
    with audio_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    actual_digest = hasher.hexdigest()
    declared_digest = str(row.get("audio_sha256") or "").lower()
    if declared_digest and declared_digest != actual_digest:
        raise ValueError(f"{conversation_id}: audio_sha256 disagrees with the rendered PCM")
    digest = actual_digest
    row.update(
        {
            "conversation_id": conversation_id,
            "source_group_id": group_id,
            "split": split,
            "audio_path": str(audio_path),
            "audio_sha256": digest,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "duration_ms": audio_samples * 1000.0 / SAMPLE_RATE_HZ,
            "tick_ms": TICK_MS,
            "synthetic": True,
        }
    )
    row["action_counts_by_class"] = action_counts
    return row


__all__ = ["FRAME_SAMPLES", "SAMPLE_RATE_HZ", "TICK_MS", "validate_audio_grounded_v5_record"]
