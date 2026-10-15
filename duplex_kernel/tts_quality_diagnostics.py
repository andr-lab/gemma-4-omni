"""Bounded, opt-in local artifacts for TTS quality diagnosis.

The helper intentionally has no model/runtime imports. It writes numeric
conditioning/code arrays to a local NPZ and leaves only shapes in JSON.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SYNTHETIC_DIALOGUE_FIXTURE_ID = "synthetic_dinner_dialogue_v1"
E2E_PAIR_DEBUG_FIXTURE_ID = "e2e_pair_debug_v1"
DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET = 96
DIAGNOSTIC_ARRAY_KEYS = frozenset({
    "audio_context_hidden",
    "semantic_anchor_hidden",
    "say_hidden",
    "v9_prefix",
    "rvq_codes",
    "preclamp_pcm",
    "streamed_pcm",
    "final_pcm",
})


def apply_diagnostic_frame_budget(
    text_derived_frames: int,
    *,
    diagnostic_max_frames: int | None,
    absolute_max_frames: int,
) -> int:
    """Extend only an explicitly captured diagnostic response, within a cap."""
    for name, value in (
        ("text_derived_frames", text_derived_frames),
        ("absolute_max_frames", absolute_max_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if absolute_max_frames <= 0:
        raise ValueError("absolute_max_frames must be positive")
    if diagnostic_max_frames is None:
        return min(text_derived_frames, absolute_max_frames)
    if (
        isinstance(diagnostic_max_frames, bool)
        or not isinstance(diagnostic_max_frames, int)
        or diagnostic_max_frames <= 0
        or diagnostic_max_frames > absolute_max_frames
    ):
        raise ValueError("diagnostic_max_frames must be positive and within the model limit")
    return min(diagnostic_max_frames, absolute_max_frames)


class TTSQualityDiagnosticCapture:
    """Allow a bounded number of explicitly marked diagnostic responses."""

    def __init__(
        self,
        *,
        output_dir: str | Path = ".tmp/tts_quality_diagnostics",
        enabled: bool | None = None,
        max_records: int = 2,
    ) -> None:
        if max_records < 1:
            raise ValueError("max_records deve essere positivo")
        self.output_dir = Path(output_dir)
        self.enabled = (
            os.environ.get("DUPLEX_TTS_QUALITY_DIAGNOSTICS") == "1"
            if enabled is None else bool(enabled)
        )
        self.max_records = int(max_records)
        self._lock = threading.Lock()
        self._claimed: set[int] = set()
        self._next_slot = 0

    def claim(self, fixture_id: Any) -> int | None:
        if not self.enabled or fixture_id not in {
            SYNTHETIC_DIALOGUE_FIXTURE_ID,
            E2E_PAIR_DEBUG_FIXTURE_ID,
        }:
            return None
        with self._lock:
            if self._next_slot >= self.max_records:
                return None
            slot = self._next_slot
            self._next_slot += 1
            self._claimed.add(slot)
            return slot

    def write(self, slot: int, record: Mapping[str, Any]) -> dict[str, Path]:
        if slot not in self._claimed:
            raise ValueError("slot diagnostico non acquisito")
        raw_arrays = {key: record.get(key) for key in DIAGNOSTIC_ARRAY_KEYS}
        arrays: dict[str, np.ndarray] = {}
        shapes: dict[str, dict[str, Any]] = {}
        for key, value in raw_arrays.items():
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.dtype.kind not in "fiu" or arr.dtype.hasobject:
                raise ValueError(f"array diagnostico non numerico: {key}")
            if key != "preclamp_pcm" and arr.dtype.kind == "f" and not np.isfinite(arr).all():
                raise ValueError(f"array diagnostico non finito: {key}")
            dtype = np.float32 if key != "rvq_codes" else np.int32
            arr = np.ascontiguousarray(arr, dtype=dtype)
            arrays[key] = arr
            shapes[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if not arrays:
            raise ValueError("nessun array diagnostico presente")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        record_metadata = dict(record.get("metadata") or {})
        fixture_id = str(record_metadata.get("fixture_id") or SYNTHETIC_DIALOGUE_FIXTURE_ID)
        if fixture_id == E2E_PAIR_DEBUG_FIXTURE_ID:
            stem = f"e2e_pair_tts_{slot + 1:02d}_{uuid.uuid4().hex[:10]}"
            schema = "e2e_pair_tts_quality_diagnostic_v1"
        else:
            stem = f"synthetic_tts_{slot + 1:02d}_{uuid.uuid4().hex[:10]}"
            schema = "synthetic_tts_quality_diagnostic_v1"
        arrays_path = self.output_dir / f"{stem}.npz"
        metadata_path = self.output_dir / f"{stem}.json"
        arrays_tmp = arrays_path.with_suffix(".npz.tmp")
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        with arrays_tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        metadata = {
            "schema": schema,
            "fixture_id": fixture_id,
            "slot": slot,
            "arrays_file": arrays_path.name,
            "arrays": shapes,
            "metadata": record_metadata,
        }
        with metadata_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(arrays_tmp, arrays_path)
        os.replace(metadata_tmp, metadata_path)
        return {"arrays": arrays_path, "metadata": metadata_path}


__all__ = [
    "DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET",
    "DIAGNOSTIC_ARRAY_KEYS",
    "SYNTHETIC_DIALOGUE_FIXTURE_ID",
    "TTSQualityDiagnosticCapture",
    "apply_diagnostic_frame_budget",
]
