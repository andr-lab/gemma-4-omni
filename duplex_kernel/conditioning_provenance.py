"""Explicit provenance contract for offline, teacher-forced and live conditioning.

The three paths intentionally produce different representations even when a
consumer happens to see the same tensor shape.  This module provides the
small, model-agnostic record format used by diagnostics; it never compares
representations by shape alone and never claims that a fixture fake certifies a
real Gemma run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

CONDITIONING_PROVENANCE_SCHEMA_VERSION = 1
PRODUCTION_CONTRACT_ID = "live_causal_listener_80ms_separate_text_plan_v1"
CONDITIONING_PATHS = {
    "A": "A.extract_runtime_cache.audio_position_hidden",
    "B": "B.generate_p7_v9_conditioning.assistant_token_hidden",
    "C": "C.server.scheduler_listener.live_pcm_hidden",
}
PROVENANCE_SCHEMA: dict[str, Any] = {
    "schema_version": CONDITIONING_PROVENANCE_SCHEMA_VERSION,
    "timestamp_unit": "nanoseconds_since_monotonic_clock_origin_for_live; logical_frame_end_for_offline",
    "frame_id_semantics": "zero_based_contiguous_listener_or_aligned_acoustic_frame_within_stream",
    "availability_semantics": {
        "available_at_ns": "earliest source envelope time at which this row may be computed",
        "available_through_frame": "latest source frame allowed to influence this row",
        "future_pcm_visible": "must be false for production C rows",
    },
    "anchor_semantics": "object with name, frame_id (nullable), and source",
    "identity_semantics": "sample_id plus stream_id plus frame_id plus representation_id; state_version identifies listener state transition",
}
_ALLOWED_PATHS = frozenset(CONDITIONING_PATHS.values())
_ALLOWED_MODES = frozenset({
    "offline_full_sequence",
    "offline_teacher_forced_text",
    "live_causal_listener",
})

REQUIRED_TRACE_FIELDS = (
    "schema_version",
    "source_path",
    "evidence_role",
    "sample_id",
    "stream_id",
    "frame_id",
    "timestamp_ns",
    "token_type",
    "mode",
    "prompt_visibility",
    "text_visibility",
    "frontend",
    "hidden_layer",
    "hidden_stage",
    "hidden_kind",
    "representation_id",
    "hidden_shape",
    "hidden_dtype",
    "hidden_stats_dtype",
    "hidden_finite",
    "hidden_sha256",
    "hidden_l2_norm",
    "hidden_rms",
    "causal_conv_stage",
    "anchor",
    "availability",
    "source_fingerprint",
    "model_provenance",
    "projector_provenance",
    "input_provenance",
    "state_version",
    "state_lineage",
    "generation_id",
    "segment_id",
)

PRODUCTION_CONTRACT: dict[str, Any] = {
    "contract_id": PRODUCTION_CONTRACT_ID,
    "status": "accepted_for_runtime_contract_audit",
    "listener_period_ms": 80,
    "listener_sample_rate_hz": 16_000,
    "audio_lookahead_ms": 0,
    "listener_hidden": {
        "mode": "live_causal_listener",
        "token_type": "audio_feature_token",
        "prompt_visibility": "not_visible",
        "text_visibility": "separate_plan_not_in_listener_hidden",
        "frontend": "Gemma4AudioFeatureExtractor.incremental_left160_current1280_sentinel1",
        "hidden_layer": "Gemma4Model.language_model.last_hidden_state.first_audio_feature_of_current_frame",
        "hidden_stage": "pre_causal_conv",
        "causal_conv_stage": "after_listener_hidden_in_incremental_v9",
        "anchor": "first_live_listener_hidden_of_current_turn",
        "availability": "capture_end_of_current_80ms_frame",
    },
    "text_plan": {
        "mode": "separate_control_plane",
        "listener_hidden_may_consume_text_tokens": False,
        "text_conditioned_path": "B.generate_p7_v9_conditioning.assistant_token_hidden",
        "text_conditioned_path_is_microphone_causality_evidence": False,
    },
    "invariants": [
        "future PCM is not available to listener frame t",
        "C does not reuse teacher-forced assistant-token hidden as listener hidden",
        "same shape is not representation identity",
        "fixture and deterministic fakes are diagnostic only",
    ],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def state_lineage_digest(
    previous_digest: str,
    *,
    state_epoch_id: str,
    frame_id: int,
    state_version: int,
    pcm_sha256: str,
    reference_sha256: str | None,
) -> str:
    """Chain the authenticated input envelope across listener state transitions."""
    return _sha256_bytes("|".join((
        previous_digest,
        state_epoch_id,
        str(frame_id),
        str(state_version),
        pcm_sha256,
        "none" if reference_sha256 is None else reference_sha256,
    )).encode("ascii"))


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@lru_cache(maxsize=1)
def runtime_source_fingerprint() -> dict[str, Any]:
    """Hash the source files that implement the three provenance boundaries.

    This is a runtime fingerprint, not a substitute for a clean Git source
    attestation.  A dirty checkout is deliberately represented as dirty.
    """
    root = _repo_root()
    tracked_rel_paths = (
        "duplex_kernel/server.py",
        "duplex_kernel/launch.py",
        "duplex_kernel/listener.py",
        "duplex_kernel/gemma_representation.py",
        "duplex_kernel/models/v9.py",
        "duplex_kernel/models/thinker.py",
        "duplex_kernel/native_listener.py",
        "duplex_kernel/shape_stable_listener.py",
        "duplex_kernel/session.py",
        "duplex_kernel/scheduler.py",
        "duplex_kernel/audio_stream.py",
        "duplex_kernel/config.py",
        "duplex_kernel/engine.py",
        "duplex_kernel/conditioning_provenance.py",
        "native/gemma4_listener/gemma4_listener.cu",
        "native/gemma4_listener/gemma4_listener_aten.cpp",
        "native/gemma4_listener/gemma4_listener_aten_norm.cpp",
        "native/gemma4_listener/gemma4_listener_attention_ordered.cu",
        "native/gemma4_listener/gemma4_listener_cache.cu",
        "native/gemma4_listener/gemma4_listener_rms_exact.cu",
        "native/gemma4_listener/Makefile",
        "native/gemma4_listener/ordered_vendor_manifest.json",
        "native/gemma4_listener/torch_build_config.mk",
        "native/gemma4_listener/generated_topology.h",
        "tests/fixtures/conditioning_contract/p7_same_example.json",
    )
    files = {rel: root / rel for rel in tracked_rel_paths if (root / rel).is_file()}
    file_hashes: dict[str, str | None] = {}
    dirty = False
    try:
        status = subprocess.check_output(
            ("git", "status", "--porcelain", "--", *files.keys()),
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = True
    for name, path in files.items():
        try:
            file_hashes[name] = _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            file_hashes[name] = None
    configured_library = os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH")
    if configured_library is not None:
        if not configured_library.strip():
            raise ValueError("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH must be a non-empty path")
        library_candidates = (Path(configured_library).expanduser(),)
    else:
        library_candidates = (
            root / "native" / "gemma4_listener" / "libgemma4_listener.so",
            Path("/workspace/native/gemma4_listener/libgemma4_listener.so"),
        )
    library_path = next((path.resolve() for path in library_candidates if path.is_file()), None)
    if configured_library is not None and library_path is None:
        raise FileNotFoundError(
            "GEMMA4_NATIVE_LISTENER_LIBRARY_PATH does not exist: "
            f"{configured_library}"
        )
    library_hash = None if library_path is None else _sha256_bytes(library_path.read_bytes())
    return {
        "git_sha": _git_sha(root),
        "git_dirty": dirty,
        "files": file_hashes,
        "native_library": {
            "path": None if library_path is None else str(library_path),
            "sha256": library_hash,
        },
        "attestation_kind": "runtime_source_fingerprint_not_clean_attestation",
    }


def _as_float_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError("hidden deve essere numerico")
    return array.astype(np.float32, copy=False)


def pcm_frame_provenance(pcm: Any, *, sample_rate_hz: int) -> dict[str, Any]:
    array = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("PCM provenance vuoto o non finito")
    return {
        "kind": "pcm_frame",
        "sample_rate_hz": int(sample_rate_hz),
        "sample_count": int(array.size),
        "dtype": "float32",
        "serialization": "little_endian_float32_contiguous",
        "sha256": _sha256_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes()),
    }


def _reported_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def hidden_statistics(value: Any) -> dict[str, Any]:
    source_dtype = _reported_dtype(value)
    array = _as_float_array(value)
    if array.size == 0:
        raise ValueError("hidden vuoto")
    finite = bool(np.isfinite(array).all())
    if not finite:
        raise ValueError("hidden non finito")
    flat = array.reshape(-1).astype(np.float64)
    return {
        "hidden_shape": list(array.shape),
        "hidden_dtype": source_dtype,
        "hidden_stats_dtype": str(array.dtype),
        "hidden_finite": finite,
        "hidden_sha256": _sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes()),
        "hidden_l2_norm": float(np.linalg.norm(flat)),
        "hidden_rms": float(np.sqrt(np.mean(flat * flat))),
    }


def make_trace_row(
    *,
    source_path: str,
    sample_id: str,
    frame_id: int,
    timestamp_ns: int,
    available_at_ns: int,
    available_through_frame: int,
    token_type: str,
    mode: str,
    prompt_visibility: str,
    text_visibility: str,
    frontend: str,
    hidden_layer: str,
    hidden_stage: str,
    hidden_kind: str,
    hidden: Any,
    causal_conv_stage: str,
    anchor: str | Mapping[str, Any],
    availability_kind: str,
    future_pcm_visible: bool,
    stream_id: str = "default",
    evidence_role: str | None = None,
    source_availability_evidence: str | None = None,
    source_fingerprint: Mapping[str, Any] | None = None,
    model_provenance: Mapping[str, Any] | None = None,
    projector_provenance: Mapping[str, Any] | None = None,
    input_provenance: Mapping[str, Any] | None = None,
    state_version: int | None = None,
    state_lineage: str = "offline_stateless",
    representation_id: str | None = None,
    generation_id: int | None = None,
    segment_id: str | None = None,
) -> dict[str, Any]:
    """Build one strict row; callers cannot omit temporal provenance."""
    if source_path not in _ALLOWED_PATHS:
        raise ValueError(f"source_path non riconosciuto: {source_path!r}")
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode non riconosciuto: {mode!r}")
    if not str(sample_id) or not str(stream_id):
        raise ValueError("sample_id e stream_id sono obbligatori")
    if int(frame_id) < 0 or int(timestamp_ns) < 0 or int(available_at_ns) < 0:
        raise ValueError("frame_id/timestamp_ns/available_at_ns non validi")
    if int(available_through_frame) < int(frame_id):
        raise ValueError("available_through_frame non può precedere frame_id")
    if source_availability_evidence is None:
        raise ValueError("source_availability_evidence obbligatoria")
    if evidence_role is None:
        evidence_role = (
            "microphone_causality_evidence"
            if source_path == CONDITIONING_PATHS["C"]
            else "offline_distribution_diagnostic"
        )
    stats = hidden_statistics(hidden)
    anchor_value: dict[str, Any]
    if isinstance(anchor, Mapping):
        anchor_value = dict(anchor)
    else:
        anchor_value = {"name": str(anchor), "frame_id": None, "source": str(anchor)}
    row = {
        "schema_version": CONDITIONING_PROVENANCE_SCHEMA_VERSION,
        "source_path": source_path,
        "evidence_role": evidence_role,
        "sample_id": str(sample_id),
        "stream_id": str(stream_id),
        "frame_id": int(frame_id),
        "timestamp_ns": int(timestamp_ns),
        "token_type": str(token_type),
        "mode": mode,
        "prompt_visibility": str(prompt_visibility),
        "text_visibility": str(text_visibility),
        "frontend": str(frontend),
        "hidden_layer": str(hidden_layer),
        "hidden_stage": str(hidden_stage),
        "hidden_kind": str(hidden_kind),
        "representation_id": str(representation_id or f"{source_path}:{hidden_kind}"),
        "causal_conv_stage": str(causal_conv_stage),
        "anchor": anchor_value,
        "availability": {
            "kind": str(availability_kind),
            "available_at_ns": int(available_at_ns),
            "available_through_frame": int(available_through_frame),
            "future_pcm_visible": bool(future_pcm_visible),
            "evidence": str(source_availability_evidence),
        },
        "source_fingerprint": dict(source_fingerprint or runtime_source_fingerprint()),
        "model_provenance": dict(model_provenance or {"status": "not_attested"}),
        "projector_provenance": dict(projector_provenance or {"used": False}),
        "input_provenance": dict(input_provenance or {"kind": "not_supplied", "reason": "caller did not attach source content metadata"}),
        "state_version": None if state_version is None else int(state_version),
        "state_lineage": str(state_lineage),
        "generation_id": None if generation_id is None else int(generation_id),
        "segment_id": None if segment_id is None else str(segment_id),
        **stats,
    }
    validate_trace_row(row, hidden=hidden)
    return row


def validate_trace_row(
    row: Mapping[str, Any],
    *,
    hidden: Any | None = None,
) -> None:
    missing = [field for field in REQUIRED_TRACE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"provenance fields mancanti: {', '.join(missing)}")
    if row["schema_version"] != CONDITIONING_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("schema provenance non supportato")
    if not isinstance(row["hidden_finite"], bool) or not row["hidden_finite"]:
        raise ValueError("hidden_finite deve essere true")
    if (
        not isinstance(row["hidden_shape"], list)
        or not row["hidden_shape"]
        or any(not isinstance(value, int) or value <= 0 for value in row["hidden_shape"])
        or not isinstance(row["hidden_dtype"], str)
        or not isinstance(row["hidden_stats_dtype"], str)
        or not _is_sha256(row["hidden_sha256"])
        or not _is_finite_nonnegative_number(row["hidden_l2_norm"])
        or not _is_finite_nonnegative_number(row["hidden_rms"])
    ):
        raise ValueError("statistiche hidden incomplete o non tipizzate")
    if hidden is not None:
        actual_stats = hidden_statistics(hidden)
        for field in (
            "hidden_shape", "hidden_dtype", "hidden_stats_dtype", "hidden_finite",
            "hidden_sha256", "hidden_l2_norm", "hidden_rms",
        ):
            if row[field] != actual_stats[field]:
                raise ValueError(f"statistica hidden non coincide con il payload: {field}")
    if row["source_path"] not in _ALLOWED_PATHS:
        raise ValueError("source_path provenance non supportato")
    if row["mode"] not in _ALLOWED_MODES:
        raise ValueError("mode provenance non supportato")
    if (
        not isinstance(row["sample_id"], str) or not row["sample_id"]
        or not isinstance(row["stream_id"], str) or not row["stream_id"]
        or not isinstance(row["state_lineage"], str) or not row["state_lineage"]
        or not isinstance(row["representation_id"], str) or not row["representation_id"]
    ):
        raise ValueError("identità provenance incompleta")
    if not isinstance(row["frame_id"], int) or isinstance(row["frame_id"], bool) or row["frame_id"] < 0:
        raise ValueError("frame_id provenance non intero/non negativo")
    if not isinstance(row["timestamp_ns"], int) or isinstance(row["timestamp_ns"], bool) or row["timestamp_ns"] < 0:
        raise ValueError("timestamp_ns provenance non intero/non negativo")
    if row["state_version"] is not None and (not isinstance(row["state_version"], int) or isinstance(row["state_version"], bool) or row["state_version"] < 0):
        raise ValueError("state_version provenance non intero/non negativo")
    if row["generation_id"] is not None and (not isinstance(row["generation_id"], int) or row["generation_id"] < 0):
        raise ValueError("generation_id provenance non intero/non negativo")
    if row["source_path"] == CONDITIONING_PATHS["C"] and row["generation_id"] is None:
        raise ValueError("C richiede generation_id per il fencing")
    if not isinstance(row["availability"], Mapping):
        raise ValueError("availability provenance non object")
    if not isinstance(row["input_provenance"], Mapping) or not row["input_provenance"].get("kind"):
        raise ValueError("input_provenance provenance non object o senza kind")
    input_provenance = row["input_provenance"]
    if not isinstance(row["source_fingerprint"], Mapping) or not row["source_fingerprint"].get("files"):
        raise ValueError("source_fingerprint provenance incompleto")
    source_files = row["source_fingerprint"]["files"]
    expected_source_files = set(runtime_source_fingerprint()["files"])
    if (
        not isinstance(source_files, Mapping)
        or set(source_files) != expected_source_files
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not _is_sha256(value)
            for name, value in source_files.items()
        )
    ):
        raise ValueError("source_fingerprint contiene sorgenti mancanti/non hashate/non canoniche")
    if not isinstance(row["source_fingerprint"].get("git_dirty"), bool):
        raise ValueError("source_fingerprint.git_dirty deve essere booleano")
    git_sha = row["source_fingerprint"].get("git_sha")
    if git_sha is not None and (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(character not in "0123456789abcdef" for character in git_sha)
    ):
        raise ValueError("source_fingerprint.git_sha non valido")
    if not isinstance(row["model_provenance"], Mapping) or not isinstance(row["projector_provenance"], Mapping):
        raise ValueError("model/projector provenance non object")
    native_library = row["source_fingerprint"].get("native_library")
    if not isinstance(native_library, Mapping):
        raise ValueError("source_fingerprint.native_library mancante")
    if native_library.get("path") is not None and not isinstance(native_library.get("path"), str):
        raise ValueError("source_fingerprint.native_library.path non valido")
    if native_library.get("sha256") is not None and not _is_sha256(native_library.get("sha256")):
        raise ValueError("source_fingerprint.native_library.sha256 non valido")
    if input_provenance["kind"] == "pcm_frame":
        for field in ("sample_rate_hz", "sample_count", "dtype", "sha256"):
            if field not in input_provenance or input_provenance[field] in (None, ""):
                raise ValueError(f"input_provenance.{field} mancante per PCM")
        if (
            not isinstance(input_provenance["sample_rate_hz"], int)
            or isinstance(input_provenance["sample_rate_hz"], bool)
            or input_provenance["sample_rate_hz"] <= 0
            or not isinstance(input_provenance["sample_count"], int)
            or isinstance(input_provenance["sample_count"], bool)
            or input_provenance["sample_count"] <= 0
            or not _is_sha256(input_provenance["sha256"])
        ):
            raise ValueError("input_provenance PCM tipizzata in modo errato")
    availability = row["availability"]
    for field in ("kind", "available_at_ns", "available_through_frame", "future_pcm_visible", "evidence"):
        if field not in availability or availability[field] in (None, ""):
            raise ValueError(f"availability.{field} mancante")
    if (
        not isinstance(availability["available_at_ns"], int)
        or isinstance(availability["available_at_ns"], bool)
        or availability["available_at_ns"] < 0
    ):
        raise ValueError("availability.available_at_ns non valido")
    if (
        not isinstance(availability["available_through_frame"], int)
        or isinstance(availability["available_through_frame"], bool)
        or availability["available_through_frame"] < row["frame_id"]
    ):
        raise ValueError("availability.available_through_frame non valido")
    if not isinstance(availability["future_pcm_visible"], bool):
        raise ValueError("availability.future_pcm_visible deve essere booleano")
    if not isinstance(availability["evidence"], str):
        raise ValueError("availability.evidence deve essere stringa")
    if row["source_path"] == CONDITIONING_PATHS["C"]:
        if row["mode"] != "live_causal_listener":
            raise ValueError("C deve essere live_causal_listener")
        if row["evidence_role"] not in {"microphone_causality_evidence", "fixture_causality_diagnostic"}:
            raise ValueError("C deve dichiarare evidenza causale reale o diagnostica fixture")
        if row["token_type"] != "audio_feature_token" and row["token_type"] != "v9_conditioning_frame":
            raise ValueError("C token_type non canonico")
        if row["hidden_kind"] == "h_raw" and row["hidden_stage"] != "pre_causal_conv":
            raise ValueError("C h_raw deve essere pre_causal_conv")
        if row["hidden_kind"] == "h_pfx" and row["hidden_stage"] != "post_causal_conv_projector_output":
            raise ValueError("C h_pfx deve essere l'output post-projector")
        if row["hidden_kind"] not in {"h_raw", "h_pfx"}:
            raise ValueError("C hidden_kind non canonico")
        projector_provenance = row["projector_provenance"]
        if row["hidden_kind"] == "h_raw" and projector_provenance.get("used") is not False:
            raise ValueError("C h_raw non deve avere projector applicato")
        if row["hidden_kind"] == "h_pfx":
            if (
                projector_provenance.get("used") is not True
                or projector_provenance.get("frozen") is not True
                or projector_provenance.get("training") is not False
                or projector_provenance.get("optimizer_membership") != "serving_no_optimizer"
                or not _is_sha256(projector_provenance.get("parameter_sha256_before"))
                or not _is_sha256(projector_provenance.get("parameter_sha256_after"))
                or projector_provenance["parameter_sha256_before"] != projector_provenance["parameter_sha256_after"]
            ):
                raise ValueError("C h_pfx richiede projector frozen verificato before/after")
        if row["prompt_visibility"] != "not_visible":
            raise ValueError("C prompt non deve essere visibile al listener hidden")
        if row["text_visibility"] != "separate_plan_not_in_listener_hidden":
            raise ValueError("C deve separare piano testuale e listener hidden")
        if row["frontend"] != "Gemma4AudioFeatureExtractor.incremental_left160_current1280_sentinel1":
            raise ValueError("C frontend non canonico")
        if availability["future_pcm_visible"]:
            raise ValueError("C non può dichiarare PCM futuro visibile")
        if availability["kind"] != "AudioFrame.capture_end_ns_current_frame_only":
            raise ValueError("C availability kind non canonico")
        if availability["available_through_frame"] != row["frame_id"]:
            raise ValueError("C availability non può oltrepassare il frame corrente")
        if input_provenance.get("kind") != "pcm_frame":
            raise ValueError("C richiede provenance del PCM del frame")
        if row["state_version"] is None:
            raise ValueError("C richiede state_version non nullo")
        state_epoch_id = input_provenance.get("state_epoch_id")
        if (
            not isinstance(state_epoch_id, str)
            or len(state_epoch_id) != 32
            or any(character not in "0123456789abcdef" for character in state_epoch_id)
        ):
            raise ValueError("C richiede state epoch autenticato")
        if not _is_sha256(input_provenance.get("state_lineage_digest")):
            raise ValueError("C richiede rolling state lineage digest")
        if (
            input_provenance.get("sample_rate_hz") != 16_000
            or input_provenance.get("sample_count") != 1_280
            or input_provenance.get("dtype") != "float32"
            or input_provenance.get("serialization") != "little_endian_float32_contiguous"
        ):
            raise ValueError("C richiede PCM normalizzato esattamente 16 kHz/1280 float32")
        if not _is_sha256(input_provenance.get("transport_sha256")):
            raise ValueError("C richiede hash del trasporto PCM raw")
        if input_provenance.get("transport_serialization") not in {
            "pcm_s16le", "normalized_float32_little_endian"
        }:
            raise ValueError("C transport serialization non supportata")
        if input_provenance.get("capture_timestamp_source") not in {
            "server_receive_clock", "microphone_hardware_clock"
        }:
            raise ValueError("C capture timestamp source non attestata")
        if (
            not isinstance(input_provenance.get("audio_frame_id"), int)
            or isinstance(input_provenance.get("audio_frame_id"), bool)
            or input_provenance.get("audio_frame_id") != row["frame_id"]
        ):
            raise ValueError("C deve correlare frame provenance e AudioFrame.frame_id")
        if (
            not isinstance(input_provenance.get("listener_output_frame_id"), int)
            or isinstance(input_provenance.get("listener_output_frame_id"), bool)
            or input_provenance.get("listener_output_frame_id") != row["frame_id"]
        ):
            raise ValueError("C deve correlare output listener e AudioFrame.frame_id")
        if input_provenance.get("capture_end_ns") != row["timestamp_ns"]:
            raise ValueError("C timestamp deve essere AudioFrame.capture_end_ns")
        if (
            not isinstance(input_provenance.get("capture_start_ns"), int)
            or not isinstance(input_provenance.get("capture_end_ns"), int)
            or input_provenance["capture_start_ns"] < 0
            or input_provenance["capture_start_ns"] > input_provenance["capture_end_ns"]
            or input_provenance["capture_end_ns"] - input_provenance["capture_start_ns"] != 80_000_000
        ):
            raise ValueError("C capture_start/end_ns non validi o non 80 ms")
        reference_frame_id = input_provenance.get("assistant_reference_frame_id")
        if reference_frame_id is not None and (
            not isinstance(reference_frame_id, int)
            or isinstance(reference_frame_id, bool)
            or reference_frame_id > row["frame_id"]
        ):
            raise ValueError("C assistant reference futura o non tipizzata")
        if input_provenance.get("assistant_reference_emitted_at_ns") is not None and (
            not isinstance(input_provenance["assistant_reference_emitted_at_ns"], int)
            or isinstance(input_provenance["assistant_reference_emitted_at_ns"], bool)
            or input_provenance["assistant_reference_emitted_at_ns"] < 0
        ):
            raise ValueError("C assistant reference timestamp non valido")
        if reference_frame_id is None:
            if any(input_provenance.get(field) is not None for field in (
                "assistant_reference_pcm_sha256",
                "assistant_reference_sample_count",
                "assistant_reference_sample_rate_hz",
            )):
                raise ValueError("C reference metadata presente senza reference frame")
        else:
            if not _is_sha256(input_provenance.get("assistant_reference_pcm_sha256")):
                raise ValueError("C reference PCM non hashato")
            if input_provenance.get("assistant_reference_sample_count") != 1_280:
                raise ValueError("C reference PCM non allineato a 1280 sample")
            if input_provenance.get("assistant_reference_sample_rate_hz") != 16_000:
                raise ValueError("C reference PCM non normalizzato a 16 kHz")
            if input_provenance.get("assistant_reference_emitted_at_ns") is None:
                raise ValueError("C reference frame senza timestamp di emissione")
        if row["evidence_role"] == "microphone_causality_evidence":
            if (
                not isinstance(row["stream_id"], str)
                or len(row["stream_id"]) != 32
                or any(character not in "0123456789abcdef" for character in row["stream_id"])
            ):
                raise ValueError("C evidenza microfono richiede stream_id server-owned")
            if row["source_fingerprint"] != runtime_source_fingerprint():
                raise ValueError("C evidenza microfono usa source fingerprint non corrente")
            if input_provenance.get("capture_timestamp_source") != "microphone_hardware_clock":
                raise ValueError("C evidenza microfono richiede clock microfono hardware")
            if row["model_provenance"].get("runtime_profile") != "production":
                raise ValueError("C evidenza microfono richiede runtime_profile production")
            if row["model_provenance"].get("text_context") != "separate_plan":
                raise ValueError("C evidenza microfono richiede piano testuale separato")
            if row["model_provenance"].get("native_attestation_observed") is not True:
                raise ValueError("C evidenza microfono richiede attestazione native osservata")
            if row["source_fingerprint"].get("git_dirty") is not False:
                raise ValueError("C evidenza microfono richiede source fingerprint clean")
            if not _is_sha256(native_library.get("sha256")):
                raise ValueError("C evidenza microfono richiede hash della libreria nativa")
            native_runtime = row["model_provenance"].get("native_runtime")
            if not isinstance(native_runtime, Mapping) or native_runtime.get("enabled") is not True:
                raise ValueError("C evidenza microfono richiede native runtime attivo")
            if (
                not _is_sha256(native_runtime.get("library_sha256"))
                or not _is_sha256(native_runtime.get("manifest_sha256"))
                or not _is_sha256(native_runtime.get("checkpoint_files_sha256"))
                or not isinstance(native_runtime.get("checkpoint_path"), str)
                or not native_runtime.get("checkpoint_path")
            ):
                raise ValueError("C evidenza microfono richiede hash nativa, manifest e checkpoint")
            if native_runtime.get("library_sha256") != native_library.get("sha256"):
                raise ValueError("C libreria caricata diversa dalla source fingerprint")
    if row["source_path"] == CONDITIONING_PATHS["A"]:
        if row["mode"] != "offline_full_sequence" or row["evidence_role"] != "offline_distribution_diagnostic":
            raise ValueError("A deve restare un diagnostic offline full-sequence")
    if row["source_path"] == CONDITIONING_PATHS["B"]:
        if row["mode"] != "offline_teacher_forced_text" or row["evidence_role"] != "offline_distribution_diagnostic":
            raise ValueError("B deve restare un diagnostic offline teacher-forced")
        if row["availability"]["future_pcm_visible"]:
            raise ValueError("B non può dichiarare PCM futuro disponibile")
        if row["token_type"] == "assistant_text_token":
            raise ValueError("B deve distinguere token sorgente da hidden interpolato")


def validate_trace_table(rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("tabella provenance vuota")
    seen: set[tuple[Any, ...]] = set()
    for row in materialized:
        validate_trace_row(row)
        identity = (row["source_path"], row["sample_id"], row["stream_id"], row["frame_id"], row["representation_id"])
        if identity in seen:
            raise ValueError(f"provenance identity duplicata: {identity}")
        seen.add(identity)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for row in materialized:
        group = (row["source_path"], row["sample_id"], row["stream_id"], row["representation_id"])
        groups.setdefault(group, []).append(row["frame_id"])
    for group, frame_ids in groups.items():
        ordered = sorted(frame_ids)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError(f"provenance frame_id non contigui: {group}")
    c_rows = [row for row in materialized if row["source_path"] == CONDITIONING_PATHS["C"]]
    lineage_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in c_rows:
        key = (
            row["sample_id"], row["stream_id"], row["representation_id"],
            row["input_provenance"]["state_epoch_id"],
        )
        lineage_groups.setdefault(key, []).append(row)
    for group, lineage_rows in lineage_groups.items():
        ordered_rows = sorted(lineage_rows, key=lambda item: item["frame_id"])
        previous: Mapping[str, Any] | None = None
        for row in ordered_rows:
            if previous is not None:
                if row["frame_id"] != previous["frame_id"] + 1:
                    raise ValueError(f"C state lineage frame non contiguo: {group}")
                if row["state_version"] != previous["state_version"] + 1:
                    raise ValueError(f"C state_version non contigua: {group}")
                if row["timestamp_ns"] <= previous["timestamp_ns"]:
                    raise ValueError(f"C capture timestamp non monotono: {group}")
                if row["input_provenance"]["capture_start_ns"] <= previous["input_provenance"]["capture_start_ns"]:
                    raise ValueError(f"C capture start non monotono: {group}")
                expected_digest = state_lineage_digest(
                    previous["input_provenance"]["state_lineage_digest"],
                    state_epoch_id=row["input_provenance"]["state_epoch_id"],
                    frame_id=row["frame_id"],
                    state_version=row["state_version"],
                    pcm_sha256=row["input_provenance"]["sha256"],
                    reference_sha256=row["input_provenance"].get("assistant_reference_pcm_sha256"),
                )
                if row["input_provenance"]["state_lineage_digest"] != expected_digest:
                    raise ValueError(f"C state lineage digest non coerente: {group}")
            previous = row
    c_state_views: dict[tuple[Any, ...], tuple[int, str]] = {}
    for row in c_rows:
        key = (
            row["sample_id"], row["stream_id"],
            row["input_provenance"]["state_epoch_id"], row["frame_id"],
        )
        value = (row["state_version"], row["input_provenance"]["state_lineage_digest"])
        if key in c_state_views and c_state_views[key] != value:
            raise ValueError(f"C representations divergono sullo stato: {key}")
        c_state_views[key] = value


def summarize_distribution(values: Any) -> dict[str, Any]:
    array = _as_float_array(values)
    if array.size == 0:
        raise ValueError("distribuzione vuota")
    flat = array.reshape(-1).astype(np.float64)
    if not np.isfinite(flat).all():
        raise ValueError("distribuzione non finita")
    vectors = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(1, -1)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    return {
        "count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "element_count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "mean_l2_norm": float(np.mean(norms)),
        "p05_l2_norm": float(np.percentile(norms, 5)),
        "p50_l2_norm": float(np.percentile(norms, 50)),
        "p95_l2_norm": float(np.percentile(norms, 95)),
        "finite": True,
    }


def compare_distributions(left: Any, right: Any) -> dict[str, Any]:
    """Compare marginal distributions; tensor equality is never an acceptance gate."""
    left_array = _as_float_array(left)
    right_array = _as_float_array(right)
    left_summary = summarize_distribution(left_array)
    right_summary = summarize_distribution(right_array)
    shape_equal = tuple(left_array.shape) == tuple(right_array.shape)
    tensor_equal = bool(shape_equal and np.array_equal(left_array, right_array))
    return {
        "assessment": "distribution_only",
        "shape_equal": shape_equal,
        "tensor_equal": tensor_equal,
        "left": left_summary,
        "right": right_summary,
        "mean_l2_norm_delta": abs(left_summary["mean_l2_norm"] - right_summary["mean_l2_norm"]),
        "mean_delta": abs(left_summary["mean"] - right_summary["mean"]),
        "std_delta": abs(left_summary["std"] - right_summary["std"]),
    }


def _hidden_array(output: Any) -> np.ndarray:
    value = getattr(output, "gemma_hidden_t", output)
    return _as_float_array(value)


def run_future_pcm_perturbation_probe(
    listener_factory: Callable[[], Any],
    pcm_frames: Iterable[np.ndarray],
    *,
    perturb_frame_id: int,
    perturbation: np.ndarray,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Probe no-lookahead from fresh listener states and raw PCM inputs.

    The altered data is injected before ``listener.step``.  No extracted
    ``h_raw`` cache is supplied or reused.  The baseline and perturbed runs
    each get a new listener and a new ``GemmaListenerState`` implicitly via
    the listener's step call.
    """
    frames = [np.asarray(frame, dtype=np.float32).reshape(-1) for frame in pcm_frames]
    if len(frames) < 2 or not 0 < int(perturb_frame_id) < len(frames):
        raise ValueError("perturb_frame_id deve indicare un frame futuro interno")
    if any(frame.size == 0 or not np.isfinite(frame).all() for frame in frames):
        raise ValueError("PCM fixture non valido")
    delta = np.asarray(perturbation, dtype=np.float32).reshape(-1)
    if delta.shape != frames[perturb_frame_id].shape:
        raise ValueError("perturbation deve avere la shape del frame futuro")
    perturbed_frames = [frame.copy() for frame in frames]
    perturbed_frames[perturb_frame_id] = perturbed_frames[perturb_frame_id] + delta

    baseline_listener = listener_factory()
    baseline_state = _new_listener_state(baseline_listener)
    baseline_outputs = [_hidden_array(baseline_listener.step(frame, None, baseline_state)) for frame in frames]
    perturbed_listener = listener_factory()
    perturbed_state = _new_listener_state(perturbed_listener)
    perturbed_outputs = [_hidden_array(perturbed_listener.step(frame, None, perturbed_state)) for frame in perturbed_frames]

    before_equal = [
        bool(np.allclose(baseline_outputs[index], perturbed_outputs[index], atol=atol, rtol=0.0))
        for index in range(perturb_frame_id)
    ]
    observed_inputs = []
    for listener in (baseline_listener, perturbed_listener):
        inputs = getattr(getattr(listener, "backend", None), "inputs", None)
        observed_inputs.append(inputs if isinstance(inputs, list) else None)
    source_changed = not np.array_equal(frames[perturb_frame_id], perturbed_frames[perturb_frame_id])
    observed_changed = None
    if all(isinstance(inputs, list) and len(inputs) > perturb_frame_id for inputs in observed_inputs):
        observed_changed = not np.array_equal(
            observed_inputs[0][perturb_frame_id], observed_inputs[1][perturb_frame_id]
        )
    else:
        observed_changed = False

    return {
        "status": "PASS" if source_changed and all(before_equal) and observed_changed else "FAIL",
        "perturbation_stage": "raw_pcm_before_frontend_and_gemma",
        "h_raw_cache_reuse": False,
        "fresh_listener_state_per_run": baseline_state is not perturbed_state,
        "fresh_listener_instance_per_run": baseline_listener is not perturbed_listener,
        "perturbed_future_frame_id": int(perturb_frame_id),
        "pre_perturbation_frames_unchanged": all(before_equal),
        "pre_perturbation_frame_results": before_equal,
        "future_source_input_changed": bool(source_changed),
        "future_input_observed_by_backend": bool(observed_changed),
        "future_input_observation_basis": "backend_raw_pcm_calls",
        "compared_frame_count": int(perturb_frame_id),
        "source_lineage": "fixture_pcm -> listener.step -> frontend -> Gemma",
    }


def _new_listener_state(listener: Any) -> Any:
    state_type = None
    try:
        from duplex_kernel.listener import GemmaListenerState
        state_type = GemmaListenerState
    except ImportError:  # pragma: no cover - package is present in production
        state_type = None
    if state_type is None:
        raise RuntimeError("GemmaListenerState non disponibile per il probe")
    return state_type()


def _fixture_pcm_sha(samples: Iterable[int]) -> str:
    values = np.asarray(list(samples), dtype="<i2")
    return _sha256_bytes(values.tobytes())


def load_conditioning_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("fixture_type") != "p7_conditioning_same_example":
        raise ValueError("fixture conditioning non valida")
    for key in ("fixture_id", "sample_id", "prompt", "reference_text", "pcm_frames", "path_sample_ids"):
        if key not in payload:
            raise ValueError(f"fixture priva di {key}")
    if payload.get("sample_rate_hz") != 16_000 or payload.get("frame_duration_ms") != 80:
        raise ValueError("fixture non conforme al clock production 16kHz/80ms")
    if payload.get("pcm_scale_to_float32") != 32768.0:
        raise ValueError("fixture PCM scale non esplicita")
    if payload.get("pcm_dtype") != "int16" or payload.get("pcm_endianness") != "little":
        raise ValueError("fixture PCM encoding non esplicita")
    if payload.get("production_contract_id") != PRODUCTION_CONTRACT_ID:
        raise ValueError("fixture non legata al contratto production dichiarato")
    if payload.get("reference_text_sha256") != _sha256_bytes(str(payload["reference_text"]).encode("utf-8")):
        raise ValueError("hash reference_text fixture errato")
    if not isinstance(payload.get("path_semantics"), Mapping):
        raise ValueError("fixture path_semantics mancante")
    frames = payload["pcm_frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("fixture senza pcm_frames")
    expected_id = 0
    float_frames: list[np.ndarray] = []
    for frame in frames:
        if frame.get("frame_id") != expected_id:
            raise ValueError("fixture frame IDs non contigui")
        if frame.get("capture_start_ns") != expected_id * 80_000_000 or frame.get("capture_end_ns") != (expected_id + 1) * 80_000_000:
            raise ValueError("fixture capture timestamps non contigui")
        if frame.get("sample_count") != 1280:
            raise ValueError("fixture sample_count non valido")
        samples = frame.get("samples")
        if not isinstance(samples, list) or len(samples) != 1280:
            raise ValueError("fixture richiede 1280 sample per frame")
        if any(not isinstance(value, int) or value < -32768 or value > 32767 for value in samples):
            raise ValueError("fixture contiene PCM int16 non valido")
        expected_hash = frame.get("pcm_sha256")
        if expected_hash != _fixture_pcm_sha(samples):
            raise ValueError(f"hash PCM fixture errato al frame {expected_id}")
        float_frames.append(np.asarray(samples, dtype=np.float32) / 32768.0)
        expected_id += 1
    if tuple(payload["path_sample_ids"].get(name) for name in ("A", "B", "C")) != (payload["sample_id"],) * 3:
        raise ValueError("A/B/C non condividono lo stesso sample_id fixture")
    if not payload["prompt"] or not payload["reference_text"]:
        raise ValueError("fixture prompt/reference vuoti")
    computed_content_hash = _sha256_bytes(
        b"".join(np.asarray(frame["samples"], dtype="<i2").tobytes() for frame in frames)
    )
    if payload.get("pcm_content_sha256") != computed_content_hash:
        raise ValueError("hash PCM globale fixture errato")
    diagnostic_hidden = payload.get("diagnostic_hidden")
    if not isinstance(diagnostic_hidden, Mapping):
        raise ValueError("fixture diagnostic_hidden mancante")
    a_hidden = np.asarray(diagnostic_hidden.get("A_audio_position_hidden"), dtype=np.float32)
    b_hidden = np.asarray(diagnostic_hidden.get("B_assistant_token_hidden"), dtype=np.float32)
    if (
        a_hidden.shape != (len(frames), 4)
        or b_hidden.ndim != 2
        or b_hidden.shape[1] != 4
        or not np.isfinite(a_hidden).all()
        or not np.isfinite(b_hidden).all()
    ):
        raise ValueError("fixture hidden diagnostici non allineati/finite")
    payload["pcm_float_frames"] = float_frames
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_provenance_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a provenance sidecar once without clobbering historical P7 data."""
    target = Path(path)
    resolved = target.resolve()
    p7_root = (_repo_root() / "reports" / "p7").resolve()
    if resolved == p7_root or p7_root in resolved.parents:
        raise ValueError("provenance output deve stare fuori da reports/p7")
    if target.exists():
        raise FileExistsError(f"provenance output già esistente (no-clobber): {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


__all__ = [
    "CONDITIONING_PATHS",
    "CONDITIONING_PROVENANCE_SCHEMA_VERSION",
    "PRODUCTION_CONTRACT",
    "PROVENANCE_SCHEMA",
    "PRODUCTION_CONTRACT_ID",
    "REQUIRED_TRACE_FIELDS",
    "compare_distributions",
    "hidden_statistics",
    "load_conditioning_fixture",
    "make_trace_row",
    "pcm_frame_provenance",
    "run_future_pcm_perturbation_probe",
    "runtime_source_fingerprint",
    "summarize_distribution",
    "validate_trace_row",
    "validate_trace_table",
    "write_json",
    "write_provenance_json",
]
