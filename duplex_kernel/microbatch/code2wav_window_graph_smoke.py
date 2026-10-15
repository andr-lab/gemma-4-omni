#!/usr/bin/env python3
"""Offline Code2Wav exact-window graph parity and memory smoke.

Default execution is a CPU-only preflight.  ``--run-cuda`` is a separate,
explicit host-side operation: it runs the repository's immutable managed-
container/GPU-release barrier, starts one non-root, network-isolated worker,
and restores the original managed container in ``finally``.  The worker loads
only the cached Qwen3-TTS tokenizer/Code2Wav model; it does not load Gemma or a
serving engine.

No results from this script imply an 80 ms latency guarantee.  The recorded
CUDA replay timing measures this standalone decoder call only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import wave
from typing import Any, Callable, Mapping, Sequence
import uuid


# Fail closed against Hugging Face network access in every mode, including
# imports performed by the preflight and the future GPU worker.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

DEFAULT_WAV_RELATIVE = Path(
    "artifacts/pair_duplex_e2e_proof_20261010T220755Z/"
    "response_A_resp_a36357143f17.wav"
)
DEFAULT_MAINTENANCE_CONTAINER = "gemma4-speech-runtime-query-film-v2-20261009"
RUNTIME_PYTHON = "/workspace/.venv_e2e_runtime/bin/python"
TOKENIZER_CACHE_RELATIVE = Path(
    ".hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots"
)
BASE_SPEECH_TOKENIZER_CACHE_RELATIVE = Path(
    ".hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots"
)
MAX_CODE2WAV_WEIGHTS_BYTES = 1_000_000_000
EXPECTED_TORCH_BASE_VERSION = "2.12.1"
SOURCE_HASH_PATHS = (
    Path("duplex_kernel/config.py"),
    Path("duplex_kernel/engine.py"),
    Path("duplex_kernel/models/talker.py"),
    Path("duplex_kernel/speaker_runtime.py"),
    Path("tests/kernel/test_p4_concurrency_contract.py"),
    Path("tools/kernel_gates/code2wav_window_graph_smoke.py"),
)
EXPECTED_SAMPLE_RATE = 24_000
EXPECTED_MIN_RVQ_FRAMES = 74
CODEBOOKS = 16
CODEBOOK_SIZE = 2048
FRAME_SAMPLES = 1920
NEW_FRAMES = 2
WINDOW_FRAMES = tuple(range(NEW_FRAMES, 74 + 1, NEW_FRAMES))
ARBITRARY_REPLAY_WINDOW_ORDER = (2, 74, 14, 2, 70, 4, 74, 16, 6, 14, 2)
ODD_TAIL_ARBITRARY_REPLAY_WINDOW_ORDER = (2, 74, 14, 1, 70, 3, 4, 73, 16, 6, 14, 2)
DEFAULT_MAX_CAPTURE_RESERVED_DELTA_BYTES = 3_000_000_000
PRODUCTION_GRAPH_ENVIRONMENT = (
    "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1",
    "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2",
)
ODD_TAIL_GRAPH_ENVIRONMENT = "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1"
EVEN_ONLY_GRAPH_ENVIRONMENT = "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=0"

# Tolerance is fixed from two eager repeats before any graph result is read.
# If the eager baseline exceeds either ceiling, the smoke fails as unstable.
EAGER_REPEAT_MULTIPLIER = 4.0
ABS_TOL_FLOOR = 1e-6
REL_TOL_FLOOR = 1e-5
ABS_TOL_CEILING = 1e-4
REL_TOL_CEILING = 1e-3
RELATIVE_DENOMINATOR_FLOOR = 1e-3


def _workspace_default() -> Path:
    if Path("/workspace").is_dir():
        return Path("/workspace")
    return Path(__file__).resolve().parents[2]


def _inside_workspace(workspace: Path, path: Path) -> Path:
    resolved_workspace = workspace.resolve()
    resolved = path if path.is_absolute() else resolved_workspace / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path}") from exc
    return resolved


def _relative_workspace_path(workspace: Path, path: Path) -> str:
    return _inside_workspace(workspace, path).relative_to(workspace.resolve()).as_posix()


def _torch_base_version(version: str) -> str:
    return str(version).split("+", 1)[0]


def _production_graph_config_report(
    enabled: bool,
    decode_frames: int,
    odd_tail_enabled: bool = False,
    odd_tail_requested: bool | None = None,
) -> dict[str, Any]:
    errors = []
    if not enabled:
        errors.append("CODE2WAV_GRAPH_FAMILY_ENABLED is false")
    if decode_frames != 2:
        errors.append(f"CODE2WAV_DECODE_FRAMES is {decode_frames}, expected 2")
    if odd_tail_requested is not None and odd_tail_enabled != odd_tail_requested:
        errors.append(
            "CODE2WAV_GRAPH_ODD_TAIL_ENABLED is "
            f"{odd_tail_enabled}, requested {odd_tail_requested}"
        )
    return {
        "CODE2WAV_GRAPH_FAMILY_ENABLED": bool(enabled),
        "CODE2WAV_DECODE_FRAMES": int(decode_frames),
        "CODE2WAV_GRAPH_ODD_TAIL_ENABLED": bool(odd_tail_enabled),
        "odd_tail_requested": odd_tail_requested,
        "passed": not errors,
        "errors": errors,
    }


def _production_graph_environment_args(*, include_odd_tail: bool = False) -> list[str]:
    values = list(PRODUCTION_GRAPH_ENVIRONMENT)
    values.append(
        ODD_TAIL_GRAPH_ENVIRONMENT
        if include_odd_tail
        else EVEN_ONLY_GRAPH_ENVIRONMENT
    )
    return [argument for value in values for argument in ("--env", value)]


def _window_frames(include_odd_tail: bool = False) -> tuple[int, ...]:
    return tuple(range(1, 75)) if include_odd_tail else WINDOW_FRAMES


def _arbitrary_replay_order(include_odd_tail: bool = False) -> tuple[int, ...]:
    return (
        ODD_TAIL_ARBITRARY_REPLAY_WINDOW_ORDER
        if include_odd_tail
        else ARBITRARY_REPLAY_WINDOW_ORDER
    )


def _output_crop_frames(window_frames: int) -> int:
    return min(NEW_FRAMES, int(window_frames))


def _production_har_parity_report(
    *,
    active_rescale_shapes: Sequence[int],
    graph_mismatch_shapes: Sequence[int],
) -> dict[str, Any]:
    active = sorted({int(shape) for shape in active_rescale_shapes})
    mismatches = sorted({int(shape) for shape in graph_mismatch_shapes})
    return {
        "passed": not mismatches,
        "active_rescale_shapes": active,
        "graph_vs_har_eager_mismatch_shapes": mismatches,
    }


def _extract_cpu_preflight_json(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("schema") == "code2wav_window_graph_cpu_preflight_v1"
        ):
            return value
    raise RuntimeError("CPU preflight stdout contains no recognized JSON result")


def _source_hashes(workspace: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_HASH_PATHS:
        source_path = _inside_workspace(workspace, relative_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"pinned source file is missing: {relative_path.as_posix()}")
        hashes[relative_path.as_posix()] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return hashes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_rope_inv_freq(rotary_embedding: Any, torch: Any) -> dict[str, Any]:
    inv_freq = getattr(rotary_embedding, "inv_freq", None)
    if inv_freq is None:
        raise RuntimeError("Code2Wav decoder rotary embedding lacks inv_freq")
    cpu = inv_freq.detach().to(device="cpu").contiguous().clone()
    payload = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "tensor": cpu,
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _compare_rope_inv_freq_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    before_tensor = before["tensor"]
    after_tensor = after["tensor"]
    same_layout = (
        tuple(before_tensor.shape) == tuple(after_tensor.shape)
        and before_tensor.dtype == after_tensor.dtype
    )
    changed_elements: list[dict[str, Any]] = []
    max_abs_delta: float | None = None
    changed_byte_count: int | None = None
    if same_layout:
        before_values = before_tensor.reshape(-1).tolist()
        after_values = after_tensor.reshape(-1).tolist()
        # Compare raw storage bytes so even signed-zero or NaN-payload changes
        # are reported rather than hidden by numeric equality.
        import torch as torch_module

        before_bytes = before_tensor.view(torch_module.uint8).reshape(-1).tolist()
        after_bytes = after_tensor.view(torch_module.uint8).reshape(-1).tolist()
        changed_byte_count = sum(left != right for left, right in zip(before_bytes, after_bytes))
        element_size = int(before_tensor.element_size())
        deltas: list[float] = []
        for index, (left, right) in enumerate(zip(before_values, after_values)):
            start = index * element_size
            if before_bytes[start : start + element_size] == after_bytes[start : start + element_size]:
                continue
            left_float = float(left)
            right_float = float(right)
            delta = abs(right_float - left_float)
            deltas.append(delta)
            changed_elements.append(
                {
                    "flat_index": index,
                    "before": left_float,
                    "after": right_float,
                    "max_abs_delta": delta,
                    "before_bytes_hex": bytes(before_bytes[start : start + element_size]).hex(),
                    "after_bytes_hex": bytes(after_bytes[start : start + element_size]).hex(),
                }
            )
        max_abs_delta = max(deltas, default=0.0)

    equal = bool(
        same_layout
        and before.get("sha256") == after.get("sha256")
        and changed_byte_count == 0
    )
    return {
        "passed": equal,
        "bitwise_equal": equal,
        "before": {key: before.get(key) for key in ("shape", "dtype", "sha256")},
        "after": {key: after.get(key) for key in ("shape", "dtype", "sha256")},
        "same_shape_and_dtype": same_layout,
        "changed_byte_count": changed_byte_count,
        "changed_elements": changed_elements,
        "max_abs_delta": max_abs_delta,
        "comparison": "bitwise_cpu_copy_after_eager_warmup_vs_after_capture",
    }


def _pool_keys_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
        return equal if isinstance(equal, bool) else bool(equal)
    except Exception:
        return False


def _pool_key_evidence(key: Any) -> dict[str, str]:
    rendered = repr(key)
    return {
        "repr": rendered,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def _aggregate_cuda_graph_pool_keys(
    pool_keys_by_slot: Sequence[Sequence[Any]],
    *,
    expected_slot_count: int,
    expected_graph_count: int,
) -> dict[str, Any]:
    """Fail closed unless each graph in a slot reports its one distinct slot pool."""
    errors: list[str] = []
    if expected_slot_count <= 0 or expected_graph_count <= 0:
        raise ValueError("expected slot and graph counts must be positive")
    if not isinstance(pool_keys_by_slot, Sequence) or isinstance(
        pool_keys_by_slot, (str, bytes)
    ):
        pool_keys_by_slot = ()
    if len(pool_keys_by_slot) != expected_slot_count:
        errors.append(
            f"expected {expected_slot_count} slot pool-key lists, got {len(pool_keys_by_slot)}"
        )

    slot_reports: list[dict[str, Any]] = []
    slot_unique_keys: list[list[Any]] = []
    for slot_index in range(expected_slot_count):
        raw_keys = pool_keys_by_slot[slot_index] if slot_index < len(pool_keys_by_slot) else ()
        if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, (str, bytes)):
            raw_keys = ()
            errors.append(f"slot {slot_index} graph pool-key list is missing or invalid")
        keys = list(raw_keys)
        if len(keys) != expected_graph_count:
            errors.append(
                f"slot {slot_index} expected {expected_graph_count} graph pool keys, got {len(keys)}"
            )
        missing_graph_indexes = [
            graph_index
            for graph_index in range(max(expected_graph_count, len(keys)))
            if graph_index >= len(keys) or keys[graph_index] is None
        ]
        if missing_graph_indexes:
            errors.append(
                f"slot {slot_index} is missing graph pool keys at {missing_graph_indexes}"
            )
        unique_keys: list[Any] = []
        for key in keys:
            if key is None:
                continue
            if not any(_pool_keys_equal(key, existing) for existing in unique_keys):
                unique_keys.append(key)
        if len(unique_keys) != 1:
            errors.append(
                f"slot {slot_index} expected one unique graph pool key, got {len(unique_keys)}"
            )
        slot_unique_keys.append(unique_keys)
        slot_reports.append(
            {
                "slot": slot_index,
                "captured_graph_count": len(keys),
                "missing_graph_indexes": missing_graph_indexes,
                "unique_pool_count": len(unique_keys),
                "pool_key": _pool_key_evidence(unique_keys[0]) if len(unique_keys) == 1 else None,
            }
        )

    distinct_across_slots = (
        len(slot_unique_keys) == expected_slot_count
        and all(len(keys) == 1 for keys in slot_unique_keys)
        and all(
            not _pool_keys_equal(slot_unique_keys[left][0], slot_unique_keys[right][0])
            for left in range(expected_slot_count)
            for right in range(left + 1, expected_slot_count)
        )
    )
    if not distinct_across_slots:
        errors.append("graph pool keys must be distinct across response slots")
    return {
        "passed": not errors,
        "expected_slot_count": expected_slot_count,
        "expected_graph_count_per_slot": expected_graph_count,
        "distinct_across_slots": distinct_across_slots,
        "slots": slot_reports,
        "errors": errors,
    }


def _capture_memory_budget_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    max_reserved_delta_bytes: int,
) -> dict[str, Any]:
    """Report CUDA allocator deltas and fail closed on missing metrics or budget."""
    if isinstance(max_reserved_delta_bytes, bool) or not isinstance(
        max_reserved_delta_bytes, int
    ) or max_reserved_delta_bytes <= 0:
        raise ValueError("max_reserved_delta_bytes must be a positive integer")
    errors: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    for phase, sample in (("before", before), ("after", after)):
        normalized[phase] = {}
        for key in ("allocated_bytes", "reserved_bytes"):
            value = sample.get(key) if isinstance(sample, Mapping) else None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{phase} memory sample is missing valid {key}")
            else:
                normalized[phase][key] = value
        free_total = sample.get("free_total_bytes") if isinstance(sample, Mapping) else None
        if (
            not isinstance(free_total, Sequence)
            or isinstance(free_total, (str, bytes))
            or len(free_total) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in free_total)
        ):
            errors.append(f"{phase} memory sample is missing valid free_total_bytes")
        else:
            normalized[phase]["free_total_bytes"] = list(free_total)

    deltas: dict[str, int | None] = {
        "allocated_delta_bytes": None,
        "reserved_delta_bytes": None,
        "free_delta_bytes": None,
    }
    if "allocated_bytes" in normalized["before"] and "allocated_bytes" in normalized["after"]:
        deltas["allocated_delta_bytes"] = (
            normalized["after"]["allocated_bytes"] - normalized["before"]["allocated_bytes"]
        )
    if "reserved_bytes" in normalized["before"] and "reserved_bytes" in normalized["after"]:
        reserved_delta = normalized["after"]["reserved_bytes"] - normalized["before"]["reserved_bytes"]
        deltas["reserved_delta_bytes"] = reserved_delta
        if reserved_delta > max_reserved_delta_bytes:
            errors.append(
                "reserved memory delta exceeds configured budget: "
                f"{reserved_delta} > {max_reserved_delta_bytes} bytes"
            )
    if (
        "free_total_bytes" in normalized["before"]
        and "free_total_bytes" in normalized["after"]
    ):
        deltas["free_delta_bytes"] = (
            normalized["after"]["free_total_bytes"][0]
            - normalized["before"]["free_total_bytes"][0]
        )
    return {
        "passed": not errors,
        "max_reserved_delta_bytes": max_reserved_delta_bytes,
        "max_reserved_delta_gb": max_reserved_delta_bytes / 1_000_000_000,
        **deltas,
        "before": normalized["before"],
        "after": normalized["after"],
        "errors": errors,
    }


def _fixture_source_manifest(
    workspace: Path,
    fixture_path: Path,
    preflight_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = _inside_workspace(workspace, fixture_path)
    observed_sha256 = _file_sha256(resolved)
    expected_sha256 = preflight_fixture.get("sha256")
    if not isinstance(expected_sha256, str) or expected_sha256 != observed_sha256:
        raise RuntimeError(
            "recorded fixture changed after CPU preflight: "
            f"expected={expected_sha256!r} observed={observed_sha256!r}"
        )
    return {
        "relative_path": _relative_workspace_path(workspace, resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": observed_sha256,
    }


def _snapshot_asset_manifest(workspace: Path, snapshot: Path) -> dict[str, Any]:
    snapshot = _inside_workspace(workspace, snapshot)
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Code2Wav config is missing: {config_path}")
    weight_paths = sorted(
        path for path in snapshot.rglob("*")
        if path.is_file() and (
            path.name.endswith(".safetensors")
            or path.name.startswith("pytorch_model") and path.suffix == ".bin"
        )
    )
    if not weight_paths:
        raise FileNotFoundError(f"Code2Wav weights are missing under {snapshot}")
    weight_bytes = sum(path.stat().st_size for path in weight_paths)
    if weight_bytes > MAX_CODE2WAV_WEIGHTS_BYTES:
        raise RuntimeError(
            f"cached Code2Wav weights exceed the 1 GB bound: {weight_bytes} bytes"
        )
    files = [config_path, *weight_paths]
    return {
        "snapshot_relative_path": _relative_workspace_path(workspace, snapshot),
        "weight_bytes": weight_bytes,
        "max_weight_bytes": MAX_CODE2WAV_WEIGHTS_BYTES,
        "files": [
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in files
        ],
    }


def _verify_snapshot_asset_manifest(
    workspace: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    relative = expected.get("snapshot_relative_path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("preflight manifest lacks a relative Code2Wav snapshot path")
    expected_snapshot = _inside_workspace(workspace, Path(relative))
    selected_snapshot = _tokenizer_snapshot(workspace)
    if expected_snapshot != selected_snapshot.resolve():
        raise RuntimeError(
            "GPU worker selected a different Code2Wav snapshot than CPU preflight: "
            f"expected={expected_snapshot} observed={selected_snapshot.resolve()}"
        )
    observed = _snapshot_asset_manifest(workspace, expected_snapshot)
    if dict(expected) != observed:
        raise RuntimeError(
            "Code2Wav config/weight snapshot changed after CPU preflight: "
            f"expected={dict(expected)!r} observed={observed!r}"
        )
    return {"verified": True, "expected": dict(expected), "observed": observed}


def _write_source_manifest(
    workspace: Path,
    manifest_path: Path,
    code2wav_snapshot_manifest: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema": "code2wav_window_graph_source_manifest_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _source_hashes(workspace),
        "code2wav_snapshot": dict(code2wav_snapshot_manifest),
        "fixture_wav": dict(fixture_manifest),
    }
    manifest_path = _inside_workspace(workspace, manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def _verify_source_manifest(
    workspace: Path, manifest_path: Path, fixture_path: Path
) -> dict[str, Any]:
    manifest_path = _inside_workspace(workspace, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "code2wav_window_graph_source_manifest_v3":
        raise RuntimeError("unexpected source manifest schema")
    expected = manifest.get("source_sha256")
    observed = _source_hashes(workspace)
    if not isinstance(expected, dict) or expected != observed:
        raise RuntimeError(
            "source files changed between host pre-pause pin and GPU worker: "
            f"expected={expected!r} observed={observed!r}"
        )
    expected_snapshot = manifest.get("code2wav_snapshot")
    if not isinstance(expected_snapshot, dict):
        raise RuntimeError("source manifest lacks the preflight Code2Wav snapshot pin")
    observed_snapshot = _verify_snapshot_asset_manifest(workspace, expected_snapshot)
    expected_fixture = manifest.get("fixture_wav")
    if not isinstance(expected_fixture, dict):
        raise RuntimeError("source manifest lacks the preflight recorded-WAV fixture pin")
    expected_fixture_path = _inside_workspace(
        workspace, Path(str(expected_fixture.get("relative_path", "")))
    )
    observed_fixture_path = _inside_workspace(workspace, fixture_path)
    if expected_fixture_path != observed_fixture_path:
        raise RuntimeError(
            "GPU worker selected a different recorded-WAV fixture than CPU preflight: "
            f"expected={expected_fixture_path} observed={observed_fixture_path}"
        )
    observed_fixture_sha256 = _file_sha256(observed_fixture_path)
    if expected_fixture.get("sha256") != observed_fixture_sha256:
        raise RuntimeError(
            "recorded-WAV fixture changed between host pre-pause pin and GPU worker: "
            f"expected={expected_fixture.get('sha256')!r} observed={observed_fixture_sha256!r}"
        )
    observed_fixture = {
        "relative_path": _relative_workspace_path(workspace, observed_fixture_path),
        "size_bytes": observed_fixture_path.stat().st_size,
        "sha256": observed_fixture_sha256,
    }
    if expected_fixture != observed_fixture:
        raise RuntimeError(
            "recorded-WAV fixture manifest changed between host pre-pause pin and GPU worker: "
            f"expected={expected_fixture!r} observed={observed_fixture!r}"
        )
    return {
        "verified": True,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "code2wav_snapshot": observed_snapshot,
        "fixture_wav": {"verified": True, "expected": expected_fixture, "observed": observed_fixture},
    }


def _read_waveform(path: Path) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        sample_rate = int(wav.getframerate())
        frame_count = int(wav.getnframes())
        if channels != 1:
            raise RuntimeError(f"fixture must be mono, got {channels} channels")
        if sample_width != 2:
            raise RuntimeError(f"fixture must be 16-bit PCM, got {sample_width * 8}-bit")
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise RuntimeError(
                f"fixture rate {sample_rate} != {EXPECTED_SAMPLE_RATE} Hz; no implicit resampling"
            )
        raw = wav.readframes(frame_count)
    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size != frame_count:
        raise RuntimeError(f"WAV frame count mismatch: header={frame_count}, decoded={pcm.size}")
    waveform = pcm.astype(np.float32) / 32768.0
    info = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "sample_count": int(pcm.size),
        "duration_s": frame_count / sample_rate,
        "estimated_12hz_frames": frame_count / sample_rate * 12.0,
    }
    return waveform, info


def _tokenizer_snapshot(workspace: Path) -> Path:
    parent = workspace / TOKENIZER_CACHE_RELATIVE
    snapshots = sorted(path for path in parent.glob("*") if path.is_dir())
    standalone = [path for path in snapshots if (path / "config.json").is_file()]
    if standalone:
        snapshot = standalone[-1]
    else:
        base_parent = workspace / BASE_SPEECH_TOKENIZER_CACHE_RELATIVE
        base_snapshots = sorted(
            path / "speech_tokenizer"
            for path in base_parent.glob("*")
            if path.is_dir() and (path / "speech_tokenizer").is_dir()
        )
        usable = [path for path in base_snapshots if (path / "config.json").is_file()]
        if not usable:
            raise FileNotFoundError(
                "cached Qwen Code2Wav config not found under standalone tokenizer snapshots "
                f"{parent} or cached Base speech_tokenizer snapshots {base_parent}"
            )
        snapshot = usable[-1]
    weight_files = [
        path for path in snapshot.rglob("*")
        if path.is_file() and (
            path.name.endswith(".safetensors")
            or path.name.startswith("pytorch_model") and path.suffix == ".bin"
        )
    ]
    if not weight_files:
        raise FileNotFoundError(f"snapshot has no local model weights: {snapshot}")
    weight_bytes = sum(path.stat().st_size for path in weight_files)
    if weight_bytes > MAX_CODE2WAV_WEIGHTS_BYTES:
        raise RuntimeError(
            f"cached Code2Wav weights exceed the 1 GB preflight bound: {weight_bytes} bytes"
        )
    return snapshot


def _load_code2wav_model(model_class: Any, snapshot: Path, device: str) -> Any:
    """Load only the cached Qwen3-TTS TokenizerV2 model, with Hub access disabled."""
    signature = inspect.signature(model_class.from_pretrained)
    accepts_local_only = (
        "local_files_only" in signature.parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )
    if not accepts_local_only:
        raise RuntimeError("Code2Wav model loader cannot accept local_files_only=True")
    model = model_class.from_pretrained(str(snapshot), local_files_only=True)
    model = model.to(device=device).eval().requires_grad_(False)
    return model


def _encode_recorded_waveform(model: Any, waveform: Any, device: str) -> Any:
    """Use the installed Qwen TokenizerV2 model.encode API on 24 kHz mono samples."""
    import torch

    input_values = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
    padding_mask = torch.ones_like(input_values, dtype=torch.long)
    with torch.inference_mode():
        return model.encode(input_values=input_values, padding_mask=padding_mask)


def _cpu_preflight(
    workspace: Path,
    wav_path: Path,
    *,
    contract_only: bool = False,
    no_model_load: bool = False,
    include_odd_tail: bool = False,
) -> dict[str, Any]:
    # Make this process incapable of selecting a GPU before Torch is imported.
    # The host barrier also passes this setting via docker exec; enforce it
    # here so direct/default preflight invocations have the same CPU contract.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    errors: list[str] = []
    result: dict[str, Any] = {
        "schema": "code2wav_window_graph_cpu_preflight_v1",
        "offline_env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
        },
        "workspace": str(workspace.resolve()),
        "contract_only": contract_only,
        "model_load_skipped": no_model_load,
        "model_loaded": False,
        "cuda_initialized": False,
        "errors": errors,
    }

    try:
        import torch

        result["python"] = sys.version.split()[0]
        result["python_executable"] = sys.executable
        result["torch"] = {
            "version": torch.__version__,
            "module_path": str(Path(torch.__file__).resolve()),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
        }
        result["cuda_initialized"] = bool(torch.cuda.is_initialized())
        if result["cuda_initialized"]:
            errors.append("CPU preflight found CUDA already initialized")
        if torch.cuda.is_available():
            errors.append("CPU preflight can still see a CUDA device")
        torch_base_version = _torch_base_version(torch.__version__)
        if not contract_only and torch_base_version != EXPECTED_TORCH_BASE_VERSION:
            errors.append(
                f"runtime torch {torch.__version__!r} does not match "
                f"expected base version {EXPECTED_TORCH_BASE_VERSION!r}"
            )

        from duplex_kernel.listener import FairModelLock
        from duplex_kernel.speaker_runtime import Code2WavWindowGraph, Code2WavWindowGraphFamily
        from duplex_kernel import config

        del FairModelLock, Code2WavWindowGraph
        result["production_graph_config"] = _production_graph_config_report(
            config.CODE2WAV_GRAPH_FAMILY_ENABLED,
            config.CODE2WAV_DECODE_FRAMES,
            config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED,
            odd_tail_requested=include_odd_tail,
        )
        family_shapes = tuple(
            Code2WavWindowGraphFamily.ODD_TAIL_WINDOW_FRAMES
            if include_odd_tail
            else Code2WavWindowGraphFamily.WINDOW_FRAMES
        )
        replay_order = _arbitrary_replay_order(include_odd_tail)
        result["graph_contract"] = {
            "window_frames": list(family_shapes),
            "window_count": len(family_shapes),
            "new_frames": NEW_FRAMES,
            "frame_samples": FRAME_SAMPLES,
            "odd_tail_requested": include_odd_tail,
            "odd_eos_73_supported": 73 in family_shapes,
            "arbitrary_replay_window_order": list(replay_order),
        }
        if family_shapes != _window_frames(include_odd_tail):
            errors.append(f"graph family window contract mismatch: {family_shapes!r}")
        if (73 in family_shapes) != include_odd_tail:
            errors.append("odd 73-frame EOS capture does not match the explicit opt-in")
        if any(frames not in family_shapes for frames in replay_order):
            errors.append("arbitrary replay order includes an unsupported graph shape")
        cuda_after_project_imports = bool(torch.cuda.is_initialized())
        result["cuda_initialized_after_project_imports"] = cuda_after_project_imports
        if cuda_after_project_imports:
            errors.append("CPU preflight project imports initialized CUDA")

        if not wav_path.is_file():
            errors.append(f"fixture WAV is missing: {wav_path}")
        else:
            waveform, wav_info = _read_waveform(wav_path)
            result["fixture"] = wav_info
            if wav_info["estimated_12hz_frames"] < EXPECTED_MIN_RVQ_FRAMES:
                errors.append(
                    "fixture duration is too short for 74 RVQ frames at 12 Hz: "
                    f"{wav_info['estimated_12hz_frames']:.2f} estimated"
                )

        if not contract_only:
            try:
                snapshot = _tokenizer_snapshot(workspace)
                result["cached_snapshot"] = {
                    "path": str(snapshot),
                    "config_present": True,
                    "weights_bytes": sum(
                        path.stat().st_size for path in snapshot.rglob("*")
                        if path.is_file() and (
                            path.name.endswith(".safetensors")
                            or path.name.startswith("pytorch_model") and path.suffix == ".bin"
                        )
                    ),
                    "max_weights_bytes": MAX_CODE2WAV_WEIGHTS_BYTES,
                    "source": (
                        "standalone_tokenizer_snapshot"
                        if snapshot.parent == workspace / TOKENIZER_CACHE_RELATIVE
                        else "cached_base_speech_tokenizer_subfolder"
                    ),
                    "asset_manifest": _snapshot_asset_manifest(workspace, snapshot),
                }
                if not no_model_load:
                    from duplex_kernel.models.talker import _install_shims

                    _install_shims(str(workspace))
                    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                        Qwen3TTSTokenizerV2Model,
                    )

                    result["qwen_tts"] = {
                        "module_path": str(Path(sys.modules["qwen_tts"].__file__).resolve()),
                        "model_class": (
                            f"{Qwen3TTSTokenizerV2Model.__module__}."
                            f"{Qwen3TTSTokenizerV2Model.__name__}"
                        ),
                    }
                if not no_model_load:
                    model = _load_code2wav_model(Qwen3TTSTokenizerV2Model, snapshot, "cpu")
                    result["model_loaded"] = True
                    decoder = getattr(model, "decoder", None)
                    if decoder is None or not callable(getattr(decoder, "chunked_decode", None)):
                        raise RuntimeError("cached TokenizerV2 model lacks decoder.chunked_decode")
                    sliding_window = int(model.config.decoder_config.sliding_window)
                    upsample_rate = int(model.config.decode_upsample_rate)
                    decoder_codebook_size = int(model.config.decoder_config.codebook_size)
                    encoder_codebook_size = int(model.config.encoder_config.codebook_size)
                    if (
                        sliding_window != 72
                        or upsample_rate != FRAME_SAMPLES
                        or decoder_codebook_size != CODEBOOK_SIZE
                        or encoder_codebook_size != CODEBOOK_SIZE
                    ):
                        raise RuntimeError(
                            f"Code2Wav contract mismatch: sliding_window={sliding_window}, "
                            f"decode_upsample_rate={upsample_rate}, "
                            f"decoder_codebook_size={decoder_codebook_size}, "
                            f"encoder_codebook_size={encoder_codebook_size}"
                        )
                    from duplex_kernel.models.talker import _rebuild_code2wav_rotary_buffers

                    rebuilt = int(_rebuild_code2wav_rotary_buffers(model))
                    if "fixture" not in result:
                        raise RuntimeError("recorded WAV fixture was not validated")
                    with torch.inference_mode():
                        encoded = _encode_recorded_waveform(model, waveform, "cpu")
                    codes = _normalize_audio_codes(encoded)
                    if int(codes.shape[0]) < EXPECTED_MIN_RVQ_FRAMES:
                        raise RuntimeError(
                            f"recorded WAV encoded to {codes.shape[0]} RVQ frames; need at least "
                            f"{EXPECTED_MIN_RVQ_FRAMES}"
                        )
                    result["decoder_preflight"] = {
                        "device": str(next(model.parameters()).device),
                        "sliding_window": sliding_window,
                        "decode_upsample_rate": upsample_rate,
                        "decoder_codebook_size": decoder_codebook_size,
                        "encoder_codebook_size": encoder_codebook_size,
                        "rotary_buffers_rebuilt": rebuilt,
                        "chunked_decode_callable": True,
                        "rvq_encode_api": "Qwen3TTSTokenizerV2Model.encode(input_values,padding_mask)",
                        "rvq_fixture_source": "recorded_mono_pcm16_24khz_wav",
                        "rvq_shape_frames_codebooks": [int(codes.shape[0]), int(codes.shape[1])],
                        "rvq_min": int(codes.min()),
                        "rvq_max": int(codes.max()),
                        "rvq_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                    }
                    del model
                    import gc
                    gc.collect()
            except Exception as exc:
                errors.append(f"Qwen offline import/cache preflight failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"CPU preflight import failed: {type(exc).__name__}: {exc}")

    if "torch" in locals():
        cuda_after_all_imports = bool(torch.cuda.is_initialized())
        result["cuda_initialized_after_all_imports"] = cuda_after_all_imports
        if cuda_after_all_imports and "CPU preflight imports initialized CUDA" not in errors:
            errors.append("CPU preflight imports initialized CUDA")
        if torch.cuda.is_available() and "CPU preflight can still see a CUDA device" not in errors:
            errors.append("CPU preflight can still see a CUDA device")
    result["passed"] = not errors
    return result


def _normalize_audio_codes(encoded: Any) -> Any:
    import numpy as np
    import torch

    codes = encoded.audio_codes if hasattr(encoded, "audio_codes") else encoded
    if isinstance(codes, (list, tuple)):
        if len(codes) != 1:
            raise RuntimeError(f"expected one tokenizer item, got batch size {len(codes)}")
        codes = codes[0]
    if isinstance(codes, torch.Tensor):
        codes = codes.detach().to(device="cpu").numpy()
    codes = np.asarray(codes)
    if codes.ndim == 3:
        if codes.shape[0] != 1:
            raise RuntimeError(f"unexpected tokenizer batch shape {codes.shape}")
        codes = codes[0]
    if codes.ndim != 2:
        raise RuntimeError(f"unexpected RVQ rank: {codes.shape}")
    if codes.shape[0] == CODEBOOKS:
        codes = codes.transpose(1, 0)
    if codes.shape[1] != CODEBOOKS or codes.shape[0] == 0:
        raise RuntimeError(f"RVQ shape must be (frames,{CODEBOOKS}), got {codes.shape}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise RuntimeError(f"RVQ dtype must be integer, got {codes.dtype}")
    code_min, code_max = int(codes.min()), int(codes.max())
    if code_min < 0 or code_max >= CODEBOOK_SIZE:
        raise RuntimeError(f"RVQ values outside [0,{CODEBOOK_SIZE}): {code_min}..{code_max}")
    return np.ascontiguousarray(codes, dtype=np.int64)


def _decode_public_with_har_probe(model: Any, codes: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from duplex_kernel.models.talker import (
        capture_code2wav_preclamp,
        rescale_code2wav_preclamp,
    )

    raw_clamped: list[Any] = []
    preclamp_stats: list[dict[str, Any]] = []

    def har_preclamp(value: Any) -> Any:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("Code2Wav pre-clamp hook returned a non-floating tensor")
        raw = value.detach().float()
        abs_raw = raw.abs()
        peak = abs_raw.amax()
        peak_value = float(peak.item())
        preclamp_stats.append({
            "peak_abs": peak_value,
            "samples_over_0_95": int((abs_raw > 0.95).sum().item()),
            "samples_at_or_over_hard_rail_1_0": int((abs_raw >= 1.0).sum().item()),
            "har_rescale_active": peak_value > 0.95,
        })
        raw_clamped.append(value.detach().clamp(-1.0, 1.0).float().reshape(-1).clone())
        return rescale_code2wav_preclamp(value, max_peak=0.95)

    with capture_code2wav_preclamp(model, har_preclamp):
        with torch.inference_mode():
            decoded = model.decode(codes)
    audio = getattr(decoded, "audio_values", None)
    if audio is None:
        raise RuntimeError("Qwen Code2Wav HAR decode did not return audio_values")
    if isinstance(audio, (list, tuple)):
        if len(audio) != 1:
            raise RuntimeError(f"unexpected decoded batch size: {len(audio)}")
        audio = audio[0]
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio, device=codes.device)
    audio = audio.detach().to(device=codes.device, dtype=torch.float32).reshape(-1)
    if not torch.isfinite(audio).all():
        raise RuntimeError("production HAR Code2Wav output contains non-finite samples")
    if len(raw_clamped) != 1 or len(preclamp_stats) != 1:
        raise RuntimeError(
            "expected one final-block pre-clamp hook call for a <=74-frame decode, "
            f"got raw={len(raw_clamped)} stats={len(preclamp_stats)}"
        )
    raw_audio = raw_clamped[0].to(device=codes.device)
    if tuple(raw_audio.shape) != tuple(audio.shape):
        raise RuntimeError(
            f"pre-clamp raw output shape {tuple(raw_audio.shape)} != HAR output {tuple(audio.shape)}"
        )
    if not torch.isfinite(raw_audio).all():
        raise RuntimeError("raw pre-clamp Code2Wav output contains non-finite samples")
    return audio, raw_audio, preclamp_stats[0]


def _error_metrics(reference: Any, candidate: Any) -> dict[str, float | int | bool]:
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        raise RuntimeError(f"output shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    ref = reference.detach().float()
    got = candidate.detach().float()
    ref_finite = torch.isfinite(ref)
    got_finite = torch.isfinite(got)
    if not bool(ref_finite.all()) or not bool(got_finite.all()):
        raise RuntimeError("non-finite eager/graph decode output")
    delta = (got - ref).abs()
    relative = delta / ref.abs().clamp_min(RELATIVE_DENOMINATOR_FLOOR)
    return {
        "sample_count": int(ref.numel()),
        "reference_finite_count": int(ref_finite.sum().item()),
        "candidate_finite_count": int(got_finite.sum().item()),
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_relative_error_floor_1e-3": float(relative.max().item()) if relative.numel() else 0.0,
        "allclose": False,
    }


def _tolerance_from_eager_repeat(first: Any, second: Any) -> dict[str, float]:
    metrics = _error_metrics(first, second)
    atol = max(ABS_TOL_FLOOR, EAGER_REPEAT_MULTIPLIER * float(metrics["max_abs_error"]))
    rtol = max(REL_TOL_FLOOR, EAGER_REPEAT_MULTIPLIER * float(metrics["max_relative_error_floor_1e-3"]))
    if atol > ABS_TOL_CEILING or rtol > REL_TOL_CEILING:
        raise RuntimeError(
            "eager repeat baseline is unstable beyond preset tolerance ceilings: "
            f"atol={atol:.8g}, rtol={rtol:.8g}"
        )
    return {
        "atol": atol,
        "rtol": rtol,
        "eager_repeat_max_abs_error": float(metrics["max_abs_error"]),
        "eager_repeat_max_relative_error_floor_1e-3": float(
            metrics["max_relative_error_floor_1e-3"]
        ),
    }


def _timed_graph_decode(graph: Any, codes: Any, lock: Any) -> tuple[Any, float, float]:
    import torch

    if not lock.acquire_fair("compute"):
        raise RuntimeError("FairModelLock compute acquisition unexpectedly cancelled")
    held_started = time.perf_counter()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    try:
        started.record()
        output = graph.decode(codes)
        ended.record()
        held_ms = (time.perf_counter() - held_started) * 1000.0
    finally:
        lock.release()
    ended.synchronize()
    return output, float(started.elapsed_time(ended)), held_ms


def _serialized_graph_decode_cpu(
    graph: Any,
    codes: Any,
    lock: Any,
) -> tuple[Any, float, float]:
    """Copy the cloned graph crop to CPU before another replay can start."""
    import torch

    if not lock.acquire_fair("compute"):
        raise RuntimeError("FairModelLock compute acquisition unexpectedly cancelled")
    held_started = time.perf_counter()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    try:
        started.record()
        output = graph.decode(codes)
        host_output = output.detach().to(device="cpu", non_blocking=False)
        ended.record()
        held_ms = (time.perf_counter() - held_started) * 1000.0
    finally:
        lock.release()
    ended.synchronize()
    return host_output, float(started.elapsed_time(ended)), held_ms


def _replay_arbitrary_shape_order(
    family: Any,
    windows: Mapping[int, Any],
    replay: Callable[[Any, Any], Any],
    window_order: Sequence[int] = ARBITRARY_REPLAY_WINDOW_ORDER,
) -> list[tuple[int, Any]]:
    graphs = family.graphs
    results = []
    for window_frames in window_order:
        graph = graphs.get(window_frames)
        if graph is None or window_frames not in windows:
            raise RuntimeError(
                f"Code2Wav replay order requires exact window {window_frames}"
            )
        results.append((window_frames, replay(graph, windows[window_frames])))
    return results


def _listener_due_lock_probe(graph: Any, codes: Any) -> dict[str, float]:
    """Measure a queued listener's wait around one decoder enqueue only."""
    from duplex_kernel.listener import FairModelLock

    lock = FairModelLock()
    if not lock.acquire_fair("compute"):
        raise RuntimeError("FairModelLock compute acquisition unexpectedly cancelled")
    queued = threading.Event()
    acquired = threading.Event()
    measurements: dict[str, float] = {}
    thread_errors: list[str] = []

    def listener_waiter() -> None:
        queued_at = time.perf_counter()
        queued.set()
        try:
            if not lock.acquire_fair("listener"):
                raise RuntimeError("listener FairModelLock acquisition unexpectedly cancelled")
            measurements["listener_wait_ms"] = (time.perf_counter() - queued_at) * 1000.0
            lock.release()
            acquired.set()
        except BaseException as exc:
            thread_errors.append(f"{type(exc).__name__}: {exc}")
            acquired.set()

    thread = threading.Thread(target=listener_waiter, name="code2wav-listener-lock-probe", daemon=True)
    thread.start()
    if not queued.wait(timeout=2.0):
        lock.release()
        raise TimeoutError("listener lock probe did not queue")
    deadline = time.monotonic() + 2.0
    while lock.pending_counts["listener"] == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    if lock.pending_counts["listener"] == 0:
        lock.release()
        thread.join(timeout=2.0)
        raise TimeoutError("listener waiter was not visible in FairModelLock")

    hold_started = time.perf_counter()
    try:
        graph.decode(codes)
    finally:
        measurements["compute_lock_hold_ms"] = (time.perf_counter() - hold_started) * 1000.0
        lock.release()
    if not acquired.wait(timeout=2.0):
        raise TimeoutError("queued listener did not acquire FairModelLock after decode")
    thread.join(timeout=2.0)
    if thread_errors:
        raise RuntimeError(f"listener FairModelLock probe failed: {thread_errors[0]}")
    return measurements


def _run_cuda_worker(
    workspace: Path,
    wav_path: Path,
    output_path: Path,
    expected_source_manifest: Path,
    max_capture_reserved_delta_bytes: int,
    include_odd_tail: bool = False,
) -> dict[str, Any]:
    started_utc = datetime.now(timezone.utc).isoformat()
    source_verification = _verify_source_manifest(
        workspace, expected_source_manifest, wav_path
    )
    import numpy as np
    import torch

    from duplex_kernel.listener import FairModelLock
    from duplex_kernel import config
    from duplex_kernel.models.talker import _install_shims, _rebuild_code2wav_rotary_buffers
    from duplex_kernel.response_talk_context import ResponseTalkContextPool
    from duplex_kernel.speaker_runtime import Code2WavWindowGraphFamily

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA worker was started without CUDA visibility")
    production_graph_config = _production_graph_config_report(
        config.CODE2WAV_GRAPH_FAMILY_ENABLED,
        config.CODE2WAV_DECODE_FRAMES,
        config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED,
        odd_tail_requested=include_odd_tail,
    )
    if not production_graph_config["passed"]:
        raise RuntimeError(
            "GPU worker production graph configuration failed: "
            f"{production_graph_config['errors']!r}"
        )
    window_shapes = _window_frames(include_odd_tail)
    replay_order = _arbitrary_replay_order(include_odd_tail)
    family_contract = tuple(
        Code2WavWindowGraphFamily.ODD_TAIL_WINDOW_FRAMES
        if include_odd_tail
        else Code2WavWindowGraphFamily.WINDOW_FRAMES
    )
    if family_contract != window_shapes:
        raise RuntimeError(
            f"Code2Wav family source contract {family_contract!r} != requested {window_shapes!r}"
        )
    torch_base_version = _torch_base_version(torch.__version__)
    if torch_base_version != EXPECTED_TORCH_BASE_VERSION:
        raise RuntimeError(
            f"torch {torch.__version__!r} != expected base version {EXPECTED_TORCH_BASE_VERSION!r}"
        )
    _install_shims(str(workspace))
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    snapshot = _tokenizer_snapshot(workspace)
    waveform, wav_info = _read_waveform(wav_path)

    # Match the production Code2Wav model loader while skipping the public
    # tokenizer wrapper, whose optional feature-extractor metadata is not used
    # by the model.encode API below. No Qwen Talker or Gemma model is built.
    model = _load_code2wav_model(Qwen3TTSTokenizerV2Model, snapshot, "cuda")
    decoder = getattr(model, "decoder", None)
    if decoder is None or not callable(getattr(model, "decode", None)):
        raise RuntimeError("cached Code2Wav model lacks decoder or public decode")
    rotary_buffers_rebuilt = int(_rebuild_code2wav_rotary_buffers(model))
    decoder_config = model.config.decoder_config
    sliding_window = int(decoder_config.sliding_window)
    if sliding_window != 72:
        raise RuntimeError(f"Code2Wav sliding window {sliding_window} != 72")
    rotary_embedding = getattr(
        getattr(decoder, "pre_transformer", None), "rotary_emb", None
    )
    if getattr(rotary_embedding, "inv_freq", None) is None:
        raise RuntimeError("Code2Wav decoder rotary embedding lacks inv_freq")
    decoder_rope_config = {
        "rope_type": str(getattr(rotary_embedding, "rope_type", "default")),
        "rope_scaling": getattr(decoder_config, "rope_scaling", None),
        "rope_scaling_present": isinstance(
            getattr(decoder_config, "rope_scaling", None), dict
        ),
        "rope_theta": getattr(decoder_config, "rope_theta", None),
        "max_position_embeddings": getattr(
            decoder_config, "max_position_embeddings", None
        ),
        "sliding_window": sliding_window,
    }

    with torch.inference_mode():
        encoded = _encode_recorded_waveform(model, waveform, "cuda")
    codes_np = _normalize_audio_codes(encoded)
    if int(codes_np.shape[0]) < EXPECTED_MIN_RVQ_FRAMES:
        raise RuntimeError(
            f"fixture encoded to {codes_np.shape[0]} RVQ frames; need at least "
            f"{EXPECTED_MIN_RVQ_FRAMES}"
        )
    codes = torch.as_tensor(codes_np, dtype=torch.long, device="cuda").unsqueeze(0)
    windows = {frames: codes[:, :frames, :].contiguous() for frames in window_shapes}

    memory_after_model = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_total_bytes": [int(value) for value in torch.cuda.mem_get_info()],
    }
    torch.cuda.reset_peak_memory_stats()

    eager_outputs: dict[int, Any] = {}
    production_har_outputs: dict[int, Any] = {}
    tolerances: dict[int, dict[str, dict[str, float]]] = {}
    preclamp_observations: dict[int, list[dict[str, Any]]] = {}
    for frames in window_shapes:
        first_har, first_raw, first_preclamp = _decode_public_with_har_probe(
            model, windows[frames]
        )
        second_har, second_raw, second_preclamp = _decode_public_with_har_probe(
            model, windows[frames]
        )
        expected_samples = frames * FRAME_SAMPLES
        if (
            first_har.numel() != expected_samples
            or second_har.numel() != expected_samples
            or first_raw.numel() != expected_samples
            or second_raw.numel() != expected_samples
        ):
            raise RuntimeError(
                f"eager window {frames} returned HAR/raw samples "
                f"{first_har.numel()}/{first_raw.numel()}/{second_har.numel()}/{second_raw.numel()}, "
                f"expected {expected_samples}"
            )
        tolerances[frames] = {
            "raw_decoder": _tolerance_from_eager_repeat(first_raw, second_raw),
            "production_har": _tolerance_from_eager_repeat(first_har, second_har),
        }
        eager_outputs[frames] = first_raw
        production_har_outputs[frames] = first_har
        preclamp_observations[frames] = [first_preclamp, second_preclamp]

    memory_before_capture = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_total_bytes": [int(value) for value in torch.cuda.mem_get_info()],
    }
    # Eager decoder warmup above can legitimately initialize or update rotary
    # state. Attest the exact state that graph capture is about to consume.
    rotary_inv_freq_before_capture = _snapshot_rope_inv_freq(rotary_embedding, torch)
    decoder_rope_config.update(
        {
            "inv_freq_shape": rotary_inv_freq_before_capture["shape"],
            "inv_freq_dtype": rotary_inv_freq_before_capture["dtype"],
            "inv_freq_sha256": rotary_inv_freq_before_capture["sha256"],
            "inv_freq_sample_point": "after_eager_warmup_immediately_before_capture",
        }
    )
    family_capture_started = time.perf_counter()
    slot_families = [
        Code2WavWindowGraphFamily(
            model, device="cuda:0", include_odd_tail=include_odd_tail
        ).capture(),
        Code2WavWindowGraphFamily(
            model, device="cuda:0", include_odd_tail=include_odd_tail
        ).capture(),
    ]
    torch.cuda.synchronize()
    family_capture_seconds = time.perf_counter() - family_capture_started
    rotary_inv_freq_after_capture = _snapshot_rope_inv_freq(rotary_embedding, torch)
    rotary_inv_freq_capture_stability = _compare_rope_inv_freq_snapshots(
        rotary_inv_freq_before_capture,
        rotary_inv_freq_after_capture,
    )
    default_rope_config = (
        decoder_rope_config["rope_type"] == "default"
        and not decoder_rope_config["rope_scaling_present"]
    )
    rotary_inv_freq_capture_stability["default_rope_config"] = default_rope_config
    rotary_inv_freq_capture_stability["default_config_equality_required"] = default_rope_config
    rotary_inv_freq_capture_stability["policy"] = (
        "bitwise equality required for default RoPE; any observed mutation fails the smoke"
    )
    memory_after_capture = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_total_bytes": [int(value) for value in torch.cuda.mem_get_info()],
    }
    for index, family in enumerate(slot_families):
        info = family.runtime_info()
        if not info["captured"] or info["graph_count"] != len(window_shapes):
            raise RuntimeError(f"slot {index} captured incomplete graph family: {info}")
    graph_pool_keys_by_slot = [
        [
            family.graphs[frames].graph.pool()
            if family.graphs[frames].graph is not None
            else None
            for frames in window_shapes
        ]
        for family in slot_families
    ]
    graph_pool_report = _aggregate_cuda_graph_pool_keys(
        graph_pool_keys_by_slot,
        expected_slot_count=2,
        expected_graph_count=len(window_shapes),
    )
    memory_budget_report = _capture_memory_budget_report(
        memory_before_capture,
        memory_after_capture,
        max_reserved_delta_bytes=max_capture_reserved_delta_bytes,
    )

    parity: list[dict[str, Any]] = []
    max_replay_ms = 0.0
    max_lock_hold_ms = 0.0
    production_har_mismatch_shapes: set[int] = set()
    for slot_index, family in enumerate(slot_families):
        lock = FairModelLock()
        for frames in window_shapes:
            graph = family.graphs[frames]
            graph_output, replay_ms, lock_hold_ms = _timed_graph_decode(
                graph, windows[frames], lock
            )
            max_replay_ms = max(max_replay_ms, replay_ms)
            max_lock_hold_ms = max(max_lock_hold_ms, lock_hold_ms)
            full_graph_output = graph.static_audio
            if full_graph_output is None:
                raise RuntimeError(f"slot {slot_index} graph {frames} has no captured output")
            raw_tolerance = tolerances[frames]["raw_decoder"]
            har_tolerance = tolerances[frames]["production_har"]
            full_error = _error_metrics(eager_outputs[frames], full_graph_output)
            output_crop_frames = _output_crop_frames(frames)
            crop_reference = eager_outputs[frames][-output_crop_frames * FRAME_SAMPLES :]
            crop_error = _error_metrics(crop_reference, graph_output)
            full_error["allclose"] = bool(torch.allclose(
                eager_outputs[frames], full_graph_output,
                atol=raw_tolerance["atol"], rtol=raw_tolerance["rtol"],
            ))
            crop_error["allclose"] = bool(torch.allclose(
                crop_reference, graph_output,
                atol=raw_tolerance["atol"], rtol=raw_tolerance["rtol"],
            ))
            if not full_error["allclose"] or not crop_error["allclose"]:
                raise RuntimeError(
                    f"slot {slot_index} window {frames} raw graph/eager mismatch: "
                    f"full={full_error}, crop={crop_error}, tolerance={raw_tolerance}"
                )
            har_full_error = _error_metrics(
                production_har_outputs[frames], full_graph_output
            )
            har_crop_reference = production_har_outputs[frames][
                -output_crop_frames * FRAME_SAMPLES :
            ]
            har_crop_error = _error_metrics(har_crop_reference, graph_output)
            har_full_error["allclose"] = bool(torch.allclose(
                production_har_outputs[frames], full_graph_output,
                atol=har_tolerance["atol"], rtol=har_tolerance["rtol"],
            ))
            har_crop_error["allclose"] = bool(torch.allclose(
                har_crop_reference, graph_output,
                atol=har_tolerance["atol"], rtol=har_tolerance["rtol"],
            ))
            if not har_full_error["allclose"] or not har_crop_error["allclose"]:
                production_har_mismatch_shapes.add(frames)
            if graph_output.numel() != output_crop_frames * FRAME_SAMPLES:
                raise RuntimeError(
                    f"slot {slot_index} window {frames} crop has {graph_output.numel()} samples"
                )
            parity.append({
                "slot": slot_index,
                "window_frames": frames,
                "raw_decoder": {
                    "full_window": full_error,
                    "new_tail_crop": crop_error,
                    "tolerance": raw_tolerance,
                },
                "production_har": {
                    "full_window": har_full_error,
                    "new_tail_crop": har_crop_error,
                    "tolerance": har_tolerance,
                },
                "preclamp_observations": preclamp_observations[frames],
                "cuda_replay_ms": replay_ms,
                "compute_lock_hold_ms": lock_hold_ms,
            })

    class SmokeSlotGraph:
        device = "cpu"

        def reset(self) -> None:
            pass

    response_slots = ResponseTalkContextPool(
        [SmokeSlotGraph(), SmokeSlotGraph()],
        code2wav_graphs=slot_families,
    )
    before_eos = response_slots.acquire(
        "code2wav-smoke", "before-eos", generation_id=1
    )
    before_eos.rvq_history.append(0, torch.zeros((16,), dtype=torch.long))
    before_eos_history_count = before_eos.rvq_history.count
    response_slots.release(before_eos)
    after_eos = response_slots.acquire(
        "code2wav-smoke", "after-eos", generation_id=2
    )
    try:
        if after_eos.rvq_history.count != 0:
            raise RuntimeError("Code2Wav response slot retained RVQ history after EOS reset")
        if after_eos.code2wav_graph is not slot_families[0]:
            raise RuntimeError("Code2Wav response slot did not reacquire its original graph family")
        order_lock = FairModelLock()
        replay_order_results = _replay_arbitrary_shape_order(
            after_eos.code2wav_graph,
            windows,
            lambda graph, input_codes: _serialized_graph_decode_cpu(
                graph, input_codes, order_lock
            ),
            replay_order,
        )
        replay_order_evidence = []
        for frames, (host_output, replay_ms, lock_hold_ms) in replay_order_results:
            reference = eager_outputs[frames][
                -_output_crop_frames(frames) * FRAME_SAMPLES :
            ].detach().cpu()
            error = _error_metrics(reference, host_output)
            tolerance = tolerances[frames]["raw_decoder"]
            error["allclose"] = bool(torch.allclose(
                reference,
                host_output,
                atol=tolerance["atol"],
                rtol=tolerance["rtol"],
            ))
            if not error["allclose"]:
                raise RuntimeError(
                    f"arbitrary replay order window {frames} raw graph/eager mismatch: "
                    f"{error}, tolerance={tolerance}"
                )
            max_replay_ms = max(max_replay_ms, replay_ms)
            max_lock_hold_ms = max(max_lock_hold_ms, lock_hold_ms)
            replay_order_evidence.append({
                "window_frames": frames,
                "crop_parity": error,
                "cuda_replay_ms": replay_ms,
                "lock_hold_including_synchronous_cpu_copy_ms": lock_hold_ms,
                "cpu_copy_completed_before_next_replay": True,
            })
    finally:
        response_slots.release(after_eos)

    # Check the decoder's returned crop is an owned clone across a replay using
    # a different, still in-range code. This is an isolation check, not parity
    # data; the parity inputs above remain the unmodified encoded RVQ codes.
    wide_codes = windows[74]
    alternate = wide_codes.clone()
    alternate[0, 0, 0] = (alternate[0, 0, 0] + 1) % CODEBOOK_SIZE
    owned = slot_families[0].decode(wide_codes)
    owned_snapshot = owned.clone()
    slot_families[0].decode(alternate)
    if not torch.equal(owned, owned_snapshot):
        raise RuntimeError("slot 0 returned crop changed after same-shape replay")
    slot_one_snapshot = slot_families[0].decode(wide_codes).clone()
    slot_families[1].decode(alternate)
    if not torch.equal(slot_one_snapshot, slot_families[0].decode(wide_codes)):
        raise RuntimeError("slot 1 replay changed slot 0 output/input state")
    torch.cuda.synchronize()

    listener_due = _listener_due_lock_probe(
        slot_families[0].graphs[74], wide_codes
    )
    active_har_rescale_shapes = sorted(
        frames
        for frames, observations in preclamp_observations.items()
        if any(bool(item["har_rescale_active"]) for item in observations)
    )
    production_har_report = _production_har_parity_report(
        active_rescale_shapes=active_har_rescale_shapes,
        graph_mismatch_shapes=sorted(production_har_mismatch_shapes),
    )
    production_har_parity_passed = production_har_report["passed"]
    result = {
        "schema": "code2wav_window_graph_gpu_smoke_v1",
        "passed": (
            production_har_parity_passed
            and production_graph_config["passed"]
            and len(slot_families) == 2
            and graph_pool_report["passed"]
            and memory_budget_report["passed"]
            and rotary_inv_freq_capture_stability["passed"]
        ),
        "started_utc": started_utc,
        "restrictions": {
            "offline_only": True,
            "thinker_loaded": False,
            "qwen_talker_loaded": False,
            "server_loaded": False,
            "network_enabled_in_worker": False,
            "gpu_latency_claim": False,
            "rvq_parity_fixture_is_recorded_audio": True,
            "rvq_encoder": "Qwen3TTSTokenizerV2Model.encode(input_values,padding_mask)",
            "clone_alternate_is_valid_id_but_modified": True,
            "production_preclamp_contract_checked": True,
        },
        "production_graph_config": {
            **production_graph_config,
            "requested_slot_count": len(slot_families),
            "requested_window_count_per_slot": len(window_shapes),
            "requested_window_frames": list(window_shapes),
            "captured_graphs_by_slot": [
                int(family.runtime_info()["graph_count"]) for family in slot_families
            ],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
        "source_verification": source_verification,
            "torch_path": str(Path(torch.__file__).resolve()),
            "qwen_tts_path": str(Path(sys.modules["qwen_tts"].__file__).resolve()),
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "device_total_memory_bytes": int(
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            ),
        },
        "fixture": {
            **wav_info,
            "rvq_shape_frames_codebooks": [int(codes.shape[1]), int(codes.shape[2])],
            "rvq_dtype": str(codes.dtype),
            "rvq_min": int(codes_np.min()),
            "rvq_max": int(codes_np.max()),
            "rvq_sha256": hashlib.sha256(codes_np.tobytes()).hexdigest(),
        },
        "decoder": {
            "snapshot": str(snapshot),
            "rotary_buffers_rebuilt": rotary_buffers_rebuilt,
            "decoder_rope_config": decoder_rope_config,
            "rotary_inv_freq_capture_stability": rotary_inv_freq_capture_stability,
            "sliding_window": sliding_window,
            "frame_samples": FRAME_SAMPLES,
            "new_frames_per_decode": NEW_FRAMES,
            "one_frame_tail_enabled": include_odd_tail,
            "output_crop_frames_by_window": {
                str(frames): _output_crop_frames(frames) for frames in window_shapes
            },
        },
        "memory": {
            "after_code2wav_model_load": memory_after_model,
            "before_capture": memory_before_capture,
            "after_capture": memory_after_capture,
            "capture_budget": memory_budget_report,
            "capture_seconds_two_slots": family_capture_seconds,
            "graphs_per_slot": len(window_shapes),
            "slot_family_static_io_bytes": [
                int(family.static_buffers_bytes) for family in slot_families
            ],
            "slot_family_graph_count": [
                int(family.runtime_info()["graph_count"]) for family in slot_families
            ],
        },
        "graph_pool_validation": graph_pool_report,
        "window_parity": parity,
        "production_har_parity": {
            **production_har_report,
            "scale_headroom_threshold": 0.95,
            "note": (
                "Production eager rescales the full raw window before Qwen's hard clamp; the graph "
                "capture applies the same tensor transform. Active rescaling is diagnostic; only a "
                "full-window or emitted-tail numerical mismatch fails parity."
            ),
        },
        "max_cuda_replay_ms_across_captured_graphs": max_replay_ms,
        "max_compute_lock_hold_ms_across_captured_graphs": max_lock_hold_ms,
        "listener_due_lock_probe": {
            **listener_due,
            "note": "synthetic queued listener on FairModelLock around decoder enqueue; not a serving contention measurement",
        },
        "clone_isolation": {
            "same_slot_replay_preserved_returned_crop": True,
            "cross_slot_replay_preserved_slot0_crop": True,
        },
        "arbitrary_replay_after_eos_reset": {
            "passed": True,
            "window_frames": list(replay_order),
            "slot_id": after_eos.slot_id,
            "before_eos_rvq_history_count": before_eos_history_count,
            "after_eos_rvq_history_count": 0,
            "same_slot_family_reacquired": True,
            "synchronous_cpu_copy_serialized_replay": True,
            "replays": replay_order_evidence,
            "scope_note": (
                "EOS reset is exercised as response-slot release and reacquisition; "
                "the standalone tokenizer smoke does not load the Qwen Talker EOS sampler."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)

    # Free this process's temporary graph/model storage before the host starts
    # the managed service again. This process exits immediately after report.
    del slot_families, model, codes, eager_outputs
    import gc
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def _load_runner_module(repo_root: Path) -> Any:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools import run_pair_duplex_proof_3090 as runner

    return runner


def _host_cuda_run(
    repo_root: Path,
    wav_path: Path,
    output_path: Path,
    *,
    container_name: str,
    timeout_s: float,
    max_capture_reserved_delta_bytes: int,
    include_odd_tail: bool = False,
) -> int:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise RuntimeError("--run-cuda must run on the Linux gpu3090 host")
    if os.geteuid() == 0:
        raise RuntimeError("refusing --run-cuda as root; use the existing unprivileged gpu3090 host user")

    runner = _load_runner_module(repo_root)
    if container_name != runner.DEFAULT_MAINTENANCE_CONTAINER:
        raise ValueError(
            "maintenance target must be the repository's exact managed speech container "
            f"{runner.DEFAULT_MAINTENANCE_CONTAINER!r}"
        )
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repo_root}")
    wav_relative = _relative_workspace_path(repo_root, wav_path)
    output_relative = _relative_workspace_path(repo_root, output_path)
    if output_relative.startswith("../"):
        raise ValueError("smoke output must remain inside the mounted repository")

    snapshot: dict[str, Any] | None = None
    worker_name = f"code2wav-window-smoke-{uuid.uuid4().hex[:12]}"
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    worker_result: subprocess.CompletedProcess[str] | None = None
    try:
        snapshot = runner.snapshot_maintenance_container(container_name, repo_root)
        if snapshot.get("was_running") is not True or not snapshot.get("id"):
            raise RuntimeError("managed service is not running; refusing an independent model load")

        # CPU-only imports/cache/WAV checks run while the service remains up.
        # CUDA is hidden; this command neither loads weights nor initializes CUDA.
        preflight_argv = [
            "docker", "exec", "--user", "1000:1000",
            "--env", "CUDA_VISIBLE_DEVICES=",
            *_production_graph_environment_args(include_odd_tail=include_odd_tail),
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "HF_HUB_DISABLE_TELEMETRY=1",
            str(snapshot["id"]),
            RUNTIME_PYTHON,
            "/workspace/tools/kernel_gates/code2wav_window_graph_smoke.py",
            "--preflight", "--workspace", "/workspace",
            "--no-model-load",
            "--wav", f"/workspace/{wav_relative}",
            *( ["--code2wav-graph-odd-tail"] if include_odd_tail else [] ),
        ]
        preflight = subprocess.run(
            preflight_argv, cwd=repo_root, check=False, text=True, capture_output=True
        )
        if preflight.stdout:
            print(preflight.stdout, end="")
        if preflight.returncode != 0:
            raise RuntimeError(
                "offline CPU preflight in managed runtime failed; service was not paused: "
                f"{preflight.stderr[-3000:]}"
            )
        try:
            preflight_result = _extract_cpu_preflight_json(preflight.stdout)
        except RuntimeError as exc:
            raise RuntimeError(
                "CPU preflight did not emit a recognized JSON result; service was not paused: "
                f"{preflight.stdout[-1000:]!r}"
            ) from exc
        if preflight_result.get("passed") is not True:
            raise RuntimeError(
                "offline CPU preflight result was not passed; service was not paused: "
                f"{preflight_result.get('errors')!r}"
            )
        graph_config = preflight_result.get("production_graph_config")
        if not isinstance(graph_config, Mapping) or graph_config.get("passed") is not True:
            raise RuntimeError(
                "offline CPU preflight did not attest the requested production graph configuration; "
                f"service was not paused: {graph_config!r}"
            )
        if preflight_result.get("model_loaded") is not False:
            raise RuntimeError(
                "offline CPU preflight loaded Code2Wav; service was not paused"
            )
        cached_snapshot = preflight_result.get("cached_snapshot")
        code2wav_snapshot_manifest = (
            cached_snapshot.get("asset_manifest")
            if isinstance(cached_snapshot, Mapping)
            else None
        )
        if not isinstance(code2wav_snapshot_manifest, Mapping):
            raise RuntimeError(
                "CPU preflight did not provide the exact Code2Wav config/weight hash manifest; "
                "service was not paused"
            )
        preflight_fixture = preflight_result.get("fixture")
        if not isinstance(preflight_fixture, Mapping):
            raise RuntimeError(
                "CPU preflight did not provide the recorded-WAV fixture hash; service was not paused"
            )
        expected_preflight_fixture_path = f"/workspace/{wav_relative}"
        if preflight_fixture.get("path") != expected_preflight_fixture_path:
            raise RuntimeError(
                "CPU preflight returned a different recorded-WAV fixture path; "
                f"expected={expected_preflight_fixture_path!r} "
                f"observed={preflight_fixture.get('path')!r}; service was not paused"
            )
        fixture_manifest = _fixture_source_manifest(
            repo_root, wav_path, preflight_fixture
        )

        manifest_path = output_path.with_suffix(output_path.suffix + ".sources.json")
        source_manifest = _write_source_manifest(
            repo_root, manifest_path, code2wav_snapshot_manifest, fixture_manifest
        )
        print(json.dumps({"source_manifest": source_manifest}, ensure_ascii=False), flush=True)

        before = runner.pause_maintenance_container(snapshot, repo_root)
        snapshot["gpu_memory_before_stop"] = before
        release = runner.wait_for_gpu_vram_release(snapshot, repo_root)
        print(json.dumps({"gpu_release_barrier": release}, ensure_ascii=False), flush=True)

        inspect = snapshot.get("inspect")
        image_id = str(inspect.get("Image") or "") if isinstance(inspect, Mapping) else ""
        if not image_id:
            raise RuntimeError("immutable maintenance inspect did not include the service image ID")
        docker_argv = [
            "docker", "run", "--name", worker_name,
            "--label", f"{runner.OWNERSHIP_LABEL}={runner.OWNERSHIP_VALUE}",
            "--user", "1000:1000",
            "--ipc", "host",
            "--gpus", "all",
            "--network", "none",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "HF_HUB_DISABLE_TELEMETRY=1",
            *_production_graph_environment_args(include_odd_tail=include_odd_tail),
            "--volume", f"{str(repo_root).replace(chr(92), '/') }:/workspace",
            "--workdir", "/workspace",
            "--pull=never",
            "--entrypoint", RUNTIME_PYTHON,
            image_id,
            "/workspace/tools/kernel_gates/code2wav_window_graph_smoke.py",
            "--worker", "--workspace", "/workspace",
            "--wav", f"/workspace/{wav_relative}",
            "--expected-source-manifest",
            f"/workspace/{_relative_workspace_path(repo_root, manifest_path)}",
            "--output", f"/workspace/{output_relative}",
            "--max-capture-reserved-delta-bytes",
            str(max_capture_reserved_delta_bytes),
            *( ["--code2wav-graph-odd-tail"] if include_odd_tail else [] ),
        ]
        worker_result = subprocess.run(
            docker_argv,
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if worker_result.stdout:
            print(worker_result.stdout, end="")
        if worker_result.stderr:
            print(worker_result.stderr[-5000:], file=sys.stderr, end="")
        if worker_result.returncode != 0:
            raise RuntimeError(f"Code2Wav GPU worker exited with {worker_result.returncode}")
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            runner.cleanup_owned_container(None, worker_name, repo_root)
        except Exception as exc:
            cleanup_errors.append(f"owned GPU worker cleanup failed: {type(exc).__name__}: {exc}")
        if snapshot and snapshot.get("was_running"):
            try:
                runner.restore_maintenance_container(snapshot, timeout_s=300.0)
                print("managed container restored and health-checked at HTTP 200", flush=True)
            except Exception as exc:
                cleanup_errors.append(
                    f"managed service restoration/health check failed: {type(exc).__name__}: {exc}"
                )

    if primary_error is not None:
        detail = "; ".join(cleanup_errors)
        if detail:
            raise RuntimeError(f"{primary_error}; cleanup: {detail}") from primary_error
        raise primary_error
    if cleanup_errors:
        raise RuntimeError("; ".join(cleanup_errors))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="offline CPU import/path/shape check (default)")
    modes.add_argument("--contract-only", action="store_true", help="CPU graph-shape contract check without Qwen imports")
    modes.add_argument("--run-cuda", action="store_true", help="explicitly pause the managed service and run the GPU worker")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-model-load",
        action="store_true",
        help="hash the offline model snapshot and WAV without loading model weights",
    )
    parser.add_argument("--workspace", type=Path, default=_workspace_default())
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV_RELATIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-manifest", type=Path)
    parser.add_argument("--maintenance-container", default=DEFAULT_MAINTENANCE_CONTAINER)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--max-capture-reserved-delta-bytes",
        type=int,
        default=DEFAULT_MAX_CAPTURE_RESERVED_DELTA_BYTES,
        help=(
            "fail closed if graph capture increases reserved CUDA memory above this byte budget "
            "(default: 3000000000, about 3 GB)"
        ),
    )
    parser.add_argument(
        "--code2wav-graph-odd-tail",
        action="store_true",
        help="opt in to the 74 exact windows 1..74 (including one-frame EOS flush)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    wav_path = _inside_workspace(workspace, args.wav)

    if args.worker:
        if not Path("/workspace").is_dir():
            parser.error("--worker is reserved for the mounted Docker worker")
        if args.expected_source_manifest is None:
            parser.error("--worker requires --expected-source-manifest")
        if args.max_capture_reserved_delta_bytes <= 0:
            parser.error("--max-capture-reserved-delta-bytes must be positive")
        output = args.output or Path("artifacts") / (
            "code2wav_window_graph_smoke_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        output_path = _inside_workspace(workspace, output)
        try:
            result = _run_cuda_worker(
                workspace,
                wav_path,
                output_path,
                args.expected_source_manifest,
                args.max_capture_reserved_delta_bytes,
                args.code2wav_graph_odd_tail,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 0 if result.get("passed") is True else 2
        except Exception as exc:
            failure = {
                "schema": "code2wav_window_graph_gpu_smoke_v1",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output_path)
            print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
            return 1

    if args.run_cuda:
        if args.timeout_s <= 0:
            parser.error("--timeout-s must be positive")
        if args.max_capture_reserved_delta_bytes <= 0:
            parser.error("--max-capture-reserved-delta-bytes must be positive")
        output = args.output or Path("artifacts") / (
            "code2wav_window_graph_smoke_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        return _host_cuda_run(
            workspace,
            wav_path,
            _inside_workspace(workspace, output),
            container_name=args.maintenance_container,
            timeout_s=args.timeout_s,
            max_capture_reserved_delta_bytes=args.max_capture_reserved_delta_bytes,
            include_odd_tail=args.code2wav_graph_odd_tail,
        )

    result = _cpu_preflight(
        workspace,
        wav_path,
        contract_only=args.contract_only,
        no_model_load=args.no_model_load,
        include_odd_tail=args.code2wav_graph_odd_tail,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
