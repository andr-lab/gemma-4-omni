"""Fail-closed contract and canonical digests for the native Qwen ABI trace.

The trace is populated from buffers copied by the native ABI.  Python-side
input observers are deliberately not accepted as native attestation.
"""
from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import torch

NATIVE_TRACE_SCHEMA_VERSION = 2
NATIVE_TRACE_ABI_VERSION = 3
NATIVE_TRACE_ATTESTATION_SOURCE = "native_abi_debug_copy"
NATIVE_TRACE_HASH_ALGORITHM = "sha256:tensor-v1"
NATIVE_TRACE_CHAIN_ALGORITHM = "sha256:native-step-chain-v1"
NATIVE_TRACE_NATIVE_CHAIN_ALGORITHM = "sha256:native-abi-step-chain-v1"
NATIVE_TRACE_NATIVE_SOURCE = "native_abi"
_TRACE_CHAIN_ZERO = "0" * 64


class NativeTraceContractError(ValueError):
    """Raised when native trace evidence is missing, malformed, or inconsistent."""


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and exact contiguous bytes deterministically."""
    contiguous = tensor.detach().to(device="cpu").contiguous()
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(
        f"{tuple(contiguous.shape)}|{contiguous.dtype}|".encode("utf-8") + payload
    ).hexdigest()


def _is_hash(value: Any, *, length: int = 64) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_hash(value: Any, name: str, *, length: int = 64) -> str:
    if not _is_hash(value, length=length):
        raise NativeTraceContractError(f"{name} non è uno SHA-hex valido")
    return str(value).lower()


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise NativeTraceContractError(f"{name} deve essere un intero valido")
    return int(value)


def _canonical_step_payload(step: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in step.items()
        if key not in {"previous_step_sha256", "step_sha256"}
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_trace_chain(steps: Sequence[Mapping[str, Any]]) -> str:
    """Materialize the append-only report chain and return its head digest."""
    previous = _TRACE_CHAIN_ZERO
    for step in steps:
        if not isinstance(step, dict):
            raise NativeTraceContractError("native trace chain step non object")
        step["previous_step_sha256"] = previous
        step["step_sha256"] = hashlib.sha256(
            (previous + "|" + _canonical_step_payload(step)).encode("utf-8")
        ).hexdigest()
        previous = step["step_sha256"]
    return previous


def _native_chain_digest(previous: str, step: Mapping[str, Any]) -> str:
    payload = struct.pack(
        "<QQQIIIIIIiQQ",
        int(step["trace_generation"]),
        int(step.get("trace_segment_step_index", step["step_index"])),
        int(step["expected_step_count"]),
        int(step["stable_length"]),
        int(step["prefix_start"]),
        int(step["audio_start"]),
        int(step["prefix_position"]),
        int(step["audio_position"]),
        int(step["cache_write"]["layers"]),
        int(step["cuda_error_state"]),
        int(step["joint_steps"]),
        int(step["cache_imports"]),
    )
    return hashlib.sha256(
        bytes.fromhex(previous)
        + b"qwen-native-step-chain-v1"
        + payload
    ).hexdigest()


def _validate_digest(
    value: Any,
    name: str,
    *,
    allowed_dtypes: set[str],
    expected_shape: tuple[int, ...] | None = None,
    require_nonzero: bool = False,
) -> None:
    if not isinstance(value, Mapping):
        raise NativeTraceContractError(f"{name} mancante/non object")
    shape = value.get("shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise NativeTraceContractError(f"{name}.shape invalida")
    if expected_shape is not None and tuple(shape) != expected_shape:
        raise NativeTraceContractError(f"{name}.shape inattesa: {tuple(shape)!r} != {expected_shape!r}")
    expected_numel = 1
    for dimension in shape:
        expected_numel *= dimension
    if type(value.get("numel")) is not int or value["numel"] != expected_numel:
        raise NativeTraceContractError(f"{name}.numel invalido")
    if type(value.get("finite")) is not bool or not value["finite"]:
        raise NativeTraceContractError(f"{name}.finite non verificato")
    if type(value.get("nonzero")) is not int or not 0 <= value["nonzero"] <= expected_numel:
        raise NativeTraceContractError(f"{name}.nonzero invalido")
    if require_nonzero and value["nonzero"] == 0:
        raise NativeTraceContractError(f"{name} completamente nullo")
    dtype = value.get("dtype")
    if dtype not in allowed_dtypes:
        raise NativeTraceContractError(f"{name}.dtype invalido")
    _require_hash(value.get("sha256"), f"{name}.sha256")


def _validate_step(
    value: Any,
    *,
    expected_index: int,
    expected_pair: tuple[int, int],
    expected_mask_capacity: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeTraceContractError(f"steps[{expected_index}] non è un object")
    required = {
        "step_index",
        "trace_generation",
        "expected_step_count",
        "completed_step_count",
        "sequence_id",
        "frame_id",
        "stable_length",
        "prefix_start",
        "audio_start",
        "prefix_position",
        "audio_position",
        "prefix_embed",
        "audio_embed",
        "effective_mask",
        "position_ids",
        "position_id_values",
        "output_hidden",
        "output_hidden_source",
        "output_hidden_consumed",
        "position_ids_source",
        "cache_write",
        "cuda_error_state",
        "joint_steps",
        "cache_imports",
        "native_previous_chain_sha256",
        "native_chain_sha256",
        "previous_step_sha256",
        "step_sha256",
    }
    missing = sorted(required - set(value))
    if missing:
        raise NativeTraceContractError(
            f"steps[{expected_index}] campi mancanti: {missing}"
        )
    if _require_int(value["step_index"], f"steps[{expected_index}].step_index") != expected_index:
        raise NativeTraceContractError(f"steps[{expected_index}].step_index non contiguo")
    _require_int(value["trace_generation"], f"steps[{expected_index}].trace_generation", minimum=1)
    pair = (
        _require_int(value["sequence_id"], f"steps[{expected_index}].sequence_id", minimum=0),
        _require_int(value["frame_id"], f"steps[{expected_index}].frame_id", minimum=0),
    )
    if pair != expected_pair:
        raise NativeTraceContractError(
            f"steps[{expected_index}] coverage mismatch: {pair!r} != {expected_pair!r}"
        )

    stable_length = _require_int(value["stable_length"], "stable_length", minimum=1)
    prefix_start = _require_int(value["prefix_start"], "prefix_start", minimum=1)
    audio_start = _require_int(value["audio_start"], "audio_start", minimum=1)
    prefix_position = _require_int(value["prefix_position"], "prefix_position", minimum=0)
    audio_position = _require_int(value["audio_position"], "audio_position", minimum=0)
    if prefix_start != stable_length or audio_start <= prefix_start:
        raise NativeTraceContractError(f"steps[{expected_index}] cache layout incoerente")
    frame_id = pair[1]
    if (
        prefix_position != prefix_start + frame_id
        or audio_position != audio_start + frame_id
        or not prefix_start <= prefix_position < audio_start
        or audio_position < audio_start
    ):
        raise NativeTraceContractError(f"steps[{expected_index}] posizioni incoerenti con frame esterno")

    _validate_digest(
        value["prefix_embed"],
        f"steps[{expected_index}].prefix_embed",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(1, 1, 2048),
    )
    _validate_digest(
        value["audio_embed"],
        f"steps[{expected_index}].audio_embed",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(1, 1, 2048),
    )
    _validate_digest(
        value["effective_mask"],
        f"steps[{expected_index}].effective_mask",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(2, expected_mask_capacity),
    )
    _validate_digest(
        value["position_ids"],
        f"steps[{expected_index}].position_ids",
        allowed_dtypes={"torch.int64"},
        expected_shape=(2,),
    )
    position_id_values = value["position_id_values"]
    if position_id_values != [prefix_position, audio_position]:
        raise NativeTraceContractError(f"steps[{expected_index}] position ID values mismatch")
    _validate_digest(
        value["output_hidden"],
        f"steps[{expected_index}].output_hidden",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(1, 1, 2048),
        require_nonzero=True,
    )
    if value["output_hidden_source"] != NATIVE_TRACE_ATTESTATION_SOURCE:
        raise NativeTraceContractError(f"steps[{expected_index}].output_hidden_source non nativa")
    consumed_output = value["output_hidden_consumed"]
    if not isinstance(consumed_output, Mapping) or consumed_output.get("source") != "native_runtime_consumer":
        raise NativeTraceContractError(f"steps[{expected_index}].output_hidden_consumed source invalida")
    _validate_digest(
        consumed_output,
        f"steps[{expected_index}].output_hidden_consumed",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(1, 1, 2048),
        require_nonzero=True,
    )
    if consumed_output["sha256"] != value["output_hidden"]["sha256"]:
        raise NativeTraceContractError(f"steps[{expected_index}] output non coincide con il consumer")
    if value["position_ids_source"] != NATIVE_TRACE_ATTESTATION_SOURCE:
        raise NativeTraceContractError(f"steps[{expected_index}].position_ids_source non nativa")
    cache_write = value["cache_write"]
    if not isinstance(cache_write, Mapping):
        raise NativeTraceContractError(f"steps[{expected_index}].cache_write invalido")
    if _require_int(cache_write.get("layers"), "cache_write.layers", minimum=1) != 28:
        raise NativeTraceContractError(f"steps[{expected_index}].cache_write.layers invalido")
    if cache_write.get("prefix_position") != prefix_position or cache_write.get("audio_position") != audio_position:
        raise NativeTraceContractError(f"steps[{expected_index}] cache write position mismatch")
    if cache_write.get("source") != NATIVE_TRACE_ATTESTATION_SOURCE:
        raise NativeTraceContractError(f"steps[{expected_index}].cache_write.source non nativa")
    _validate_digest(
        cache_write.get("key"),
        f"steps[{expected_index}].cache_write.key",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(28, 2, 8, 128),
        require_nonzero=True,
    )
    _validate_digest(
        cache_write.get("value"),
        f"steps[{expected_index}].cache_write.value",
        allowed_dtypes={"torch.bfloat16"},
        expected_shape=(28, 2, 8, 128),
        require_nonzero=True,
    )
    _require_hash(value["native_previous_chain_sha256"], f"steps[{expected_index}].native_previous_chain_sha256")
    _require_hash(value["native_chain_sha256"], f"steps[{expected_index}].native_chain_sha256")
    _require_hash(value["previous_step_sha256"], f"steps[{expected_index}].previous_step_sha256")
    _require_hash(value["step_sha256"], f"steps[{expected_index}].step_sha256")
    _require_int(value["joint_steps"], f"steps[{expected_index}].joint_steps", minimum=1)
    _require_int(value["cache_imports"], f"steps[{expected_index}].cache_imports", minimum=1)
    if _require_int(value["cuda_error_state"], f"steps[{expected_index}].cuda_error_state") != 0:
        raise NativeTraceContractError(f"steps[{expected_index}] cuda_error_state non-zero")
    return dict(value)


def validate_native_trace_contract(
    trace: Mapping[str, Any],
    *,
    expected_step_pairs: Sequence[tuple[int, int]],
    expected_abi_version: int,
    expected_topology_sha256: str,
    expected_effective_mask_digests: Mapping[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Validate a complete native-produced trace and return a materialized copy."""
    if not isinstance(trace, Mapping):
        raise NativeTraceContractError("native trace non object")
    required = {
        "schema_version",
        "backend",
        "attestation_source",
        "abi_version",
        "library_sha256",
        "topology_sha256",
        "hash_algorithm",
        "mask_capacity",
        "trace_chain_algorithm",
        "native_trace_chain_algorithm",
        "trace_complete",
        "expected_step_count",
        "native_completion",
        "trace_chain_head_sha256",
        "steps",
    }
    missing = sorted(required - set(trace))
    if missing:
        raise NativeTraceContractError(f"native trace campi mancanti: {missing}")
    if trace["schema_version"] != NATIVE_TRACE_SCHEMA_VERSION:
        raise NativeTraceContractError("native trace schema_version invalido")
    if trace["backend"] != "native":
        raise NativeTraceContractError("native trace backend invalido")
    if trace["attestation_source"] != NATIVE_TRACE_ATTESTATION_SOURCE:
        raise NativeTraceContractError("native trace attestation_source non verificata")
    if trace["abi_version"] != expected_abi_version:
        raise NativeTraceContractError("native trace ABI version mismatch")
    _require_hash(trace["library_sha256"], "native trace library_sha256")
    if _require_hash(trace["topology_sha256"], "native trace topology_sha256") != expected_topology_sha256.lower():
        raise NativeTraceContractError("native trace topology SHA mismatch")
    if trace["hash_algorithm"] != NATIVE_TRACE_HASH_ALGORITHM:
        raise NativeTraceContractError("native trace hash_algorithm invalido")
    mask_capacity = _require_int(trace["mask_capacity"], "native trace mask_capacity", minimum=1)
    if trace["trace_chain_algorithm"] != NATIVE_TRACE_CHAIN_ALGORITHM:
        raise NativeTraceContractError("native trace trace_chain_algorithm invalido")
    if trace["native_trace_chain_algorithm"] != NATIVE_TRACE_NATIVE_CHAIN_ALGORITHM:
        raise NativeTraceContractError("native trace native_trace_chain_algorithm invalido")
    if trace["trace_complete"] is not True:
        raise NativeTraceContractError("native trace incompleto")
    completion = trace["native_completion"]
    if not isinstance(completion, Mapping):
        raise NativeTraceContractError("native_completion mancante/non object")
    if completion.get("source") != NATIVE_TRACE_NATIVE_SOURCE:
        raise NativeTraceContractError("native_completion.source non nativa")
    completion_expected = _require_int(
        completion.get("expected_step_count"),
        "native_completion.expected_step_count",
        minimum=0,
    )
    completion_completed = _require_int(
        completion.get("completed_step_count"),
        "native_completion.completed_step_count",
        minimum=0,
    )
    completion_last = _require_int(
        completion.get("last_step_index"),
        "native_completion.last_step_index",
        minimum=0,
    )
    segmented_completion = completion.get("scope") == "segments"
    completion_generation = _require_int(
        completion.get("trace_generation"),
        "native_completion.trace_generation",
        minimum=1,
    ) if not segmented_completion else 0
    _require_int(completion.get("cache_imports"), "native_completion.cache_imports", minimum=1)
    if _require_int(completion.get("cuda_error_state"), "native_completion.cuda_error_state") != 0:
        raise NativeTraceContractError("native_completion.cuda_error_state non-zero")
    if not segmented_completion:
        _require_hash(completion.get("native_chain_sha256"), "native_completion.native_chain_sha256")
    if completion.get("complete") is not True:
        raise NativeTraceContractError("native_completion.complete non verificato")
    terminal_query = completion.get("terminal_query")
    if terminal_query is not None:
        if not isinstance(terminal_query, Mapping) or terminal_query.get("source") != NATIVE_TRACE_NATIVE_SOURCE:
            raise NativeTraceContractError("native_completion.terminal_query non attestata da native ABI")
        if _require_int(terminal_query.get("count"), "terminal_query.count", minimum=0) != 1:
            raise NativeTraceContractError("native_completion.terminal_query count deve essere uno")
        if _require_int(terminal_query.get("query_length"), "terminal_query.query_length", minimum=0) != 1:
            raise NativeTraceContractError("native_completion.terminal_query deve usare Q=1")
        visible_prefix_position = _require_int(
            terminal_query.get("visible_prefix_position"),
            "terminal_query.visible_prefix_position",
            minimum=0,
        )
        terminal_audio_position = _require_int(
            terminal_query.get("audio_position"), "terminal_query.audio_position", minimum=0
        )
        if _require_int(terminal_query.get("prefix_cache_appends"), "terminal_query.prefix_cache_appends", minimum=0) != 0:
            raise NativeTraceContractError("terminal_query non deve appendere un prefix KV")
        if _require_int(terminal_query.get("audio_cache_appends"), "terminal_query.audio_cache_appends", minimum=0) != 1:
            raise NativeTraceContractError("terminal_query deve appendere un solo audio KV")
        if terminal_audio_position >= mask_capacity:
            raise NativeTraceContractError("terminal_query audio_position oltre mask_capacity")
    expected_pairs = [(int(sequence), int(frame)) for sequence, frame in expected_step_pairs]
    expected_count = _require_int(trace["expected_step_count"], "expected_step_count", minimum=0)
    expected_frames_by_sequence: dict[int, list[int]] = {}
    for sequence_id, frame_id in expected_pairs:
        expected_frames_by_sequence.setdefault(sequence_id, []).append(frame_id)
    if expected_count != len(expected_pairs) or completion_expected != expected_count:
        raise NativeTraceContractError("native trace expected_step_count incoerente")
    if completion_completed != expected_count or completion_last != expected_count - 1:
        raise NativeTraceContractError("native_completion.completed_step_count incoerente")
    segment_specs: dict[int, dict[str, Any]] = {}
    if segmented_completion:
        segments = completion.get("segments")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
            raise NativeTraceContractError("native_completion.segments mancante")
        for raw_segment in segments:
            if not isinstance(raw_segment, Mapping):
                raise NativeTraceContractError("native_completion segment non object")
            segment_index = _require_int(raw_segment.get("segment_index"), "native_completion.segment_index", minimum=0)
            if segment_index in segment_specs:
                raise NativeTraceContractError("native_completion segment duplicato")
            segment_specs[segment_index] = dict(raw_segment)
            if _require_int(raw_segment.get("expected_step_count"), "native_completion.segment.expected_step_count", minimum=1) <= 0:
                raise NativeTraceContractError("native_completion segment expected_step_count invalido")
            if _require_int(raw_segment.get("completed_step_count"), "native_completion.segment.completed_step_count", minimum=0) != raw_segment["expected_step_count"]:
                raise NativeTraceContractError("native_completion segment completed_step_count incoerente")
            if _require_int(raw_segment.get("last_step_index"), "native_completion.segment.last_step_index", minimum=0) != raw_segment["expected_step_count"] - 1:
                raise NativeTraceContractError("native_completion segment last_step_index incoerente")
            _require_int(raw_segment.get("trace_generation"), "native_completion.segment.trace_generation", minimum=1)
            if _require_int(raw_segment.get("cache_imports"), "native_completion.segment.cache_imports", minimum=1) != 1:
                raise NativeTraceContractError("native_completion segment cache_imports incoerente")
            if _require_int(raw_segment.get("cuda_error_state"), "native_completion.segment.cuda_error_state") != 0:
                raise NativeTraceContractError("native_completion segment cuda_error_state non-zero")
            _require_hash(raw_segment.get("native_chain_sha256"), "native_completion.segment.native_chain_sha256")
            if raw_segment.get("complete") is not True:
                raise NativeTraceContractError("native_completion segment incompleto")
        if len(segment_specs) != int(completion.get("segment_count", -1)):
            raise NativeTraceContractError("native_completion.segment_count incoerente")
        if len(segment_specs) != len(expected_frames_by_sequence):
            raise NativeTraceContractError("native_completion segment count non allineato alle sequenze esterne")
    steps = trace["steps"]
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise NativeTraceContractError("native trace steps invalido")
    if len(steps) != len(expected_pairs):
        raise NativeTraceContractError(
            f"native trace coverage incompleta: {len(steps)} != {len(expected_pairs)}"
        )
    seen: set[tuple[int, int]] = set()
    validated_steps: list[dict[str, Any]] = []
    previous_chain = _TRACE_CHAIN_ZERO
    previous_native_chains: dict[int, str] = {}
    segment_counts: dict[int, int] = {}
    segment_sequences: dict[int, set[int]] = {}
    expected_masks = expected_effective_mask_digests or {}
    for index, (step, expected_pair) in enumerate(zip(steps, expected_pairs, strict=True)):
        normalized = _validate_step(
            step,
            expected_index=index,
            expected_pair=expected_pair,
            expected_mask_capacity=mask_capacity,
        )
        pair = expected_pair
        if segmented_completion:
            segment_index = _require_int(
                normalized.get("trace_segment_index"),
                f"steps[{index}].trace_segment_index",
                minimum=0,
            )
            segment_spec = segment_specs.get(segment_index)
            if segment_spec is None:
                raise NativeTraceContractError(f"steps[{index}] segment sconosciuto")
            segment_local_index = _require_int(
                normalized.get("trace_segment_step_index"),
                f"steps[{index}].trace_segment_step_index",
                minimum=0,
            )
            segment_counts[segment_index] = segment_counts.get(segment_index, 0) + 1
            segment_sequences.setdefault(segment_index, set()).add(pair[0])
            if normalized["trace_generation"] != segment_spec["trace_generation"]:
                raise NativeTraceContractError(f"steps[{index}] trace_generation mismatch")
            if normalized["expected_step_count"] != segment_spec["expected_step_count"]:
                raise NativeTraceContractError(f"steps[{index}] expected_step_count mismatch")
            if normalized["completed_step_count"] != segment_local_index + 1:
                raise NativeTraceContractError(f"steps[{index}] completed_step_count non contiguo")
            previous_native_chain = previous_native_chains.get(segment_index, _TRACE_CHAIN_ZERO)
        else:
            segment_index = -1
            segment_local_index = index
            if normalized["trace_generation"] != completion_generation:
                raise NativeTraceContractError(f"steps[{index}] trace_generation mismatch")
            if normalized["expected_step_count"] != expected_count:
                raise NativeTraceContractError(f"steps[{index}] expected_step_count mismatch")
            if normalized["completed_step_count"] != index + 1:
                raise NativeTraceContractError(f"steps[{index}] completed_step_count non contiguo")
            previous_native_chain = previous_native_chains.get(segment_index, _TRACE_CHAIN_ZERO)
        if normalized["native_previous_chain_sha256"] != previous_native_chain:
            raise NativeTraceContractError(f"steps[{index}] native chain previous mismatch")
        expected_native_chain = _native_chain_digest(previous_native_chain, normalized)
        if normalized["native_chain_sha256"] != expected_native_chain:
            raise NativeTraceContractError(f"steps[{index}] native chain digest mismatch")
        previous_native_chains[segment_index] = expected_native_chain
        expected_mask = expected_masks.get(pair)
        if expected_mask is not None:
            observed_mask = normalized["effective_mask"]["sha256"]
            if observed_mask != expected_mask:
                raise NativeTraceContractError(
                    f"steps[{index}].effective_mask.sha256 mismatch"
                )
        if normalized["previous_step_sha256"] != previous_chain:
            raise NativeTraceContractError(f"steps[{index}] hash chain previous mismatch")
        expected_chain = hashlib.sha256(
            (previous_chain + "|" + _canonical_step_payload(normalized)).encode("utf-8")
        ).hexdigest()
        if normalized["step_sha256"] != expected_chain:
            raise NativeTraceContractError(f"steps[{index}] hash chain digest mismatch")
        previous_chain = expected_chain
        if pair in seen:
            raise NativeTraceContractError(f"native trace duplicate step: {pair!r}")
        seen.add(pair)
        validated_steps.append(normalized)
    if seen != set(expected_pairs):
        raise NativeTraceContractError("native trace coverage mismatch")
    if terminal_query is not None:
        if not validated_steps:
            raise NativeTraceContractError("terminal_query senza una traiettoria acustica precedente")
        last_step = validated_steps[-1]
        if (
            terminal_query["visible_prefix_position"] != last_step["prefix_position"]
            or terminal_query["audio_position"] != last_step["audio_position"] + 1
        ):
            raise NativeTraceContractError("terminal_query positions non contigue dopo JOINT Q=2")
    if _require_hash(trace["trace_chain_head_sha256"], "trace_chain_head_sha256") != previous_chain:
        raise NativeTraceContractError("native trace hash chain head mismatch")
    if segmented_completion:
        if set(segment_counts) != set(segment_specs):
            raise NativeTraceContractError("native trace segment coverage mismatch")
        for segment_index, segment_spec in segment_specs.items():
            if segment_counts.get(segment_index, 0) != int(segment_spec["expected_step_count"]):
                raise NativeTraceContractError("native trace segment frame count mismatch")
            if segment_sequences.get(segment_index) is None or len(segment_sequences[segment_index]) != 1:
                raise NativeTraceContractError("native trace segment contiene più sequenze")
            sequence_id = next(iter(segment_sequences[segment_index]))
            if len(expected_frames_by_sequence.get(sequence_id, [])) != int(segment_spec["expected_step_count"]):
                raise NativeTraceContractError("native trace segment non allineato alla sequenza esterna")
            if segment_spec["native_chain_sha256"] != previous_native_chains[segment_index]:
                raise NativeTraceContractError("native_completion segment chain mismatch")
    elif completion.get("native_chain_sha256") != previous_native_chains.get(-1):
        raise NativeTraceContractError("native_completion.native_chain_sha256 mismatch")
    result = dict(trace)
    result["steps"] = validated_steps
    return result


__all__ = [
    "NATIVE_TRACE_ABI_VERSION",
    "NATIVE_TRACE_ATTESTATION_SOURCE",
    "NATIVE_TRACE_HASH_ALGORITHM",
    "NATIVE_TRACE_CHAIN_ALGORITHM",
    "NATIVE_TRACE_NATIVE_CHAIN_ALGORITHM",
    "NATIVE_TRACE_NATIVE_SOURCE",
    "build_trace_chain",
    "NATIVE_TRACE_SCHEMA_VERSION",
    "NativeTraceContractError",
    "tensor_digest",
    "validate_native_trace_contract",
]
