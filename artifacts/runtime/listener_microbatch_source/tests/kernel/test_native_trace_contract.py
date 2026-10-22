from __future__ import annotations

import copy
import hashlib
import json
import struct

import pytest

from duplex_kernel.native_trace import NativeTraceContractError, validate_native_trace_contract


HASH = "a" * 64
TOPOLOGY_HASH = "b" * 64
TRACE_SCHEMA_VERSION = 2
TRACE_ABI_VERSION = 3


def _digest(*, shape: list[int], dtype: str = "torch.bfloat16") -> dict[str, object]:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return {
        "shape": shape,
        "dtype": dtype,
        "numel": numel,
        "finite": True,
        "nonzero": 1,
        "sha256": HASH,
    }


def _record(step_index: int, frame_id: int) -> dict[str, object]:
    return {
        "step_index": step_index,
        "trace_generation": 1,
        "expected_step_count": 2,
        "completed_step_count": step_index + 1,
        "sequence_id": 0,
        "frame_id": frame_id,
        "stable_length": 4,
        "prefix_start": 4,
        "audio_start": 8,
        "prefix_position": 4 + frame_id,
        "audio_position": 8 + frame_id,
        "prefix_embed": _digest(shape=[1, 1, 2048]),
        "audio_embed": _digest(shape=[1, 1, 2048]),
        "effective_mask": _digest(shape=[2, 16]),
        "position_ids": _digest(shape=[2], dtype="torch.int64"),
        "position_id_values": [4 + frame_id, 8 + frame_id],
        "output_hidden": _digest(shape=[1, 1, 2048]),
        "output_hidden_source": "native_abi_debug_copy",
        "output_hidden_consumed": {
            **_digest(shape=[1, 1, 2048]),
            "source": "native_runtime_consumer",
        },
        "position_ids_source": "native_abi_debug_copy",
        "cache_write": {
            "layers": 28,
            "prefix_position": 4 + frame_id,
            "audio_position": 8 + frame_id,
            "source": "native_abi_debug_copy",
            "key": _digest(shape=[28, 2, 8, 128]),
            "value": _digest(shape=[28, 2, 8, 128]),
        },
        "cuda_error_state": 0,
        "joint_steps": step_index + 1,
        "cache_imports": 1,
    }


def _with_chain(records: list[dict[str, object]]) -> str:
    previous = "0" * 64
    native_previous = "0" * 64
    for index, record in enumerate(records):
        record["native_previous_chain_sha256"] = native_previous
        native_payload = struct.pack(
            "<QQQIIIIIIiQQ",
            record["trace_generation"], record["step_index"], record["expected_step_count"],
            record["stable_length"], record["prefix_start"], record["audio_start"],
            record["prefix_position"], record["audio_position"], record["cache_write"]["layers"],
            record["cuda_error_state"], record["joint_steps"], record["cache_imports"],
        )
        native_current = hashlib.sha256(
            bytes.fromhex(native_previous) + b"qwen-native-step-chain-v1" + native_payload
        ).hexdigest()
        record["native_chain_sha256"] = native_current
        native_previous = native_current
        record["trace_segment_index"] = 0
        record["trace_segment_step_index"] = index
    for record in records:
        record["previous_step_sha256"] = previous
        canonical = {
            key: value
            for key, value in record.items()
            if key not in {"previous_step_sha256", "step_sha256"}
        }
        current = hashlib.sha256(
            (previous + "|" + json.dumps(canonical, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()
        record["step_sha256"] = current
        previous = current
    return previous


def _trace(*, records: list[dict[str, object]] | None = None) -> dict[str, object]:
    records = records if records is not None else [_record(0, 0), _record(1, 1)]
    head = _with_chain(records)
    native_head = str(records[-1]["native_chain_sha256"])
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "backend": "native",
        "attestation_source": "native_abi_debug_copy",
        "abi_version": TRACE_ABI_VERSION,
        "library_sha256": "c" * 64,
        "topology_sha256": TOPOLOGY_HASH,
        "hash_algorithm": "sha256:tensor-v1",
        "mask_capacity": 16,
        "trace_chain_algorithm": "sha256:native-step-chain-v1",
        "native_trace_chain_algorithm": "sha256:native-abi-step-chain-v1",
        "trace_complete": True,
        "expected_step_count": len(records),
        "native_completion": {
            "source": "native_abi",
            "expected_step_count": len(records),
            "completed_step_count": len(records),
            "last_step_index": len(records) - 1,
            "trace_generation": 1,
            "complete": True,
            "cache_imports": 1,
            "cuda_error_state": 0,
            "native_chain_sha256": native_head,
            "scope": "segments",
            "segment_count": 1,
            "segments": [{
                "segment_index": 0,
                "expected_step_count": len(records),
                "completed_step_count": len(records),
                "last_step_index": len(records) - 1,
                "trace_generation": 1,
                "complete": True,
                "cache_imports": 1,
                "cuda_error_state": 0,
                "native_chain_sha256": native_head,
            }],
        },
        "trace_chain_head_sha256": head,
        "steps": records,
    }


def test_native_trace_contract_accepts_complete_native_records() -> None:
    validated = validate_native_trace_contract(
        _trace(),
        expected_step_pairs=((0, 0), (0, 1)),
        expected_abi_version=TRACE_ABI_VERSION,
        expected_topology_sha256=TOPOLOGY_HASH,
    )

    assert validated["trace_complete"] is True
    assert len(validated["steps"]) == 2


def test_native_trace_contract_rejects_missing_effective_mask() -> None:
    trace = _trace()
    del trace["steps"][0]["effective_mask"]  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="effective_mask"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_duplicate_or_missing_steps() -> None:
    records = [_record(0, 0), _record(1, 0)]

    with pytest.raises(NativeTraceContractError, match="duplicate|coverage"):
        validate_native_trace_contract(
            _trace(records=records),
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_python_observer_attestation() -> None:
    trace = _trace()
    trace["attestation_source"] = "python_observer"

    with pytest.raises(NativeTraceContractError, match="attestation_source"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_effective_mask_digest_mutation() -> None:
    trace = _trace()
    expected = {(0, 0): HASH, (0, 1): HASH}
    trace["steps"][1]["effective_mask"]["sha256"] = "d" * 64  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="effective_mask.*sha256"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
            expected_effective_mask_digests=expected,
        )


def test_native_trace_contract_rejects_missing_native_completion_evidence() -> None:
    trace = _trace()
    del trace["native_completion"]

    with pytest.raises(NativeTraceContractError, match="native_completion"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_accepts_one_audio_only_terminal_query() -> None:
    trace = _trace()
    trace["native_completion"]["terminal_query"] = {
        "source": "native_abi",
        "count": 1,
        "query_length": 1,
        "visible_prefix_position": 5,
        "audio_position": 10,
        "prefix_cache_appends": 0,
        "audio_cache_appends": 1,
    }

    validated = validate_native_trace_contract(
        trace,
        expected_step_pairs=((0, 0), (0, 1)),
        expected_abi_version=TRACE_ABI_VERSION,
        expected_topology_sha256=TOPOLOGY_HASH,
    )

    assert validated["native_completion"]["terminal_query"]["query_length"] == 1


def test_native_trace_contract_rejects_terminal_prefix_append_or_wrong_position() -> None:
    trace = _trace()
    trace["native_completion"]["terminal_query"] = {
        "source": "native_abi",
        "count": 1,
        "query_length": 1,
        "visible_prefix_position": 6,
        "audio_position": 11,
        "prefix_cache_appends": 1,
        "audio_cache_appends": 1,
    }

    with pytest.raises(NativeTraceContractError, match="terminal_query"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_step_chain_mutation() -> None:
    trace = _trace()
    trace["steps"][1]["cuda_error_state"] = 7  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="cuda_error_state|chain"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_nonzero_cuda_error() -> None:
    trace = _trace()
    trace["native_completion"]["cuda_error_state"] = 7  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="cuda_error_state"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_external_position_mismatch() -> None:
    trace = _trace()
    trace["steps"][1]["prefix_position"] = 99  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="posizioni"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_native_chain_mutation() -> None:
    trace = _trace()
    trace["steps"][1]["native_chain_sha256"] = "d" * 64  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="native chain"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )


def test_native_trace_contract_rejects_native_count_shortfall() -> None:
    trace = _trace()
    trace["native_completion"]["completed_step_count"] = 1  # type: ignore[index]

    with pytest.raises(NativeTraceContractError, match="completed_step_count"):
        validate_native_trace_contract(
            trace,
            expected_step_pairs=((0, 0), (0, 1)),
            expected_abi_version=TRACE_ABI_VERSION,
            expected_topology_sha256=TOPOLOGY_HASH,
        )
