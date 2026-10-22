from __future__ import annotations

import json

from native.export_qwen_native_topology import (
    build_topology,
    render_generated_header,
    write_artifacts,
)
from duplex_kernel.native_qwen_talker import (
    _compiled_qwen_contract,
    _validate_qwen_runtime_config,
)
from types import SimpleNamespace
import pytest


def _sample_topology() -> dict:
    return build_topology(
        config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "max_position_embeddings": 128,
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_parameters": {
                "rope_type": "default",
                "interleaved": True,
                "mrope_section": [2, 1, 1],
            },
        },
        layer_descriptors=[
            {"layer_index": 0, "weights": []},
            {"layer_index": 1, "weights": []},
        ],
        global_descriptors=[],
        codec_head={"out_features": 7, "in_features": 16, "dtype": "torch.bfloat16"},
        source_snapshot="/snapshot",
        git_sha="test-sha",
    )


def test_topology_serialization_has_required_runtime_contract() -> None:
    topology = _sample_topology()

    assert topology["schema"] == "QWEN_NATIVE_TOPOLOGY_V1"
    assert topology["decoder"]["num_hidden_layers"] == 2
    assert topology["decoder"]["hidden_size"] == 16
    assert topology["decoder"]["num_attention_heads"] == 4
    assert topology["decoder"]["num_key_value_heads"] == 2
    assert topology["decoder"]["head_dim"] == 4
    assert topology["decoder"]["rope"]["mrope_section"] == [2, 1, 1]
    assert topology["codec_head"]["out_features"] == 7
    json.dumps(topology, sort_keys=True)


def test_generated_header_is_derived_from_topology() -> None:
    header = render_generated_header(_sample_topology())

    assert "#define QWEN_NATIVE_NUM_LAYERS 2u" in header
    assert "#define QWEN_NATIVE_HIDDEN_SIZE 16u" in header
    assert "#define QWEN_NATIVE_INTERMEDIATE_SIZE 32u" in header
    assert "#define QWEN_NATIVE_NUM_ATTN_HEADS 4u" in header
    assert "#define QWEN_NATIVE_NUM_KV_HEADS 2u" in header
    assert "#define QWEN_NATIVE_HEAD_DIM 4u" in header
    assert "#define QWEN_NATIVE_MROPE_SECTION_0 2u" in header
    assert "#define QWEN_NATIVE_CODEC_VOCAB_SIZE 7u" in header
    assert "#define QWEN_NATIVE_CODEC_HEAD_DIM QWEN_NATIVE_CODEC_VOCAB_SIZE" in header
    assert "QWEN_NATIVE_ROPE_INTERLEAVED 1" in header


def test_topology_attestation_ignores_process_specific_pointers() -> None:
    first = _sample_topology()
    second = _sample_topology()
    first["global_descriptors"] = [{"name": "x", "shape": [1], "data_ptr": 11, "storage_ptr": 11, "device": "cuda:0"}]
    second["global_descriptors"] = [{"name": "x", "shape": [1], "data_ptr": 99, "storage_ptr": 99, "device": "cuda:1"}]

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first_written = write_artifacts(first, root / "a.json", root / "a.h")
        second_written = write_artifacts(second, root / "b.json", root / "b.h")
    assert first_written["topology_sha256"] == second_written["topology_sha256"]


def test_write_artifacts_attests_json_and_header_from_same_payload(tmp_path) -> None:
    topology = _sample_topology()
    output = tmp_path / "topology.json"
    header_path = tmp_path / "generated.h"

    materialized = write_artifacts(topology, output, header_path)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert materialized["topology_sha256"] == loaded["topology_sha256"]
    assert materialized["topology_sha256"] in header_path.read_text(encoding="utf-8")


def test_native_runtime_contract_comes_from_compiled_header_without_json_manifest() -> None:
    contract = _compiled_qwen_contract()
    config = SimpleNamespace(
        num_hidden_layers=28,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        vocab_size=3072,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_scaling={"interleaved": True, "mrope_section": [24, 20, 20]},
    )

    values = _validate_qwen_runtime_config(config, contract)

    assert values["num_hidden_layers"] == 28
    assert values["hidden_size"] == 2048
    assert values["head_dim"] == 128


def test_native_runtime_contract_rejects_config_that_differs_from_compiled_header() -> None:
    contract = _compiled_qwen_contract()
    config = SimpleNamespace(
        num_hidden_layers=28,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        vocab_size=3072,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_scaling={"interleaved": True, "mrope_section": [24, 20, 20]},
    )

    with pytest.raises(ValueError, match="hidden_act"):
        _validate_qwen_runtime_config(config, contract)
