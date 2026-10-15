#!/usr/bin/env python3
"""Export the real Qwen3-TTS Talker topology and its native compile header.

The exporter is deliberately the only source of decoder dimensions used by the
native Qwen executor.  It records runtime parameter descriptors (including
borrowed CUDA pointers) for auditability, while the generated C++ header keeps
only compile-time topology constants.  Pointer values are process-specific and
are therefore never compiled into native code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Config objects are intentionally reduced to public attributes rather
    # than serialized through pickle or implementation-private state.
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    return str(value)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return _json_value(value)


def _parameter_descriptor(name: str, parameter: torch.Tensor) -> dict[str, Any]:
    if not isinstance(parameter, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor")
    if parameter.device.type != "cuda":
        raise ValueError(f"{name}: expected CUDA parameter, got {parameter.device}")
    storage = parameter.untyped_storage()
    return {
        "name": name,
        "shape": [int(value) for value in parameter.shape],
        "stride": [int(value) for value in parameter.stride()],
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "contiguous": bool(parameter.is_contiguous()),
        "requires_grad": bool(parameter.requires_grad),
        "data_ptr": int(parameter.data_ptr()),
        "storage_ptr": int(storage.data_ptr()),
        "storage_offset": int(parameter.storage_offset()),
        "numel": int(parameter.numel()),
    }


def _shared_groups(descriptors: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[str]] = {}
    for descriptor in descriptors:
        key = (int(descriptor["storage_ptr"]), int(descriptor["storage_offset"]))
        groups.setdefault(key, []).append(str(descriptor["name"]))
    return [
        {
            "storage_ptr": storage_ptr,
            "storage_offset": storage_offset,
            "names": names,
        }
        for (storage_ptr, storage_offset), names in sorted(groups.items())
        if len(names) > 1
    ]


def _module_descriptor(name: str, module: Any) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    result: dict[str, Any] = {
        "name": name,
        "type": f"{type(module).__module__}.{type(module).__qualname__}",
        "weight_shape": None if weight is None else [int(v) for v in weight.shape],
        "weight_dtype": None if weight is None else str(weight.dtype),
        "weight_stride": None if weight is None else [int(v) for v in weight.stride()],
        "weight_contiguous": None if weight is None else bool(weight.is_contiguous()),
        "bias": bias is not None,
        "bias_shape": None if bias is None else [int(v) for v in bias.shape],
        "bias_dtype": None if bias is None else str(bias.dtype),
    }
    return result


def build_topology(
    *,
    config: dict[str, Any],
    layer_descriptors: list[dict[str, Any]],
    global_descriptors: list[dict[str, Any]],
    codec_head: dict[str, Any],
    source_snapshot: str,
    git_sha: str,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Build the stable, JSON-serializable topology object.

    This function accepts plain dictionaries so its schema and header mapping
    can be tested without loading a multi-GB checkpoint.
    """
    rope_parameters = config.get("rope_parameters") or config.get("rope_scaling") or {}
    attention_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // attention_heads)
    decoder = {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "num_key_value_groups": attention_heads // kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "hidden_act": str(config["hidden_act"]),
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "attention_bias": bool(config["attention_bias"]),
        "attention_dropout": float(config.get("attention_dropout", 0.0)),
        "rope": {
            "type": rope_parameters.get("rope_type", rope_parameters.get("type", "default")),
            "theta": float(config["rope_theta"]),
            "interleaved": bool(rope_parameters.get("interleaved", False)),
            "mrope_section": [int(v) for v in rope_parameters.get("mrope_section", [])],
            "parameters": _json_value(rope_parameters),
        },
        "attention_scale": float(head_dim ** -0.5),
    }
    all_descriptors = list(global_descriptors)
    for layer in layer_descriptors:
        all_descriptors.extend(layer.get("weights", []))
    return {
        "schema": "QWEN_NATIVE_TOPOLOGY_V1",
        "model_type": model_type,
        "git_sha": git_sha,
        "source_snapshot": source_snapshot,
        "decoder": decoder,
        "layer_descriptors": layer_descriptors,
        "global_descriptors": global_descriptors,
        "shared_storage_groups": _shared_groups(all_descriptors),
        "codec_head": codec_head,
        "weight_dtype": sorted({str(item["dtype"]) for item in all_descriptors}),
    }


def _git_sha(workspace: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def extract_topology(talker: Any, *, source_snapshot: str, git_sha: str) -> dict[str, Any]:
    """Extract topology from a loaded ``TalkerModel`` wrapper."""
    model = talker.talker
    decoder = model.model
    config = decoder.config
    layer_descriptors: list[dict[str, Any]] = []
    all_decoder_descriptors: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(decoder.layers):
        descriptors = []
        for name, parameter in layer.named_parameters():
            descriptor = _parameter_descriptor(f"model.layers.{layer_index}.{name}", parameter)
            descriptors.append(descriptor)
            all_decoder_descriptors.append(descriptor)
        attention = layer.self_attn
        layer_descriptors.append(
            {
                "layer_index": layer_index,
                "type": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "weights": descriptors,
                "operations": {
                    "input_norm": "input_layernorm",
                    "q_proj": _module_descriptor("self_attn.q_proj", attention.q_proj),
                    "k_proj": _module_descriptor("self_attn.k_proj", attention.k_proj),
                    "v_proj": _module_descriptor("self_attn.v_proj", attention.v_proj),
                    "q_norm": _module_descriptor("self_attn.q_norm", attention.q_norm),
                    "k_norm": _module_descriptor("self_attn.k_norm", attention.k_norm),
                    "rope": "model.rotary_emb + apply_multimodal_rotary_pos_emb",
                    "attention": "eager_attention_forward_or_configured_attention",
                    "o_proj": _module_descriptor("self_attn.o_proj", attention.o_proj),
                    "post_attention_norm": "post_attention_layernorm",
                    "mlp_gate": _module_descriptor("mlp.gate_proj", layer.mlp.gate_proj),
                    "mlp_up": _module_descriptor("mlp.up_proj", layer.mlp.up_proj),
                    "activation": str(type(layer.mlp.act_fn).__module__ + "." + type(layer.mlp.act_fn).__qualname__),
                    "mlp_down": _module_descriptor("mlp.down_proj", layer.mlp.down_proj),
                },
            }
        )

    global_descriptors: list[dict[str, Any]] = []
    for name, parameter in decoder.named_parameters():
        if name.startswith("layers."):
            continue
        global_descriptors.append(_parameter_descriptor(f"model.{name}", parameter))
    rotary = decoder.rotary_emb
    global_buffers = []
    for name, tensor in rotary.named_buffers(recurse=True):
        if tensor.device.type == "cuda":
            global_buffers.append(_parameter_descriptor(f"model.rotary_emb.{name}", tensor))
    codec = model.codec_head
    codec_head = {
        "name": "codec_head",
        "type": f"{type(codec).__module__}.{type(codec).__qualname__}",
        "in_features": int(codec.in_features),
        "out_features": int(codec.out_features),
        "bias": codec.bias is not None,
        "weight": _parameter_descriptor("codec_head.weight", codec.weight),
        "bias_descriptor": None if codec.bias is None else _parameter_descriptor("codec_head.bias", codec.bias),
    }
    topology = build_topology(
        config={
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", None),
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
            "hidden_act": config.hidden_act,
            "attention_bias": config.attention_bias,
            "attention_dropout": getattr(config, "attention_dropout", 0.0),
            "rope_parameters": getattr(config, "rope_parameters", None),
        },
        layer_descriptors=layer_descriptors,
        global_descriptors=global_descriptors + global_buffers,
        codec_head=codec_head,
        source_snapshot=source_snapshot,
        git_sha=git_sha,
        model_type=getattr(config, "model_type", None),
    )
    topology["decoder"]["rotary_inv_freq"] = [float(value) for value in rotary.inv_freq.detach().float().cpu().tolist()]
    topology["decoder"]["rotary_inv_freq_dtype"] = str(rotary.inv_freq.dtype)
    topology["decoder"]["rotary_max_seq_len_cached"] = int(rotary.max_seq_len_cached)
    topology["decoder"]["rotary_original_max_seq_len"] = int(rotary.original_max_seq_len)
    topology["global_descriptors"] = global_descriptors + global_buffers
    topology["all_decoder_weight_count"] = len(all_decoder_descriptors)
    topology["all_weight_count"] = len(global_descriptors) + len(global_buffers) + len(all_decoder_descriptors)
    return topology


_VOLATILE_ATTESTATION_FIELDS = {"data_ptr", "storage_ptr", "device"}


def _attestation_payload(value: Any) -> Any:
    """Remove process-specific pointer/device fields before hashing topology."""
    if isinstance(value, dict):
        return {
            key: _attestation_payload(item)
            for key, item in value.items()
            if key not in _VOLATILE_ATTESTATION_FIELDS and key != "topology_sha256"
        }
    if isinstance(value, list):
        return [_attestation_payload(item) for item in value]
    return value


def _unsigned_hash(topology: dict[str, Any]) -> str:
    payload = json.dumps(
        _attestation_payload(topology), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_generated_header(topology: dict[str, Any]) -> str:
    """Render compile-time constants from an exported topology object."""
    decoder = topology["decoder"]
    rope = decoder["rope"]
    section = list(rope.get("mrope_section", []))
    while len(section) < 3:
        section.append(0)
    attention_scale = float(decoder["attention_scale"])
    topology_hash = str(topology.get("topology_sha256", "UNHASHED"))

    def c_float(value: float) -> str:
        rendered = f"{float(value):.9g}"
        if "e" not in rendered.lower() and "." not in rendered:
            rendered += ".0"
        return rendered
    return f'''/* Generated by tools/kernel_gates/export_qwen_native_topology.py. DO NOT EDIT. */
#ifndef QWEN_NATIVE_TOPOLOGY_H
#define QWEN_NATIVE_TOPOLOGY_H

#include <stdint.h>

#define QWEN_NATIVE_TOPOLOGY_SCHEMA 1u
#define QWEN_NATIVE_TOPOLOGY_SHA256 "{topology_hash}"
#define QWEN_NATIVE_NUM_LAYERS {int(decoder["num_hidden_layers"])}u
#define QWEN_NATIVE_HIDDEN_SIZE {int(decoder["hidden_size"])}u
#define QWEN_NATIVE_INTERMEDIATE_SIZE {int(decoder["intermediate_size"])}u
#define QWEN_NATIVE_NUM_ATTN_HEADS {int(decoder["num_attention_heads"])}u
#define QWEN_NATIVE_NUM_KV_HEADS {int(decoder["num_key_value_heads"])}u
#define QWEN_NATIVE_NUM_KV_GROUPS {int(decoder["num_key_value_groups"])}u
#define QWEN_NATIVE_HEAD_DIM {int(decoder["head_dim"])}u
#define QWEN_NATIVE_MAX_POSITION_EMBEDDINGS {int(decoder["max_position_embeddings"])}u
#define QWEN_NATIVE_CODEC_VOCAB_SIZE {int(topology["codec_head"]["out_features"])}u
#define QWEN_NATIVE_CODEC_HEAD_DIM QWEN_NATIVE_CODEC_VOCAB_SIZE
#define QWEN_NATIVE_ROPE_INTERLEAVED {1 if bool(rope.get("interleaved", False)) else 0}
#define QWEN_NATIVE_MROPE_SECTION_0 {int(section[0])}u
#define QWEN_NATIVE_MROPE_SECTION_1 {int(section[1])}u
#define QWEN_NATIVE_MROPE_SECTION_2 {int(section[2])}u
#define QWEN_NATIVE_MROPE_SECTION_COUNT 3u
#define QWEN_NATIVE_HIDDEN_ACT_SILU 1u

static constexpr float QWEN_NATIVE_RMS_NORM_EPS = {c_float(float(decoder["rms_norm_eps"]))}f;
static constexpr float QWEN_NATIVE_ROPE_THETA = {c_float(float(rope["theta"]))}f;
static constexpr float QWEN_NATIVE_ATTENTION_SCALE = {c_float(attention_scale)}f;

#endif /* QWEN_NATIVE_TOPOLOGY_H */
'''


def write_artifacts(topology: dict[str, Any], output: Path, header: Path) -> dict[str, Any]:
    materialized = dict(topology)
    materialized["topology_sha256"] = _unsigned_hash(materialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    header.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(materialized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    header.write_text(render_generated_header(materialized), encoding="utf-8")
    return materialized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/kernel_gates/QWEN_NATIVE/QWEN_NATIVE_TOPOLOGY.json"),
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("native/qwen_talker/generated/qwen_native_topology.h"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from duplex_kernel.models.talker import load_talker

    talker = load_talker(str(args.workspace), device=args.device, load_code2wav=False)
    source_snapshot = str(sorted((args.workspace / ".hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots").glob("*"))[-1])
    topology = extract_topology(talker, source_snapshot=source_snapshot, git_sha=_git_sha(args.workspace))
    topology = write_artifacts(topology, args.output, args.header)
    print(json.dumps({
        "output": str(args.output),
        "header": str(args.header),
        "topology_sha256": topology["topology_sha256"],
        "layers": topology["decoder"]["num_hidden_layers"],
        "hidden": topology["decoder"]["hidden_size"],
        "intermediate": topology["decoder"]["intermediate_size"],
        "q_heads": topology["decoder"]["num_attention_heads"],
        "kv_heads": topology["decoder"]["num_key_value_heads"],
        "head_dim": topology["decoder"]["head_dim"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
