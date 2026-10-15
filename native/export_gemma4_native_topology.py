#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_gemma4_native_topology.py — Phase 0: Attest the actual Gemma 4 checkpoint topology.

This script loads the live checkpoint used in production on RTX 3090, inspects
the actual module architecture and parameters, and outputs:
    validation/kernel_gates/P4_NATIVE/GEMMA4_NATIVE_TOPOLOGY.json

As specified in the P4-NATIVE Implementation Plan §4.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_checkpoint_hash(checkpoint_dir: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    patterns = ["*.safetensors", "config.json", "model.safetensors.index.json"]
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(checkpoint_dir, pat))):
            name = os.path.basename(p)
            hashes[name] = compute_file_sha256(p)
    return hashes


def extract_topology(workspace: str = "/workspace") -> dict[str, Any]:
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path

    ckpt_path = resolve_thinker_path(workspace)
    print(f"[export_topology] Loading thinker from {ckpt_path}...")
    ckpt_hashes = compute_checkpoint_hash(ckpt_path)

    thinker = load_thinker(workspace, device="cuda" if torch.cuda.is_available() else "cpu")
    model = thinker.model

    # Locate the inner language model
    if hasattr(model, "language_model"):
        lm = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    elif hasattr(model, "model"):
        lm = model.model
    else:
        lm = model

    cfg = lm.config if hasattr(lm, "config") else model.config
    print(f"[export_topology] LM type: {type(lm).__name__}, Config type: {type(cfg).__name__}")

    num_layers = getattr(cfg, "num_hidden_layers", len(lm.layers))
    hidden_size = getattr(cfg, "hidden_size", 1536)
    intermediate_size = getattr(cfg, "intermediate_size", 6144)
    max_position_embeddings = getattr(cfg, "max_position_embeddings", 8192)
    hidden_size_per_layer_input = getattr(cfg, "hidden_size_per_layer_input", 256)

    # Inspect rotary_emb
    rotary_info: dict[str, Any] = {}
    if hasattr(lm, "rotary_emb"):
        rot = lm.rotary_emb
        rotary_info = {
            "type": type(rot).__name__,
            "dim": getattr(rot, "dim", None),
            "max_position_embeddings": getattr(rot, "max_position_embeddings", None),
            "base": getattr(rot, "base", None),
        }
    unique_layer_types = list(getattr(lm, "unique_layer_types", []))
    layer_types_list = list(getattr(cfg, "layer_types", [getattr(l, "layer_type", "unknown") for l in lm.layers]))

    # Global modules
    norm_eps = getattr(getattr(lm, "norm", None), "eps", None)

    # Per-layer input (PLE) projection
    ple_info: dict[str, Any] = {
        "hidden_size_per_layer_input": hidden_size_per_layer_input,
        "per_layer_model_projection_shape": list(lm.per_layer_model_projection.weight.shape) if hasattr(lm, "per_layer_model_projection") and hasattr(lm.per_layer_model_projection, "weight") else None,
        "per_layer_model_projection_scale": float(getattr(lm, "per_layer_model_projection_scale", 1.0)),
        "per_layer_projection_norm_eps": float(getattr(getattr(lm, "per_layer_projection_norm", None), "eps", 1e-6)),
        "per_layer_input_scale": float(getattr(lm, "per_layer_input_scale", 1.0)),
    }

    # Inspect each layer
    layers_data: list[dict[str, Any]] = []
    for idx, layer in enumerate(lm.layers[:num_layers]):
        attn = layer.self_attn
        mlp = layer.mlp
        l_type = layer_types_list[idx] if idx < len(layer_types_list) else getattr(layer, "layer_type", "unknown")

        # Attention shapes
        q_shape = list(attn.q_proj.weight.shape) if hasattr(attn, "q_proj") and attn.q_proj is not None else None
        k_shape = list(attn.k_proj.weight.shape) if hasattr(attn, "k_proj") and attn.k_proj is not None else None
        v_shape = list(attn.v_proj.weight.shape) if hasattr(attn, "v_proj") and attn.v_proj is not None else None
        o_shape = list(attn.o_proj.weight.shape) if hasattr(attn, "o_proj") and attn.o_proj is not None else None

        # MLP shapes
        gate_shape = list(mlp.gate_proj.weight.shape) if hasattr(mlp, "gate_proj") and mlp.gate_proj is not None else None
        up_shape = list(mlp.up_proj.weight.shape) if hasattr(mlp, "up_proj") and mlp.up_proj is not None else None
        down_shape = list(mlp.down_proj.weight.shape) if hasattr(mlp, "down_proj") and mlp.down_proj is not None else None

        # PLE shapes
        ple_gate_shape = list(layer.per_layer_input_gate.weight.shape) if hasattr(layer, "per_layer_input_gate") and layer.per_layer_input_gate is not None else None
        ple_proj_shape = list(layer.per_layer_projection.weight.shape) if hasattr(layer, "per_layer_projection") and layer.per_layer_projection is not None else None

        # RoPE on attention
        attn_rope_base = getattr(attn, "rope_theta", getattr(cfg, "rope_theta", 10000.0))
        attn_rope_scaling = getattr(attn, "rope_scaling", getattr(cfg, "rope_scaling", None))

        # Layer scalar
        layer_scalar = getattr(layer, "layer_scalar", None)
        if layer_scalar is not None:
            if isinstance(layer_scalar, torch.Tensor):
                layer_scalar_val = float(layer_scalar.item()) if layer_scalar.numel() == 1 else layer_scalar.tolist()
            else:
                layer_scalar_val = float(layer_scalar)
        else:
            layer_scalar_val = 1.0

        layer_dict = {
            "index": idx,
            "layer_type": l_type,
            "num_attention_heads": getattr(attn, "num_heads", getattr(cfg, "num_attention_heads", 8)),
            "num_key_value_heads": getattr(attn, "num_key_value_heads", getattr(cfg, "num_key_value_heads", 1)),
            "head_dim": getattr(attn, "head_dim", 256),
            "sliding_window": getattr(attn, "sliding_window", None),
            "attention_scaling": float(getattr(attn, "scaling", 1.0 / (getattr(attn, "head_dim", 256) ** 0.5))),
            "rope_base": float(attn_rope_base) if isinstance(attn_rope_base, (int, float)) else str(attn_rope_base),
            "rope_scaling": attn_rope_scaling,
            "is_kv_shared_layer": bool(getattr(attn, "is_kv_shared_layer", False)),
            "kv_shared_layer_index": getattr(attn, "kv_shared_layer_index", None),
            "store_full_length_kv": bool(getattr(attn, "store_full_length_kv", False)),
            "q_proj_shape": q_shape,
            "k_proj_shape": k_shape,
            "v_proj_shape": v_shape,
            "o_proj_shape": o_shape,
            "gate_proj_shape": gate_shape,
            "up_proj_shape": up_shape,
            "down_proj_shape": down_shape,
            "per_layer_input_gate_shape": ple_gate_shape,
            "per_layer_projection_shape": ple_proj_shape,
            "input_layernorm_eps": float(getattr(getattr(layer, "input_layernorm", None), "eps", 1e-6)),
            "post_attention_layernorm_eps": float(getattr(getattr(layer, "post_attention_layernorm", None), "eps", 1e-6)),
            "pre_feedforward_layernorm_eps": float(getattr(getattr(layer, "pre_feedforward_layernorm", None), "eps", 1e-6)),
            "post_feedforward_layernorm_eps": float(getattr(getattr(layer, "post_feedforward_layernorm", None), "eps", 1e-6)),
            "q_norm_eps": float(getattr(getattr(attn, "q_norm", None), "eps", 1e-6)) if hasattr(attn, "q_norm") and attn.q_norm is not None else None,
            "k_norm_eps": float(getattr(getattr(attn, "k_norm", None), "eps", 1e-6)) if hasattr(attn, "k_norm") and attn.k_norm is not None else None,
            "v_norm_eps": float(getattr(getattr(attn, "v_norm", None), "eps", 1e-6)) if hasattr(attn, "v_norm") and attn.v_norm is not None else None,
            "post_per_layer_input_norm_eps": float(getattr(getattr(layer, "post_per_layer_input_norm", None), "eps", 1e-6)) if hasattr(layer, "post_per_layer_input_norm") and layer.post_per_layer_input_norm is not None else None,
            "layer_scalar": layer_scalar_val,
        }
        layers_data.append(layer_dict)

    # MLP and PLE activation types
    mlp_act = getattr(lm.layers[0].mlp, "act_fn", None)
    mlp_act_name = mlp_act.__name__ if hasattr(mlp_act, "__name__") else type(mlp_act).__name__
    ple_act = getattr(lm.layers[0], "act_fn", None)
    ple_act_name = ple_act.__name__ if hasattr(ple_act, "__name__") else type(ple_act).__name__

    # The canonical listener representation is [BOS | 32 text slots | BOA].
    # Keep this contract in the attested manifest so generated native headers
    # do not duplicate or guess the prefix length.
    from duplex_kernel.gemma_representation import (
        AUDIO_START_AFTER_TEXT,
        CANONICAL_LM_PREFILL_TARGET_LEN,
        TEXT_BLOCK_LEN,
    )

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "checkpoint_path": ckpt_path,
        "canonical_text_block_len": int(TEXT_BLOCK_LEN),
        "canonical_prefix_len": int(AUDIO_START_AFTER_TEXT),
        "canonical_lm_prefill_target_len": int(CANONICAL_LM_PREFILL_TARGET_LEN),
        "checkpoint_files": ckpt_hashes,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "max_position_embeddings": max_position_embeddings,
        "hidden_size_per_layer_input": hidden_size_per_layer_input,
        "mlp_activation_type": mlp_act_name,
        "ple_activation_type": ple_act_name,
        "final_norm_eps": float(norm_eps) if norm_eps is not None else 1e-6,
        "unique_layer_types": unique_layer_types,
        "rotary_info": rotary_info,
        "ple_global_config": ple_info,
        "layers": layers_data,
    }

    manifest_json = json.dumps(manifest, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    return manifest


def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    manifest = extract_topology(workspace)

    out_dir = Path("validation/kernel_gates/P4_NATIVE")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "GEMMA4_NATIVE_TOPOLOGY.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SUCCESS] Attested topology exported to {out_path}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Layers: {manifest['num_hidden_layers']}, Hidden: {manifest['hidden_size']}, Intermediate: {manifest['intermediate_size']}")
    print(f"PLE dim: {manifest['hidden_size_per_layer_input']}, Final norm eps: {manifest['final_norm_eps']}")

    # Check shared KV stats
    shared_layers = [l["index"] for l in manifest["layers"] if l["is_kv_shared_layer"]]
    print(f"Shared KV layers ({len(shared_layers)}): {shared_layers}")


if __name__ == "__main__":
    main()
