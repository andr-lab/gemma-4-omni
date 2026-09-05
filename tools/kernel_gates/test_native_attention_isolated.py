#!/usr/bin/env python3
"""Isolate Gemma 4 attention against the public ATen SDPA implementation.

The script obtains Q/K/V and the explicit mask from the same instrumented
Transformers path used by the layer-local gate.  It then compares that oracle
with the C ABI ATen bridge, including several consecutive Q=4 frames.  This
is diagnostic-only and deliberately does not change the native executor.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
import torch.nn.functional as F

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import (
    GEMMA4_HIDDEN_SIZE,
    NativeGemmaListenerLM,
)
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from tools.kernel_gates.p4_listener_cuda_graph_bisect import InstrumentedGemmaListenerLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

STRICT_MAE_MAX = 1e-4
STRICT_COSINE_MIN = 0.9999
CONTRACT_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_SDPA_CONTRACT.json")
ISOLATION_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_SDPA_ISOLATION.json")
TENSOR_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_SDPA_TENSORS.pt")


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    delta = (ref - cand).abs()
    denom = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand)
    cosine = torch.dot(ref, cand) / denom if float(denom) else torch.tensor(1.0, device=ref.device)
    return {
        "mae": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
        "cosine": float(cosine.item()),
        "exact_element_stream": bool(torch.equal(ref, cand)),
    }


def tensor_contract(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "strides": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
    }


def repeat_kv_transformers(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Literal transformers.integrations.sdpa_attention.repeat_kv layout."""
    batch, kv_heads, seq_len, head_dim = tensor.shape
    groups = query_heads // kv_heads
    return tensor[:, :, None, :, :].expand(
        batch, kv_heads, groups, seq_len, head_dim
    ).reshape(batch, query_heads, seq_len, head_dim)


def main() -> int:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    frames = 3
    thinker = load_thinker("/workspace", device=device, dtype=dtype)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    shape_stable = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

    input_ids, attention = backend._build_multimodal_inputs((), 1024, device=device)
    prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
    prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT].to(device=device, dtype=torch.long).contiguous()
    with torch.inference_mode():
        prefix_embeds = lm.embed_tokens(prefix_ids).contiguous()
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds).contiguous()
    prefix_positions = torch.arange(AUDIO_START_AFTER_TEXT, device=device, dtype=torch.long).view(1, -1)
    prefix_masks = create_masks_for_generate(
        lm.config, prefix_embeds, prefix_attention,
        past_key_values=None, position_ids=prefix_positions,
    )
    with torch.inference_mode():
        prefix_output = shape_stable.forward(
            inputs_embeds=prefix_embeds,
            per_layer_inputs=prefix_ple,
            attention_mask=prefix_masks,
            position_ids=prefix_positions,
            past_key_values=None,
            shared_kv_states=None,
        )
    prefix_cache = prefix_output.past_key_values

    taps: dict[tuple[int, int, str], torch.Tensor] = {}

    def callback(frame_id: int, layer_idx: int, operation: str, tensor: torch.Tensor) -> None:
        taps[(frame_id, layer_idx, operation)] = tensor.detach().clone()

    oracle = InstrumentedGemmaListenerLM(lm, tap_callback=callback)
    oracle._workspace = oracle._workspace_stream
    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    rng = np.random.default_rng(20260905)
    rows: list[dict[str, Any]] = []
    tensor_artifact: dict[str, torch.Tensor] = {}
    contract: dict[str, Any] | None = None

    try:
        for frame_id in range(frames):
            frame = torch.tensor(
                0.05 * rng.standard_normal((1, 4, GEMMA4_HIDDEN_SIZE)),
                device=device,
                dtype=dtype,
            ).contiguous()
            raw_frame_ple = backend._frame_pad_per_layer_inputs(lm)
            with torch.inference_mode():
                projected_ple = oracle._project_per_layer_inputs(frame, raw_frame_ple).contiguous()

            current_position = AUDIO_START_AFTER_TEXT + frame_id * 4
            positions = torch.arange(current_position, current_position + 4, device=device, dtype=torch.long).view(1, -1)
            step_attention = torch.ones(
                (1, current_position + 4), device=device, dtype=torch.long
            )
            step_attention[:, :AUDIO_START_AFTER_TEXT] = prefix_attention
            step_masks = create_masks_for_generate(
                lm.config, frame, step_attention,
                past_key_values=prefix_cache, position_ids=positions,
            )
            with torch.inference_mode():
                oracle.forward_instrumented(
                    inputs_embeds=frame,
                    per_layer_inputs=raw_frame_ple,
                    attention_mask=step_masks,
                    position_ids=positions,
                    past_key_values=prefix_cache,
                    shared_kv_states=getattr(prefix_cache, "shared_layers", None),
                    frame_id=frame_id,
                )

            attention_layer = lm.layers[0].self_attn
            layer_type = str(getattr(attention_layer, "layer_type", "full_attention"))
            mask = step_masks[layer_type]
            # Instrumented q_rope is already [B,Hq,Q,D] after the model's
            # transpose(1, 2); canonical_oracle_tensor transposes only to
            # match the native [Q,H,D] scratch layout.
            query = taps[(frame_id, 0, "q_rope")].contiguous()
            key = taps[(frame_id, 0, "kv_write_key")]
            value = taps[(frame_id, 0, "kv_write_val")]
            if mask is None:
                raise RuntimeError("layer 0 attention mask unexpectedly absent")
            if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or mask.ndim != 4:
                raise RuntimeError("unexpected Q/K/V/mask rank in oracle capture")

            query_heads = int(query.shape[1])
            repeated_key = repeat_kv_transformers(key, query_heads)
            repeated_value = repeat_kv_transformers(value, query_heads)
            oracle_heads = F.scaled_dot_product_attention(
                query,
                repeated_key,
                repeated_value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=float(attention_layer.scaling),
                enable_gqa=False,
            )
            oracle_output, _ = sdpa_attention_forward(
                attention_layer,
                query,
                key,
                value,
                mask,
                dropout=0.0,
                scaling=float(attention_layer.scaling),
                is_causal=False,
            )
            if not torch.equal(oracle_output, oracle_heads.transpose(1, 2).contiguous()):
                raise AssertionError("direct explicit-mask SDPA differs from Transformers SDPA wrapper")

            bridge_heads = native.debug_attention(
                query, key, value, mask, scale=float(attention_layer.scaling)
            )
            torch.cuda.synchronize()
            bridge_output = bridge_heads.transpose(1, 2).contiguous()
            row = {
                "frame": frame_id,
                "layer": 0,
                "operation": "sdpa_output",
                **metrics(oracle_output, bridge_output),
            }
            rows.append(row)
            print("[SDPA FRAME]", json.dumps(row, sort_keys=True), flush=True)

            if contract is None:
                contract = {
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "aten_schema": str(torch.ops.aten.scaled_dot_product_attention.default._schema),
                    "device": str(device),
                    "layer": 0,
                    "layer_type": layer_type,
                    "query": tensor_contract(query),
                    "key": tensor_contract(key),
                    "value": tensor_contract(value),
                    "attn_mask": tensor_contract(mask),
                    "oracle_output": tensor_contract(oracle_output),
                    "repeat_kv": {
                        "source": "transformers.integrations.sdpa_attention.repeat_kv",
                        "expand_shape": [1, int(key.shape[1]), 1, int(key.shape[2]), int(key.shape[3])],
                        "result_shape": [1, query_heads, int(key.shape[2]), int(key.shape[3])],
                        "enable_gqa": False,
                    },
                    "dropout_p": 0.0,
                    "is_causal": False,
                    "scale": float(attention_layer.scaling),
                    "mask_semantics": "explicit mask passed unchanged to public ATen SDPA; bool True=allowed",
                    "source_references": {
                        "transformers_repeat_kv": "https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py",
                        "pytorch_sdpa": "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
                    },
                }

            tensor_artifact[f"frame_{frame_id}/query"] = query.detach().cpu()
            tensor_artifact[f"frame_{frame_id}/key"] = key.detach().cpu()
            tensor_artifact[f"frame_{frame_id}/value"] = value.detach().cpu()
            tensor_artifact[f"frame_{frame_id}/mask"] = mask.detach().cpu()
            tensor_artifact[f"frame_{frame_id}/oracle_output"] = oracle_output.detach().cpu()
            tensor_artifact[f"frame_{frame_id}/bridge_output"] = bridge_output.detach().cpu()
    finally:
        native.close()

    if contract is None:
        raise RuntimeError("no SDPA frame was captured")
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_artifact, TENSOR_PATH)
    tensor_sha256 = hashlib.sha256(TENSOR_PATH.read_bytes()).hexdigest()
    contract["tensor_artifact"] = {"path": str(TENSOR_PATH), "sha256": tensor_sha256}
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    first_nonzero = next((row for row in rows if not row["exact_element_stream"]), None)
    first_mae_gt_1e_7 = next((row for row in rows if row["mae"] > 1e-7), None)
    first_mae_gt_1e_6 = next((row for row in rows if row["mae"] > 1e-6), None)
    first_formal_failure = next(
        (row for row in rows if row["mae"] > STRICT_MAE_MAX or row["cosine"] < STRICT_COSINE_MIN),
        None,
    )
    result = {
        "status": "PASS" if first_formal_failure is None else "NO-GO",
        "strict_thresholds": {"mae_max": STRICT_MAE_MAX, "cosine_min": STRICT_COSINE_MIN},
        "frames_checked": len(rows),
        "rows": rows,
        "first_divergence_markers": {
            "first_nonzero_difference": first_nonzero,
            "first_mae_gt_1e-7": first_mae_gt_1e_7,
            "first_mae_gt_1e-6": first_mae_gt_1e_6,
            "first_formal_failure": first_formal_failure,
        },
        "contract_artifact": {"path": str(CONTRACT_PATH)},
        "tensor_artifact": {"path": str(TENSOR_PATH), "sha256": tensor_sha256},
    }
    ISOLATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
