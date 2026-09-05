#!/usr/bin/env python3
"""Strict local P4-NATIVE layer bisect from an exact ShapeStable prefix cache.

The native entrypoint receives the oracle hidden state, projected PLE and
prefix KV for one layer.  The CUDA executor captures operation-local tensors
without changing the production path; this gate compares the captures in
execution order and stops at the first strict parity failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import (
    GEMMA4_HIDDEN_SIZE,
    GEMMA4_NUM_LAYERS,
    GEMMA4_PLE_DIM,
    NativeGemmaListenerLM,
)
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from tools.kernel_gates.p4_listener_cuda_graph_bisect import InstrumentedGemmaListenerLM
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

STRICT_MAE_MAX = 1e-4
STRICT_COSINE_MIN = 0.9999
OUTPUT_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_LAYER_LOCAL_BISECT.json")
PREATTENTION_TENSOR_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_PREATTENTION_TENSORS.pt")
PREATTENTION_OPERATIONS = {
    "input_hidden", "input_norm", "q_proj", "q_norm", "q_rope",
    "k_proj", "k_norm", "k_rope", "v_proj", "v_norm",
    "kv_write_key", "kv_write_val",
}


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
    }


class PrefixLayerCache:
    """Minimal cache facade for one instrumented oracle layer."""

    def __init__(self, layers: list[tuple[torch.Tensor, torch.Tensor]], shared: dict[int, Any]):
        self.layers = [SimpleNamespace(keys=k, values=v) for k, v in layers]
        self.shared_layers = dict(shared)

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, **_: Any):
        layer = self.layers[int(layer_idx)]
        layer.keys = torch.cat((layer.keys, key), dim=-2)
        layer.values = torch.cat((layer.values, value), dim=-2)
        return layer.keys, layer.values


def canonical_oracle_tensor(operation: str, tensor: torch.Tensor) -> torch.Tensor:
    """Convert ShapeStable tap layouts to the native contiguous layout."""
    tensor = tensor.detach()
    if operation == "q_rope":
        # Oracle: [B, H, Q, D]; native: [Q, H, D].
        return tensor.transpose(1, 2).contiguous()
    if operation == "k_rope":
        # H=1, but make the intended native [Q, D] contract explicit.
        return tensor.transpose(1, 2).contiguous()
    if operation == "v_norm":
        return tensor.transpose(1, 2).contiguous()
    return tensor.contiguous()


def copy_native_cache(native: NativeGemmaListenerLM, layer_idx: int, which: str, seq_len: int, head_dim: int) -> torch.Tensor:
    ptr_fn = native._lib.gemma4_listener_get_cached_k if which == "k" else native._lib.gemma4_listener_get_cached_v
    ptr = ptr_fn(native._ctx, layer_idx)
    if not ptr:
        raise RuntimeError(f"native cache {which} layer {layer_idx} null")
    out = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=native.output_tensor.device)
    # Keep the copy implementation local to the diagnostic gate; production
    # execution never performs this host-visible cache readback.
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so")
    ret = libcudart.cudaMemcpy(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(out.numel() * out.element_size()),
        ctypes.c_int(3),
    )
    if ret != 0:
        raise RuntimeError(f"native cache cudaMemcpy failed with code {ret}")
    return out


TAP_SPECS = (
    ("input_hidden", "tap_input_hidden", (4, GEMMA4_HIDDEN_SIZE)),
    ("input_norm", "tap_input_norm", (4, GEMMA4_HIDDEN_SIZE)),
    ("q_proj", "tap_q_proj", (4, 8, None)),
    ("q_norm", "tap_q_norm", (4, 8, None)),
    ("q_rope", "tap_q_rope", (4, 8, None)),
    ("k_proj", "tap_k_proj", (4, None)),
    ("v_proj", "tap_v_proj", (4, None)),
    ("k_norm", "tap_k_norm", (4, None)),
    ("k_rope", "tap_k_rope", (4, None)),
    ("v_norm", "tap_v_norm", (4, None)),
    ("attn_output", "tap_attn_output", (4, 8, None)),
    ("o_proj", "tap_o_proj", (4, GEMMA4_HIDDEN_SIZE)),
    ("post_attn_norm", "tap_post_attn_norm", (4, GEMMA4_HIDDEN_SIZE)),
    ("attn_residual", "tap_attn_residual", (4, GEMMA4_HIDDEN_SIZE)),
    ("pre_ff_norm", "tap_pre_ff_norm", (4, GEMMA4_HIDDEN_SIZE)),
    ("gate_proj", "tap_gate_proj", (4, None)),
    ("up_proj", "tap_up_proj", (4, None)),
    ("mlp_act", "tap_mlp_act", (4, None)),
    ("mlp_gate_up", "tap_mlp_gate_up", (4, None)),
    ("down_proj", "tap_down_proj", (4, GEMMA4_HIDDEN_SIZE)),
    ("post_ff_norm", "tap_post_ff_norm", (4, GEMMA4_HIDDEN_SIZE)),
    ("ff_residual", "tap_ff_residual", (4, GEMMA4_HIDDEN_SIZE)),
    ("ple_gate", "tap_ple_gate", (4, GEMMA4_PLE_DIM)),
    ("ple_act", "tap_ple_act", (4, GEMMA4_PLE_DIM)),
    ("ple_mult", "tap_ple_mult", (4, GEMMA4_PLE_DIM)),
    ("ple_proj", "tap_ple_proj", (4, GEMMA4_HIDDEN_SIZE)),
    ("ple_norm", "tap_ple_norm", (4, GEMMA4_HIDDEN_SIZE)),
    ("ple_residual", "tap_ple_residual", (4, GEMMA4_HIDDEN_SIZE)),
    ("layer_output", "tap_layer_output", (4, GEMMA4_HIDDEN_SIZE)),
)


def main() -> int:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
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
    prefix_layers = [(layer.keys, layer.values) for layer in prefix_cache.layers]
    prefix_shared = {
        int(key): (value[0], value[1])
        for key, value in getattr(prefix_cache, "shared_layers", {}).items()
    }

    taps: dict[tuple[int, str], torch.Tensor] = {}
    def callback(_: int, layer_idx: int, operation: str, tensor: torch.Tensor) -> None:
        taps[(layer_idx, operation)] = tensor.detach().clone()

    oracle = InstrumentedGemmaListenerLM(lm, tap_callback=callback)
    oracle._workspace = oracle._workspace_stream
    rng = np.random.default_rng(2026)
    frame = torch.tensor(0.05 * rng.standard_normal((1, 4, GEMMA4_HIDDEN_SIZE)), device=device, dtype=dtype).contiguous()
    raw_frame_ple = backend._frame_pad_per_layer_inputs(lm)
    with torch.inference_mode():
        projected_ple = oracle._project_per_layer_inputs(frame, raw_frame_ple).contiguous()

    positions = torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device, dtype=torch.long).view(1, -1)
    step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
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
            frame_id=0,
        )

    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    native.reset()

    # Numerical future-leak probe: replace masked prefix KV slots while
    # holding the canonical BOS/BOA slots and all query inputs fixed.
    native.debug_execute_layer(
        layer_idx=0,
        hidden_input=taps[(0, "input_hidden")],
        projected_ple=projected_ple,
        past_key_values=PrefixLayerCache(prefix_layers, prefix_shared),
        current_position=AUDIO_START_AFTER_TEXT,
        query_position=AUDIO_START_AFTER_TEXT,
    )
    torch.cuda.synchronize()
    baseline_output = native.output_tensor.detach().clone()
    native.reset()
    mutated_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, (keys, values) in enumerate(prefix_layers):
        if layer_idx == 0:
            keys = keys.clone()
            values = values.clone()
            keys[..., 1:AUDIO_START_AFTER_TEXT - 1, :] = 123.0
            values[..., 1:AUDIO_START_AFTER_TEXT - 1, :] = -123.0
        mutated_layers.append((keys, values))
    native.debug_execute_layer(
        layer_idx=0,
        hidden_input=taps[(0, "input_hidden")],
        projected_ple=projected_ple,
        past_key_values=PrefixLayerCache(mutated_layers, prefix_shared),
        current_position=AUDIO_START_AFTER_TEXT,
        query_position=AUDIO_START_AFTER_TEXT,
    )
    torch.cuda.synchronize()
    mutated_output = native.output_tensor.detach().clone()
    future_leak = float((baseline_output.float() - mutated_output.float()).abs().max().item())
    native.reset()
    print(f"[FUTURE LEAK] max_abs={future_leak:.8e}", flush=True)

    first: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    divergence_markers: dict[str, dict[str, Any]] = {}
    pre_attention_tensors: dict[str, torch.Tensor] = {}

    def observe(row: dict[str, Any], reference: torch.Tensor, candidate: torch.Tensor) -> None:
        """Record the first exact/threshold crossing without changing execution."""
        if row["operation"] in PREATTENTION_OPERATIONS:
            prefix = f"layer_{row['layer']}/{row['operation']}"
            pre_attention_tensors[f"{prefix}/oracle"] = reference.detach().to("cpu").clone()
            pre_attention_tensors[f"{prefix}/native"] = candidate.detach().to("cpu").clone()
        # Layout views may differ at this boundary (e.g. [1,Q,H,D] vs
        # [Q,H,D]); exactness is over the ordered BF16 element stream.
        if "first_nonzero_difference" not in divergence_markers and not torch.equal(
            reference.reshape(-1), candidate.reshape(-1)
        ):
            divergence_markers["first_nonzero_difference"] = dict(row)
        if "first_mae_gt_1e-7" not in divergence_markers and row["mae"] > 1e-7:
            divergence_markers["first_mae_gt_1e-7"] = dict(row)
        if "first_mae_gt_1e-6" not in divergence_markers and row["mae"] > 1e-6:
            divergence_markers["first_mae_gt_1e-6"] = dict(row)
        if "first_formal_failure" not in divergence_markers and (
            row["mae"] > STRICT_MAE_MAX or row["cosine"] < STRICT_COSINE_MIN
        ):
            divergence_markers["first_formal_failure"] = dict(row)

    try:
        for layer_idx in range(GEMMA4_NUM_LAYERS):
            attention_layer = lm.layers[layer_idx].self_attn
            head_dim = int(attention_layer.head_dim)
            intermediate = int(lm.layers[layer_idx].mlp.intermediate_size)
            local_cache = PrefixLayerCache(prefix_layers, prefix_shared)
            native.debug_execute_layer(
                layer_idx=layer_idx,
                hidden_input=taps[(layer_idx, "input_hidden")],
                projected_ple=projected_ple,
                past_key_values=local_cache,
                current_position=AUDIO_START_AFTER_TEXT,
                query_position=AUDIO_START_AFTER_TEXT,
            )
            torch.cuda.synchronize()

            layer_rows: list[dict[str, Any]] = []
            operation_order = [spec[0] for spec in TAP_SPECS]
            if bool(getattr(attention_layer, "is_kv_shared_layer", False)):
                operation_order = [op for op in operation_order if op not in {"k_proj", "v_proj", "k_norm", "k_rope", "v_norm"}]
            for operation in operation_order:
                reference = canonical_oracle_tensor(operation, taps[(layer_idx, operation)])
                spec = next(item for item in TAP_SPECS if item[0] == operation)
                shape = tuple(head_dim if value is None else value for value in spec[2])
                if operation in {"gate_proj", "up_proj", "mlp_act", "mlp_gate_up"}:
                    shape = (4, intermediate)
                candidate = native.copy_debug_tensor(spec[1], shape)
                row = {"layer": layer_idx, "operation": operation, **metrics(reference, candidate)}
                layer_rows.append(row)
                observe(row, reference, candidate)
                if first is None and (
                    row["mae"] > STRICT_MAE_MAX or row["cosine"] < STRICT_COSINE_MIN
                ):
                    first = row
                    print("[FIRST DIVERGENCE]", json.dumps(row, sort_keys=True), flush=True)
                    break

            # KV writes are a separate cache boundary, not a scratch tap.
            # Capture them for the exact pre-attention artifact even when the
            # attention output is already the first formal failure.
            if not bool(getattr(attention_layer, "is_kv_shared_layer", False)):
                for which, operation in (("k", "kv_write_key"), ("v", "kv_write_val")):
                    ref = canonical_oracle_tensor(operation, taps[(layer_idx, operation)])
                    candidate = copy_native_cache(native, layer_idx, which, AUDIO_START_AFTER_TEXT + 4, head_dim)
                    row = {"layer": layer_idx, "operation": operation, **metrics(ref, candidate)}
                    if first is None:
                        layer_rows.append(row)
                    observe(row, ref, candidate)
                    if first is None and (
                        row["mae"] > STRICT_MAE_MAX or row["cosine"] < STRICT_COSINE_MIN
                    ):
                        first = row
                        print("[FIRST DIVERGENCE]", json.dumps(row, sort_keys=True), flush=True)
                        break
            rows.extend(layer_rows)
            if first is not None:
                break
    finally:
        native.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pre_attention_tensors, PREATTENTION_TENSOR_PATH)
    tensor_sha256 = hashlib.sha256(PREATTENTION_TENSOR_PATH.read_bytes()).hexdigest()

    result = {
        "status": "PASS" if first is None and future_leak <= 1e-5 else "NO-GO",
        "prefix_len": AUDIO_START_AFTER_TEXT,
        "layers_checked": (first["layer"] + 1) if first is not None else GEMMA4_NUM_LAYERS,
        "strict_thresholds": {
            "mae_max": STRICT_MAE_MAX,
            "cosine_min": STRICT_COSINE_MIN,
            "future_leak_max_abs": 1e-5,
        },
        "future_leak_max_abs": future_leak,
        "first_divergence": first,
        "first_divergence_markers": divergence_markers,
        "pre_attention_tensor_values": {
            "path": str(PREATTENTION_TENSOR_PATH),
            "sha256": tensor_sha256,
            "keys": sorted(pre_attention_tensors),
            "note": "Exact CPU BF16 oracle/native tensors captured before the first formal failure; use torch.load(..., weights_only=True).",
        },
        "rows_checked": len(rows),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
