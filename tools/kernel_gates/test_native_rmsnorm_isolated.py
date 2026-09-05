#!/usr/bin/env python3
"""Isolate the first layer-local RMSNorm mismatch.

The input is the exact oracle ``layer 2 / attn_residual`` tensor produced by
ShapeStableGemmaListenerLM from the canonical prefix and deterministic frame
used by the P4 layer-local gate.  No synthetic RMSNorm input is generated.

The native side captures the current kernel's FP32 intermediates; this test
compares each boundary against the literal ShapeStable sequence and reports
per-row scalar differences, including float32 bit/ULP information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from tools.kernel_gates.p4_listener_cuda_graph_bisect import InstrumentedGemmaListenerLM
from tools.kernel_gates.test_native_layer_local_parity import PrefixLayerCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

DEFAULT_REPORT = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_RMSNORM_ISOLATION.json")
DEFAULT_INPUT = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_RMSNORM_INPUT.pt")

# The diagnostic target is deliberately stronger than the final model gate.
DIAGNOSTIC_MAE_MAX = 1e-7
EXPECTED_TORCH_VERSION = "2.12.1+cu130"


def float_bits(value: float) -> str:
    bits = int(np.asarray(np.float32(value)).view(np.uint32))
    return f"0x{bits:08x}"


def float_ulp_distance(a: float, b: float) -> int:
    a_bits = int(np.asarray(np.float32(a)).view(np.int32))
    b_bits = int(np.asarray(np.float32(b)).view(np.int32))

    def ordered(bits: int) -> int:
        return 0x80000000 - bits if bits < 0 else bits

    return abs(ordered(a_bits) - ordered(b_bits))


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    delta = (ref - cand).abs()
    return {
        "mae": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
        "exact_element_stream": bool(torch.equal(ref, cand)),
    }


def scalar_rows(reference: torch.Tensor, candidate: torch.Tensor) -> list[dict[str, Any]]:
    ref = reference.detach().float().reshape(-1).cpu()
    cand = candidate.detach().float().reshape(-1).cpu()
    rows: list[dict[str, Any]] = []
    for index, (oracle, native) in enumerate(zip(ref.tolist(), cand.tolist())):
        rows.append({
            "row": index,
            "oracle": oracle,
            "native": native,
            "absolute_difference": abs(oracle - native),
            "oracle_bits": float_bits(oracle),
            "native_bits": float_bits(native),
            "ulp_distance": float_ulp_distance(oracle, native),
        })
    return rows


def derive_exact_oracle_input(
    lm: Any, backend: GemmaAudioTowerBackend
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any, dict[tuple[int, str], torch.Tensor]]:
    """Reproduce the canonical layer-local gate and return all exact taps."""
    device = torch.device("cuda:0")
    shape_stable = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    taps: dict[tuple[int, str], torch.Tensor] = {}

    def callback(_: int, layer_idx: int, operation: str, tensor: torch.Tensor) -> None:
        taps[(layer_idx, operation)] = tensor.detach().clone()

    oracle = InstrumentedGemmaListenerLM(lm, tap_callback=callback)
    oracle._workspace = oracle._workspace_stream

    input_ids, attention = backend._build_multimodal_inputs((), 1024, device=device)
    prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
    prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT].to(
        device=device, dtype=torch.long
    ).contiguous()
    with torch.inference_mode():
        prefix_embeds = lm.embed_tokens(prefix_ids).contiguous()
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds).contiguous()

    prefix_positions = torch.arange(
        AUDIO_START_AFTER_TEXT, device=device, dtype=torch.long
    ).view(1, -1)
    prefix_masks = create_masks_for_generate(
        lm.config,
        prefix_embeds,
        prefix_attention,
        past_key_values=None,
        position_ids=prefix_positions,
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
    native_prefix_layers = [(layer.keys, layer.values) for layer in prefix_cache.layers]
    native_prefix_shared = {
        int(key): (value[0], value[1])
        for key, value in getattr(prefix_cache, "shared_layers", {}).items()
    }

    rng = np.random.default_rng(2026)
    frame = torch.tensor(
        0.05 * rng.standard_normal((1, 4, 1536)),
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    raw_frame_ple = backend._frame_pad_per_layer_inputs(lm)
    with torch.inference_mode():
        projected_ple = oracle._project_per_layer_inputs(frame, raw_frame_ple).contiguous()

    positions = torch.arange(
        AUDIO_START_AFTER_TEXT,
        AUDIO_START_AFTER_TEXT + 4,
        device=device,
        dtype=torch.long,
    ).view(1, -1)
    step_attention = torch.ones(
        (1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long
    )
    step_attention[:, :AUDIO_START_AFTER_TEXT] = prefix_attention
    step_masks = create_masks_for_generate(
        lm.config,
        frame,
        step_attention,
        past_key_values=prefix_cache,
        position_ids=positions,
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

    native_cache = PrefixLayerCache(native_prefix_layers, native_prefix_shared)
    return (
        taps[(2, "input_hidden")],
        taps[(2, "attn_residual")],
        taps[(2, "pre_ff_norm")],
        projected_ple,
        native_cache,
        taps,
    )


def literal_oracle(norm: Any, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """The literal ShapeStable RMSNorm sequence, boundary by boundary."""
    input_bf16 = input_tensor.detach().contiguous()
    input_fp32 = input_bf16.float()
    x_squared = input_fp32.pow(2)
    row_sum = x_squared.sum(-1).reshape(-1)
    row_mean = x_squared.mean(-1).reshape(-1)
    mean_plus_eps = row_mean + float(norm.eps)
    pow_neg_half = torch.pow(mean_plus_eps, -0.5)
    normalized = input_fp32 * pow_neg_half.unsqueeze(-1)
    weighted = normalized * norm.weight.float()
    final_bf16 = weighted.to(torch.bfloat16)
    return {
        "input_bf16": input_bf16,
        "input_fp32": input_fp32,
        "x_squared_fp32": x_squared,
        "row_sum_fp32": row_sum,
        "row_mean_fp32": row_mean,
        "mean_plus_eps_fp32": mean_plus_eps,
        "pow_neg_half_fp32": pow_neg_half,
        "normalized_fp32": normalized,
        "weighted_fp32": weighted,
        "final_bf16": final_bf16,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input-output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--operation", choices=("pre_ff_norm", "post_attn_norm"), default="pre_ff_norm")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("RMSNorm isolation richiede CUDA")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"RMSNorm isolation richiede PyTorch {EXPECTED_TORCH_VERSION}, "
            f"trovato {torch.__version__}"
        )
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # The SDPA bridge is the already-certified baseline; it prevents the
    # previously known layer-0 attention drift from contaminating layer-2's
    # exact oracle input while this test examines RMSNorm only.
    os.environ["GEMMA4_ATEN_SDPA_BRIDGE"] = "1"
    thinker = load_thinker("/workspace", device=device, dtype=dtype)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    layer_input, _, _, projected_ple, prefix_cache, taps = derive_exact_oracle_input(
        lm, backend
    )
    if not 0 <= args.layer < len(lm.layers):
        raise ValueError(f"layer fuori range: {args.layer}")
    input_operation = "attn_residual" if args.operation == "pre_ff_norm" else "o_proj"
    norm = (
        lm.layers[args.layer].pre_feedforward_layernorm
        if args.operation == "pre_ff_norm"
        else lm.layers[args.layer].post_attention_layernorm
    )
    layer_input = taps[(args.layer, "input_hidden")]
    oracle_input = taps[(args.layer, input_operation)]
    oracle_norm_output = taps[(args.layer, args.operation)]
    oracle = literal_oracle(norm, oracle_input)

    input_artifact = {
        "layer": args.layer,
        "boundary": f"{args.operation} input / {input_operation}",
        "tensor": oracle_input.detach().cpu(),
        "dtype": str(oracle_input.dtype),
        "shape": list(oracle_input.shape),
    }
    args.input_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(input_artifact, args.input_output)

    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    try:
        native_trace = native.debug_rmsnorm_trace(
            oracle_input,
            norm.weight,
            eps=float(norm.eps),
        )
        aten_output = native.aten_rmsnorm_parity(
            oracle_input,
            norm.weight,
            eps=float(norm.eps),
        )
        native.debug_execute_layer(
            layer_idx=args.layer,
            hidden_input=layer_input,
            projected_ple=projected_ple,
            past_key_values=prefix_cache,
            current_position=AUDIO_START_AFTER_TEXT,
            query_position=AUDIO_START_AFTER_TEXT,
        )
        torch.cuda.synchronize()
        native_layer_input = native.copy_debug_tensor(
            f"tap_{input_operation}", (4, 1536)
        )
        native_layer_output = native.copy_debug_tensor(
            f"tap_{args.operation}", (4, 1536)
        )
    finally:
        native.close()

    boundary_pairs = (
        ("input_bf16", oracle["input_bf16"], native_trace["input_bf16"]),
        ("input_fp32", oracle["input_fp32"], native_trace["input_fp32"]),
        ("x_squared_fp32", oracle["x_squared_fp32"], native_trace["x_squared_fp32"]),
        ("row_sum_fp32", oracle["row_sum_fp32"], native_trace["row_sum_fp32"]),
        ("row_mean_fp32", oracle["row_mean_fp32"], native_trace["row_mean_fp32"]),
        ("mean_plus_eps_fp32", oracle["mean_plus_eps_fp32"], native_trace["mean_plus_eps_fp32"]),
        ("pow_neg_half_fp32", oracle["pow_neg_half_fp32"], native_trace["pow_neg_half_fp32"]),
        ("normalized_fp32", oracle["normalized_fp32"], native_trace["normalized_fp32"]),
        ("weighted_fp32", oracle["weighted_fp32"], native_trace["weighted_fp32"]),
        ("final_bf16", oracle["final_bf16"], native_trace["final_bf16"]),
    )
    boundaries: dict[str, Any] = {}
    first_nonzero: dict[str, Any] | None = None
    for name, reference, candidate in boundary_pairs:
        row = {"boundary": name, **tensor_metrics(reference, candidate)}
        if reference.ndim == 1 or reference.numel() == reference.shape[0]:
            row["rows"] = scalar_rows(reference, candidate)
        boundaries[name] = row
        if first_nonzero is None and not row["exact_element_stream"]:
            first_nonzero = row

    aten_row = {
        "boundary": "aten_literal_rmsnorm_final_bf16",
        **tensor_metrics(oracle["final_bf16"], aten_output),
    }
    layer_input_row = {
        "boundary": f"layer{args.layer}_{input_operation}",
        **tensor_metrics(oracle_input.reshape(4, 1536), native_layer_input),
    }
    layer_output_row = {
        "boundary": f"layer{args.layer}_{args.operation}_output",
        **tensor_metrics(oracle_norm_output, native_layer_output),
    }
    report = {
        "status": "PASS" if first_nonzero is None else "NO-GO",
        "diagnostic_target": {"mae_max": DIAGNOSTIC_MAE_MAX},
        "contract": {
            "torch_version": torch.__version__,
            "layer": args.layer,
            "operation": args.operation,
            "input_source": f"exact ShapeStable oracle {input_operation}, canonical prefix + deterministic gate frame",
            "input_shape": list(oracle_input.shape),
            "input_dtype": str(oracle_input.dtype),
            "weight_shape": list(norm.weight.shape),
            "weight_dtype": str(norm.weight.dtype),
            "eps": float(norm.eps),
            "sequence": "float -> pow(2) -> sum/mean -> + eps -> pow(-0.5) -> input*scale -> weight -> BF16",
            "native_kernel": "current kernel_rmsnorm_bf16 (diagnostic trace variant only; production kernel unchanged)",
        },
        "input_artifact": {
            "path": str(args.input_output),
            "sha256": hashlib.sha256(args.input_output.read_bytes()).hexdigest(),
        },
        "first_nonzero_difference": first_nonzero,
        "aten_literal_bridge": aten_row,
        "boundaries": boundaries,
        "layer_boundary_checks": [layer_input_row, layer_output_row],
        "note": "A nonzero row_mean/row_sum is evidence for reduction-order drift; this gate does not modify RMSNorm production behavior.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
