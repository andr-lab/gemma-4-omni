#!/usr/bin/env python3
"""Isolate layer-2 FFN primitives with exact ShapeStable oracle inputs."""
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
import torch.nn.functional as F

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from tools.kernel_gates.p4_listener_cuda_graph_bisect import InstrumentedGemmaListenerLM
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

DEFAULT_REPORT = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_FFN_PRIMITIVES.json")
DEFAULT_TENSORS = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_FFN_PRIMITIVES_TENSORS.pt")
EXPECTED_TORCH_VERSION = "2.12.1+cu130"


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    delta = (ref - cand).abs()
    denom = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand)
    cosine = torch.dot(ref, cand) / denom if float(denom) else torch.tensor(1.0)
    return {
        "mae": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
        "cosine": float(cosine.item()),
        "exact_element_stream": bool(torch.equal(reference.reshape(-1), candidate.reshape(-1))),
    }


def derive_layer2_taps(lm: Any, backend: GemmaAudioTowerBackend) -> dict[str, torch.Tensor]:
    device = torch.device("cuda:0")
    shape_stable = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    taps: dict[str, torch.Tensor] = {}

    def callback(_: int, layer_idx: int, operation: str, tensor: torch.Tensor) -> None:
        if layer_idx == 2:
            taps[operation] = tensor.detach().clone()

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
    rng = np.random.default_rng(2026)
    frame = torch.tensor(
        0.05 * rng.standard_normal((1, 4, 1536)),
        device=device, dtype=torch.bfloat16,
    ).contiguous()
    raw_frame_ple = backend._frame_pad_per_layer_inputs(lm)
    with torch.inference_mode():
        taps["projected_ple"] = oracle._project_per_layer_inputs(
            frame, raw_frame_ple
        ).contiguous()
    positions = torch.arange(
        AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4,
        device=device, dtype=torch.long,
    ).view(1, -1)
    step_attention = torch.ones(
        (1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long
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
            frame_id=0,
        )
    required = {"pre_ff_norm", "gate_proj", "up_proj", "mlp_act", "mlp_gate_up", "down_proj"}
    missing = sorted(required - taps.keys())
    if missing:
        raise RuntimeError(f"layer-2 oracle taps mancanti: {missing}")
    return taps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tensor-output", type=Path, default=DEFAULT_TENSORS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FFN isolation richiede CUDA")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"FFN isolation richiede PyTorch {EXPECTED_TORCH_VERSION}, "
            f"trovato {torch.__version__}"
        )

    os.environ["GEMMA4_ATEN_SDPA_BRIDGE"] = "1"
    thinker = load_thinker("/workspace", device=torch.device("cuda:0"), dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device="cuda:0", use_native_lm=False)
    taps = derive_layer2_taps(lm, backend)
    mlp = lm.layers[2].mlp

    pre_ff = taps["pre_ff_norm"]
    gate_weight = mlp.gate_proj.weight
    up_weight = mlp.up_proj.weight
    down_weight = mlp.down_proj.weight
    with torch.inference_mode():
        gate_oracle = F.linear(pre_ff, gate_weight)
        up_oracle = F.linear(pre_ff, up_weight)
        act_oracle = mlp.act_fn(gate_oracle)
        gate_up_oracle = act_oracle * up_oracle
        down_oracle = F.linear(gate_up_oracle, down_weight)

    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    try:
        gate_native = native.debug_linear(pre_ff, gate_weight)
        up_native = native.debug_linear(pre_ff, up_weight)
        act_native = native.debug_gelu_tanh(gate_oracle)
        gate_up_native = native.debug_mul(act_oracle, up_oracle)
        down_native = native.debug_linear(gate_up_oracle, down_weight)
        torch.cuda.synchronize()
    finally:
        native.close()

    stages = {
        "gate_proj": {
            "contract": "F.linear(exact oracle pre_ff_norm, gate_proj.weight)",
            **metrics(gate_oracle, gate_native),
            "oracle_tap_consistency": metrics(taps["gate_proj"], gate_oracle),
        },
        "up_proj": {
            "contract": "F.linear(exact oracle pre_ff_norm, up_proj.weight)",
            **metrics(up_oracle, up_native),
            "oracle_tap_consistency": metrics(taps["up_proj"], up_oracle),
        },
        "activation": {
            "contract": "mlp.act_fn(exact oracle gate_proj)",
            **metrics(act_oracle, act_native),
            "oracle_tap_consistency": metrics(taps["mlp_act"], act_oracle),
        },
        "gate_times_up": {
            "contract": "exact oracle activation * exact oracle up_proj",
            **metrics(gate_up_oracle, gate_up_native),
            "oracle_tap_consistency": metrics(taps["mlp_gate_up"], gate_up_oracle),
        },
        "down_proj": {
            "contract": "F.linear(exact oracle mlp_gate_up, down_proj.weight)",
            **metrics(down_oracle, down_native),
            "oracle_tap_consistency": metrics(taps["down_proj"], down_oracle),
        },
    }
    first_nonzero = next(
        ({"stage": name, **stage} for name, stage in stages.items()
         if not stage["exact_element_stream"]),
        None,
    )
    first_formal = next(
        ({"stage": name, **stage} for name, stage in stages.items()
         if stage["mae"] > 1e-4 or stage["cosine"] < 0.9999),
        None,
    )
    tensor_artifact = {
        "layer": 2,
        "pre_ff_norm": pre_ff.detach().cpu(),
        "gate_input": pre_ff.detach().cpu(),
        "gate_oracle": gate_oracle.detach().cpu(),
        "up_oracle": up_oracle.detach().cpu(),
        "activation_oracle": act_oracle.detach().cpu(),
        "gate_up_oracle": gate_up_oracle.detach().cpu(),
        "down_oracle": down_oracle.detach().cpu(),
    }
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_artifact, args.tensor_output)
    report = {
        "status": "PASS" if first_formal is None else "NO-GO",
        "contract": {
            "torch_version": torch.__version__,
            "layer": 2,
            "input_source": "exact ShapeStable oracle taps from canonical prefix + deterministic gate frame",
            "dtype": "torch.bfloat16",
            "rows": 4,
            "intermediate_size": int(gate_oracle.shape[-1]),
            "native_linear": "run_linear_cublas, exact oracle input",
            "native_activation": "current kernel_gelu_tanh_bf16, exact oracle gate",
            "native_multiply": "current kernel_mul_bf16, exact oracle activation and up",
        },
        "first_nonzero_difference": first_nonzero,
        "first_formal_failure": first_formal,
        "stages": stages,
        "tensor_artifact": {
            "path": str(args.tensor_output),
            "sha256": hashlib.sha256(args.tensor_output.read_bytes()).hexdigest(),
        },
        "note": "No ATen linear bridge is selected; this gate determines whether the current cuBLAS/elementwise primitives independently diverge with exact inputs.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
