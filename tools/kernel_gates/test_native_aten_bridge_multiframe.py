#!/usr/bin/env python3
"""Validate the opt-in ATen attention bridge through full native layers/frames."""
from __future__ import annotations

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
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

STRICT_MAE_MAX = 1e-4
STRICT_COSINE_MIN = 0.9999
REPORT_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_ATEN_BRIDGE_MULTIFRAME.json")
TENSOR_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_ATEN_BRIDGE_MULTIFRAME_TENSORS.pt")
EXPECTED_TORCH_VERSION = "2.12.1+cu130"


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
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


def main() -> int:
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"ATen bridge gate richiede PyTorch {EXPECTED_TORCH_VERSION}, "
            f"trovato {torch.__version__}"
        )
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
        oracle_prefix = shape_stable.forward(
            inputs_embeds=prefix_embeds,
            per_layer_inputs=prefix_ple,
            attention_mask=prefix_masks,
            position_ids=prefix_positions,
            past_key_values=None,
            shared_kv_states=None,
        )
    oracle_cache = oracle_prefix.past_key_values

    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    rng = np.random.default_rng(20260905)
    try:
        native.prime_async(prefix_embeds, prefix_ple, prefix_attention)
        for frame_id in range(frames):
            frame = torch.tensor(
                0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype
            ).contiguous()
            current_position = AUDIO_START_AFTER_TEXT + frame_id * 4
            positions = torch.arange(current_position, current_position + 4, device=device, dtype=torch.long).view(1, -1)
            step_attention = torch.ones((1, current_position + 4), device=device, dtype=torch.long)
            step_attention[:, :AUDIO_START_AFTER_TEXT] = prefix_attention
            step_masks = create_masks_for_generate(
                lm.config, frame, step_attention,
                past_key_values=oracle_cache, position_ids=positions,
            )
            with torch.inference_mode():
                oracle_step = shape_stable.forward(
                    inputs_embeds=frame,
                    per_layer_inputs=backend._frame_pad_per_layer_inputs(lm),
                    attention_mask=step_masks,
                    position_ids=positions,
                    past_key_values=oracle_cache,
                    shared_kv_states=getattr(oracle_cache, "shared_layers", None),
                ).last_hidden_state
            with torch.inference_mode():
                native_step = native.step_debug(
                    frame, num_layers=35, apply_final_norm=True
                ).detach().clone()
            torch.cuda.synchronize()
            row = {"frame": frame_id, "operation": "layer_output", **metrics(oracle_step, native_step)}
            rows.append(row)
            tensors[f"frame_{frame_id}/oracle"] = oracle_step.detach().cpu()
            tensors[f"frame_{frame_id}/native"] = native_step.detach().cpu()
            print("[ATEN BRIDGE FRAME]", json.dumps(row, sort_keys=True), flush=True)
    finally:
        native.close()

    torch.save(tensors, TENSOR_PATH)
    tensor_sha256 = hashlib.sha256(TENSOR_PATH.read_bytes()).hexdigest()
    first_nonzero = next((row for row in rows if not row["exact_element_stream"]), None)
    first_mae_gt_1e_7 = next((row for row in rows if row["mae"] > 1e-7), None)
    first_mae_gt_1e_6 = next((row for row in rows if row["mae"] > 1e-6), None)
    first_formal_failure = next(
        (row for row in rows if row["mae"] > STRICT_MAE_MAX or row["cosine"] < STRICT_COSINE_MIN),
        None,
    )
    result = {
        "status": "PASS" if first_formal_failure is None else "NO-GO",
        "torch_version": torch.__version__,
        "attention_backend": "public ATen SDPA bridge (GEMMA4_ATEN_SDPA_BRIDGE=1)",
        "rmsnorm_backend": (
            "literal ATen pre_feedforward_layernorm bridge (GEMMA4_ATEN_RMSNORM_BRIDGE=1)"
            if os.environ.get("GEMMA4_ATEN_RMSNORM_BRIDGE", "0") != "0"
            else "custom RMSNorm kernels (GEMMA4_ATEN_RMSNORM_BRIDGE unset/0)"
        ),
        "frames_checked": len(rows),
        "strict_thresholds": {"mae_max": STRICT_MAE_MAX, "cosine_min": STRICT_COSINE_MIN},
        "rows": rows,
        "first_divergence_markers": {
            "first_nonzero_difference": first_nonzero,
            "first_mae_gt_1e-7": first_mae_gt_1e_7,
            "first_mae_gt_1e-6": first_mae_gt_1e_6,
            "first_formal_failure": first_formal_failure,
        },
        "tensor_artifact": {"path": str(TENSOR_PATH), "sha256": tensor_sha256},
        "note": "This validates the opt-in bridge path only; it does not certify native production parity.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
