#!/usr/bin/env python3
"""P4-NATIVE cache-import parity gate.

Builds the canonical prefix with ShapeStableGemmaListenerLM, imports its
DynamicCache into the native fixed cache using device-to-device copies, then
runs the first Q=4 step from the identical cache and frame input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
OUTPUT_PATH = Path("/workspace/validation/kernel_gates/P4_NATIVE/P4_NATIVE_PARITY_AFTER_CACHE_IMPORT.json")


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    cosine = torch.dot(ref, cand) / (torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand))
    return {
        "mae": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
        "cosine": float(cosine.item()),
    }


def reference_kv(past_key_values, shared_kv_states, lm, layer_idx: int):
    attention = lm.layers[layer_idx].self_attn
    if bool(getattr(attention, "is_kv_shared_layer", False)):
        source = int(attention.kv_shared_layer_index)
        return shared_kv_states[source]
    layer = past_key_values.layers[layer_idx]
    return layer.keys, layer.values


def main() -> int:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    thinker = load_thinker("/workspace", device=device, dtype=dtype)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    shape_stable = ShapeStableGemmaListenerLM(
        lm, target_length=291, stream_length=4
    )

    input_ids, attention = backend._build_multimodal_inputs((), 1024, device=device)
    prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
    prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT].to(
        device=device, dtype=torch.long
    )
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
    past = prefix_output.past_key_values
    shared = getattr(past, "shared_layers", {})

    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    native.reset()
    native.import_cache_async(past, current_position=AUDIO_START_AFTER_TEXT)
    torch.cuda.synchronize()

    cache_rows = []
    cache_exact = True
    for layer_idx in range(len(lm.layers)):
        reference_k, reference_v = reference_kv(past, shared, lm, layer_idx)
        native_k, native_v = native.get_cached_kv(layer_idx)
        k_metrics = metrics(reference_k, native_k)
        v_metrics = metrics(reference_v, native_v)
        exact = bool(torch.equal(reference_k, native_k) and torch.equal(reference_v, native_v))
        cache_exact = cache_exact and exact
        cache_rows.append({
            "layer": layer_idx,
            "exact": exact,
            "k": k_metrics,
            "v": v_metrics,
        })

    rng = np.random.default_rng(2026)
    frame = torch.tensor(
        0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype
    ).contiguous()
    frame_ple = backend._frame_pad_per_layer_inputs(lm)
    positions = torch.arange(
        AUDIO_START_AFTER_TEXT,
        AUDIO_START_AFTER_TEXT + 4,
        device=device,
        dtype=torch.long,
    ).view(1, -1)
    step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
    step_attention[:, :AUDIO_START_AFTER_TEXT] = prefix_attention
    step_masks = create_masks_for_generate(
        lm.config,
        frame,
        step_attention,
        past_key_values=past,
        position_ids=positions,
    )
    with torch.inference_mode():
        reference = shape_stable.forward(
            inputs_embeds=frame,
            per_layer_inputs=frame_ple,
            attention_mask=step_masks,
            position_ids=positions,
            past_key_values=past,
            shared_kv_states=shared,
        ).last_hidden_state
    candidate = native.step_async(frame)
    torch.cuda.synchronize()
    step_metrics = metrics(reference, candidate)
    native.verify_output_health()

    result = {
        "git_sha": "7f116cea3048ab87954f0292ba9809276bc15f27",
        "prefix_len": AUDIO_START_AFTER_TEXT,
        "native_position": native.current_position,
        "cache_import": {
            "exact": cache_exact,
            "max_k_mae": max(row["k"]["mae"] for row in cache_rows),
            "max_v_mae": max(row["v"]["mae"] for row in cache_rows),
            "rows": cache_rows,
        },
        "step_from_identical_cache": step_metrics,
        "strict_thresholds": {
            "mae_max": STRICT_MAE_MAX,
            "cosine_min": STRICT_COSINE_MIN,
        },
        "status": "PASS" if cache_exact and step_metrics["mae"] <= STRICT_MAE_MAX and step_metrics["cosine"] >= STRICT_COSINE_MIN else "NO-GO",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cache_import_exact": cache_exact,
        "native_position": native.current_position,
        "step": step_metrics,
        "status": result["status"],
        "report": str(OUTPUT_PATH),
    }, indent=2))
    native.close()
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
