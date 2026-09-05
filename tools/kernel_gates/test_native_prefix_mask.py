#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical prefix-mask parity gate for the native listener.

This gate uses the exact stable prefix and attention mask produced by the
realtime backend, not an all-ones diagnostic mask.
"""
from __future__ import annotations

import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM


STRICT_MAE_MAX = 1e-4
STRICT_COSINE_MIN = 0.9999


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    delta = (reference.float() - candidate.float()).abs()
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    cosine = torch.dot(ref, cand) / (torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand))
    return float(delta.mean().item()), float(cosine.item()), float(delta.max().item())


def main() -> int:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    thinker = load_thinker("/workspace", device=device, dtype=dtype)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device))
    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

    input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
    stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
    stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(dtype=torch.int64).contiguous()
    with torch.inference_mode():
        stable_embeds = lm.embed_tokens(stable_ids).contiguous()
        stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds).contiguous()

    state = GemmaListenerState()
    backend.prime_steady_state(state)
    native.reset()
    native.prime_async(stable_embeds, stable_ple, stable_attention)
    torch.cuda.synchronize()

    rng = np.random.default_rng(2026)
    audio_soft = torch.tensor(
        0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype
    )
    step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
    step_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
    reference = backend._forward_incremental_lm(
        embeds=audio_soft,
        per_layer=backend._frame_pad_per_layer_inputs(lm),
        attention=step_attention,
        positions=torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device),
        state=state,
    )
    candidate = native.step_async(audio_soft)
    torch.cuda.synchronize()

    mae, cosine, max_abs = metrics(reference, candidate)
    cache_metrics = {}
    for layer_idx in (13, 14):
        py_layer = state.gemma_lm_cache.layers[layer_idx]
        py_k, py_v = py_layer.keys, py_layer.values
        native_k, native_v = native.get_cached_kv(layer_idx)
        cache_metrics[str(layer_idx)] = {
            "k_mae": metrics(py_k, native_k)[0],
            "k_cosine": metrics(py_k, native_k)[1],
            "v_mae": metrics(py_v, native_v)[0],
            "v_cosine": metrics(py_v, native_v)[1],
        }
    result = {
        "prefix_len": AUDIO_START_AFTER_TEXT,
        "prefix_attention": stable_attention[0].tolist(),
        "native_position": native.current_position,
        "mae": mae,
        "cosine": cosine,
        "max_abs": max_abs,
        "cache_metrics": cache_metrics,
    }
    print(result)
    if mae > STRICT_MAE_MAX or cosine < STRICT_COSINE_MIN:
        print(
            f"[NO-GO] strict parity fallisce: mae={mae:.6e} (max {STRICT_MAE_MAX:.1e}), "
            f"cosine={cosine:.8f} (min {STRICT_COSINE_MIN:.4f})",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
