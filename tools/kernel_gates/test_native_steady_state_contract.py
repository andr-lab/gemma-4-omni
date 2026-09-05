#!/usr/bin/env python3
"""Smoke gate for the native step's no-sync/no-allocation steady state."""
from __future__ import annotations

import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT


def main() -> None:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    thinker = load_thinker("/workspace", device=device, dtype=dtype)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device))
    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

    ids, mask = backend._build_multimodal_inputs((), 1024, device=device)
    prefix_ids = ids[:, :AUDIO_START_AFTER_TEXT]
    prefix_attention = mask[:, :AUDIO_START_AFTER_TEXT].to(device=device, dtype=torch.int64).contiguous()
    prefix_embeds = lm.embed_tokens(prefix_ids).contiguous()
    prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds).contiguous()
    native.prime_async(prefix_embeds, prefix_ple, prefix_attention)
    torch.cuda.synchronize()

    frame_storage = torch.tensor(
        0.05 * np.random.default_rng(2026).standard_normal((1, 8, 1536, 2)),
        device=device,
        dtype=dtype,
    ).contiguous()
    frame = frame_storage[:, 2:6, :, 0]
    assert not frame.is_contiguous()
    for _ in range(10):
        native.step_async(frame)
    torch.cuda.synchronize()

    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    free_before, total_before = torch.cuda.mem_get_info()
    synchronize = torch.cuda.synchronize

    def forbidden_synchronize(*_args, **_kwargs):
        raise AssertionError("step_async ha invocato una synchronize Python")

    torch.cuda.synchronize = forbidden_synchronize
    try:
        for _ in range(100):
            native.step_async(frame)
    finally:
        torch.cuda.synchronize = synchronize
    synchronize()

    allocated_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()
    free_after, total_after = torch.cuda.mem_get_info()
    assert allocated_after == allocated_before, (allocated_before, allocated_after)
    assert reserved_after == reserved_before, (reserved_before, reserved_after)
    assert total_after == total_before
    assert free_after == free_before
    print({
        "steady_state": True,
        "frames": 100,
        "allocated_delta": allocated_after - allocated_before,
        "reserved_delta": reserved_after - reserved_before,
        "free_delta": free_after - free_before,
        "final_position": native.current_position,
    })


if __name__ == "__main__":
    main()
