#!/usr/bin/env python3
"""End-to-end GemmaListener native steady-state gate.

The gate intentionally exercises the full encode path after the 64-frame
rolling window is established, not only NativeGemmaListenerLM.step_async.
"""
from __future__ import annotations

import sys

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker


def main() -> None:
    device = torch.device("cuda:0")
    thinker = load_thinker("/workspace", device=device, dtype=torch.bfloat16)
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=True)
    listener = GemmaListener(backend=backend)
    state = GemmaListenerState(context_frames=64)
    frame = np.zeros(1280, dtype=np.float32)

    # Establish the fixed rolling graph and the persistent native LM session.
    for _ in range(65):
        output = listener.step(frame, None, state)
    torch.cuda.synchronize()

    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    original_synchronize = torch.cuda.synchronize

    def forbidden_synchronize(*_args, **_kwargs):
        raise AssertionError("GemmaListener native steady state ha invocato synchronize")

    torch.cuda.synchronize = forbidden_synchronize
    try:
        for _ in range(10):
            output = listener.step(frame, None, state)
    finally:
        torch.cuda.synchronize = original_synchronize
    original_synchronize()

    assert torch.cuda.memory_allocated() == allocated_before
    assert torch.cuda.memory_reserved() == reserved_before
    assert output.finite
    assert tuple(output.gemma_hidden_t.shape) == (1, 1536)
    assert state.gemma_lm_native is not None
    assert state.gemma_lm_native.current_position == 34 + state.next_frame_id * 4
    print(
        {
            "full_encode_steady_state": True,
            "frames": 10,
            "next_frame_id": state.next_frame_id,
            "native_position": state.gemma_lm_native.current_position,
            "allocated_delta": torch.cuda.memory_allocated() - allocated_before,
            "reserved_delta": torch.cuda.memory_reserved() - reserved_before,
            "finite": output.finite,
        }
    )


if __name__ == "__main__":
    main()
