#!/usr/bin/env python3
"""Benchmark LinearOnly fast dispatch on RTX 3090."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


class LinearFastShapeStableLM(ShapeStableGemmaListenerLM):
    def _padded_linear(self, linear, value):
        axis = self._token_axis(value)
        if value.shape[axis] == self._workspace.target_length:
            return F.linear(value, linear.weight, linear.bias)
        return super()._padded_linear(linear, value)


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[BENCHMARK] LinearFastShapeStableLM Latency & Parity")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    listener = GemmaListener.from_thinker(th)
    listener.backend._shape_stable_lm = LinearFastShapeStableLM(
        th.model.model.language_model, target_length=291
    )
    state = GemmaListenerState(context_frames=64)
    listener.prime_steady_state(state)

    rng = np.random.default_rng(2026)
    pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(50)]

    # Warmup
    for pcm in pcm_frames[:10]:
        listener.step(pcm, None, state)
    torch.cuda.synchronize()

    times = []
    for pcm in pcm_frames:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        listener.step(pcm, None, state)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print(f"Listener Latency with LinearFast (50 frames):")
    print(f"  p50:  {np.percentile(t, 50):.2f} ms")
    print(f"  p95:  {np.percentile(t, 95):.2f} ms")
    print(f"  mean: {np.mean(t):.2f} ms")
    print(f"  min:  {np.min(t):.2f} ms")
    print(f"  max:  {np.max(t):.2f} ms")


if __name__ == "__main__":
    main()
