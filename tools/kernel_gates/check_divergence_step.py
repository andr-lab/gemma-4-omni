#!/usr/bin/env python3
"""Check frame-by-frame divergence between baseline and FastShapeStable."""
from __future__ import annotations

import sys
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


class FastShapeStableGemmaListenerLM(ShapeStableGemmaListenerLM):
    def _padded_linear(self, linear, value):
        axis = self._token_axis(value)
        if value.shape[axis] == self._workspace.target_length:
            return F.linear(value, linear.weight, linear.bias)
        return super()._padded_linear(linear, value)

    def _padded_rmsnorm(self, norm, value):
        axis = self._token_axis(value)
        weight = getattr(norm, "weight", None)
        eps = float(norm.eps)
        norm_shape = (value.shape[-1],)
        if value.shape[axis] == self._workspace.target_length:
            return F.rms_norm(value, norm_shape, weight, eps)
        padded = self._workspace.padded(value, axis=axis)
        out = F.rms_norm(padded, norm_shape, weight, eps)
        if padded is value:
            return out
        original_length = int(value.shape[axis])
        slices = [slice(None)] * out.ndim
        slices[axis] = slice(0, original_length)
        return out[tuple(slices)]


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    th = load_thinker("/workspace", device=device, dtype=dtype)
    
    listener_orig = GemmaListener.from_thinker(th)
    state_orig = GemmaListenerState(context_frames=64)
    listener_orig.prime_steady_state(state_orig)

    listener_fast = GemmaListener.from_thinker(th)
    listener_fast.backend._shape_stable_lm = FastShapeStableGemmaListenerLM(
        th.model.model.language_model, target_length=291
    )
    state_fast = GemmaListenerState(context_frames=64)
    listener_fast.prime_steady_state(state_fast)

    rng = np.random.default_rng(2026)
    for i in range(10):
        pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
        out_orig = listener_orig.step(pcm, None, state_orig)
        out_fast = listener_fast.step(pcm, None, state_fast)
        h_orig = out_orig.gemma_hidden_t.float()
        h_fast = out_fast.gemma_hidden_t.float()
        diff = (h_orig - h_fast).abs().max().item()
        cos = F.cosine_similarity(h_orig.view(1, -1), h_fast.view(1, -1)).item()
        print(f"Frame {i}: max_diff = {diff:.8e}, cosine = {cos:.8f}")
        if diff > 1e-4:
            break

if __name__ == "__main__":
    main()
