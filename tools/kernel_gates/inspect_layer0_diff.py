#!/usr/bin/env python3
"""Inspect layer 0 step by step to find exact line of divergence."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaListener, GemmaListenerState


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    listener = GemmaListener.from_thinker(th)
    state = GemmaListenerState(context_frames=64)
    listener.prime_steady_state(state)

    rng = np.random.default_rng(2026)
    pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)

    # Let's see what inputs are passed to forward
    backend = listener.backend
    mel, past = backend._extract_single_frame(pcm, None)
    # Replicate listener.step preparation:
    # 1. Audio Tower
    # In steady state:
    audio_feat = backend._static_features
    audio_soft = backend._graph_output.last_hidden_state if hasattr(backend._graph_output, "last_hidden_state") else backend._graph_output[0]
    
    # Let's inspect prime_steady_state:
    # Notice: prime_steady_state ran with ShapeStableGemmaListenerLM!
    # Does state.gemma_lm_cache have the same KV cache in both?
    # Check cache type
    print("state.gemma_lm_cache type:", type(state.gemma_lm_cache))

    # Now let's test:
    # What if we only replace _padded_linear?
    class LinearOnlyLM(ShapeStableGemmaListenerLM):
        def _padded_linear(self, linear, value):
            axis = self._token_axis(value)
            if value.shape[axis] == self._workspace.target_length:
                return F.linear(value, linear.weight, linear.bias)
            return super()._padded_linear(linear, value)

    # What if we only replace _padded_rmsnorm?
    class RMSNormOnlyLM(ShapeStableGemmaListenerLM):
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

    # Test LinearOnly
    l_base = GemmaListener.from_thinker(th)
    s_base = GemmaListenerState(context_frames=64)
    l_base.prime_steady_state(s_base)

    l_lin = GemmaListener.from_thinker(th)
    l_lin.backend._shape_stable_lm = LinearOnlyLM(lm, target_length=291)
    s_lin = GemmaListenerState(context_frames=64)
    l_lin.prime_steady_state(s_lin)

    out_base = l_base.step(pcm, None, s_base)
    out_lin = l_lin.step(pcm, None, s_lin)
    diff_lin = (out_base.gemma_hidden_t.float() - out_lin.gemma_hidden_t.float()).abs().max().item()
    print(f"LinearOnly diff: {diff_lin:.8e}")

    # Test RMSNormOnly
    l_rms = GemmaListener.from_thinker(th)
    l_rms.backend._shape_stable_lm = RMSNormOnlyLM(lm, target_length=291)
    s_rms = GemmaListenerState(context_frames=64)
    l_rms.prime_steady_state(s_rms)

    out_rms = l_rms.step(pcm, None, s_rms)
    diff_rms = (out_base.gemma_hidden_t.float() - out_rms.gemma_hidden_t.float()).abs().max().item()
    print(f"RMSNormOnly diff: {diff_rms:.8e}")


if __name__ == "__main__":
    main()
