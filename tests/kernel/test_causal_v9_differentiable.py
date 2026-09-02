#!/usr/bin/env python3
"""Test di verifica per il percorso di gradiente differenziabile di Causal V9."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.causal_v9_trainer import DifferentiableCausalQwenV9, compute_causal_v9_loss
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import load_v9_conditioning


def test_differentiable_graph() -> None:
    device = "cuda"
    dtype = torch.bfloat16
    print("[TEST] Loading V9 conditioning...", flush=True)
    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)

    print("[TEST] Loading Talker 1.7B...", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=False)

    print("[TEST] Instantiating DifferentiableCausalQwenV9...", flush=True)
    model = DifferentiableCausalQwenV9(projector, causal_conv, talker, train_conv=False)

    # Synthetic batch of 2 sequences of 64 frames
    B, T = 2, 64
    h_raw = torch.randn((B, T, 1536), device=device, dtype=dtype)
    h_tx = h_raw[:, 0, :].clone()
    codes = torch.randint(0, 2048, (B, T, 16), device=device, dtype=torch.long)

    print("[TEST] Running forward pass (L=0)...", flush=True)
    logits, h_pfx = model(h_raw, h_tx, codes, lookahead_frames=0)
    print(f"[TEST] Forward output: logits shape={logits.shape}, h_pfx shape={h_pfx.shape}", flush=True)

    assert logits.shape == (B, T, 3072)
    assert h_pfx.shape == (B, T, 2048)
    assert torch.isfinite(logits).all()

    loss, metrics = compute_causal_v9_loss(logits, codes)
    print(f"[TEST] Initial loss: {loss.item():.4f}, metrics={metrics}", flush=True)

    print("[TEST] Running backward pass...", flush=True)
    loss.backward()

    # Verify that V9 projector has gradients
    proj_grad = projector.audio_in.weight.grad
    assert proj_grad is not None, "projector.audio_in.weight.grad deve esistere!"
    assert not torch.isnan(proj_grad).any(), "Gradiente V9 non deve contenere NaN!"
    grad_norm = float(proj_grad.norm().item())
    print(f"[TEST] V9 projector audio_in grad norm: {grad_norm:.4f}", flush=True)
    assert grad_norm > 1e-6, "Gradiente V9 deve essere non-nullo!"

    # Verify that Qwen weights have NO gradients
    for name, param in talker.talker.named_parameters():
        assert param.grad is None, f"Parametro Qwen {name} non deve ricevere gradienti!"
    print("[TEST] All Qwen parameters confirmed completely frozen (zero gradients)!", flush=True)

    print(">>> DIFFERENTIABLE GRAPH TEST: 100% PASS <<<", flush=True)


if __name__ == "__main__":
    test_differentiable_graph()
