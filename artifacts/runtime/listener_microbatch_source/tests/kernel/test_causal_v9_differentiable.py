#!/usr/bin/env python3
"""Test di verifica per il percorso di gradiente differenziabile di Causal V9."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
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
    if not torch.cuda.is_available():
        pytest.skip("GPU CUDA necessaria per il test differenziabile")
    configured_checkpoint = Path("/workspace/checkpoints/v9_causal_l0_final.pt")
    if not configured_checkpoint.is_file():
        pytest.skip("checkpoint causale-L0 configurato non disponibile")
    print("[TEST] Loading V9 conditioning...", flush=True)
    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)

    print("[TEST] Loading Talker 1.7B...", flush=True)
    speaker_reference_wav = Path(
        "/workspace/reports/p7/assets/qwen3_tts_official_clone.wav"
    )
    if not speaker_reference_wav.is_file():
        pytest.skip("canonical P7 speaker reference non disponibile")
    talker = load_talker(
        "/workspace",
        device=device,
        load_code2wav=False,
        speaker_reference_wav=str(speaker_reference_wav),
        speaker_reference_sha256=(
            "480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c"
        ),
    )

    print("[TEST] Instantiating DifferentiableCausalQwenV9...", flush=True)
    model = DifferentiableCausalQwenV9(
        projector,
        causal_conv,
        talker,
        train_conv=False,
        voice_conditioning=talker.default_voice_conditioning,
    )

    # Synthetic batch of 2 sequences of 64 frames
    B, T = 2, 64
    h_raw = torch.randn((B, T, 1536), device=device, dtype=dtype)
    h_tx = h_raw[:, 0, :].clone()
    codes = torch.randint(0, 2048, (B, T, 16), device=device, dtype=torch.long)

    texts = [
        "Questa è una frase di test."
        for _ in range(B)
    ]

    print("[TEST] Running forward pass (L=0)...", flush=True)
    logits, h_pfx = model(
        h_raw,
        h_tx,
        codes,
        texts,
        lookahead_frames=0,
    )
    print(
        f"[TEST] Forward output: {len(logits)} samples, "
        f"h_pfx shape={h_pfx.shape}",
        flush=True,
    )

    assert len(logits) == B
    for sample_logits in logits:
        assert sample_logits.shape == (1, T + 1, 3072)
        assert torch.isfinite(sample_logits).all()
    assert h_pfx.shape == (B, T, 2048)

    loss, metrics = compute_causal_v9_loss(
        logits,
        codes,
        eos_token_id=model.codec_eos_id,
        terminal_mask=torch.ones(B, dtype=torch.bool),
    )
    assert "eos_ce_loss" in metrics
    assert "eos_top1_acc" in metrics
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
