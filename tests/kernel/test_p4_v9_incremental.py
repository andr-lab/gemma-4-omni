from __future__ import annotations

import pytest
import torch

from duplex_kernel.models.v9 import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    IncrementalV9Projector,
)


def _tiny_stream() -> tuple[DeepCausalTransformerProjectorV9, CausalConv1D, IncrementalV9Projector]:
    torch.manual_seed(17)
    projector = DeepCausalTransformerProjectorV9(
        raw_dim=8,
        text_dim=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
    ).eval()
    causal_conv = CausalConv1D(hidden_size=8, kernel_size=3).eval()
    return projector, causal_conv, IncrementalV9Projector(projector, causal_conv)


def test_incremental_v9_matches_causal_batch_forward_in_fp32() -> None:
    projector, causal_conv, stream = _tiny_stream()
    raw = torch.randn(2, 7, 8)
    anchor = torch.randn(2, 8)

    batch = projector(causal_conv(raw), anchor)
    state = stream.new_state(batch_size=2, device="cpu", dtype=torch.float32, max_seq_len=16)
    incremental = torch.stack([stream.step(raw[:, index], anchor, state) for index in range(raw.shape[1])], dim=1)

    torch.testing.assert_close(incremental, batch, rtol=2e-5, atol=2e-5)
    assert state.frames_processed == raw.shape[1]
    assert state.rope_position == raw.shape[1]


def test_incremental_v9_prefix_is_independent_of_future_frames() -> None:
    _, _, stream = _tiny_stream()
    prefix = torch.randn(1, 3, 8)
    future_a = torch.randn(1, 4, 8)
    future_b = torch.randn(1, 4, 8)
    anchor = torch.randn(1, 8)

    def run(sequence: torch.Tensor) -> torch.Tensor:
        state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=16)
        return torch.stack([stream.step(sequence[:, i], anchor, state) for i in range(sequence.shape[1])], dim=1)

    output_a = run(torch.cat((prefix, future_a), dim=1))
    output_b = run(torch.cat((prefix, future_b), dim=1))
    torch.testing.assert_close(output_a[:, :3], output_b[:, :3], rtol=0.0, atol=1e-6)


def test_incremental_v9_requires_a_fixed_text_anchor_and_supports_reset() -> None:
    _, _, stream = _tiny_stream()
    state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=4)
    raw = torch.zeros(1, 8)
    anchor = torch.ones(1, 8)

    stream.step(raw, anchor, state)
    with pytest.raises(ValueError, match="anchor"):
        stream.step(raw, anchor + 1.0, state)
    assert state.frames_processed == 1

    state.reset()
    assert state.frames_processed == 0
    assert state.rope_position == 0
    stream.step(raw, anchor, state)
    assert state.frames_processed == 1


def test_incremental_v9_rejects_bad_shapes_and_sequence_overrun() -> None:
    _, _, stream = _tiny_stream()
    state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=1)
    with pytest.raises(ValueError, match="1536|raw"):
        stream.step(torch.zeros(1, 7), torch.zeros(1, 8), state)
    stream.step(torch.zeros(1, 8), torch.zeros(1, 8), state)
    with pytest.raises(OverflowError, match="max_seq_len"):
        stream.step(torch.zeros(1, 8), torch.zeros(1, 8), state)
