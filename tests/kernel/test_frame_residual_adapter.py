import torch

from duplex_kernel.frame_residual_adapter import (
    FrameResidualAdapter,
    apply_response_state_residual,
    response_state_at_cursor,
)


def test_zero_initialized_adapter_is_exact_qwen_baseline():
    torch.manual_seed(31)
    adapter = FrameResidualAdapter(input_dim=1536, output_dim=2048, bottleneck_dim=256)
    hidden = torch.randn(1, 5, 1536)
    qwen_audio_input = torch.randn(1, 5, 2048)

    adapted = adapter(hidden, qwen_audio_input)

    assert torch.equal(adapted, qwen_audio_input)
    assert adapter.scale.item() == pytest.approx(0.05, abs=1e-6)


def test_residual_shape_scale_bound_and_projector_gradient():
    adapter = FrameResidualAdapter(input_dim=1536, output_dim=2048, bottleneck_dim=256)
    with torch.no_grad():
        adapter.output.weight.normal_(mean=0.0, std=0.01)
        adapter.output.bias.normal_(mean=0.0, std=0.01)
        adapter.scale_logit.fill_(12.0)
    hidden = torch.randn(1, 4, 1536, requires_grad=False)
    qwen_audio_input = torch.randn(1, 4, 2048, requires_grad=True)

    adapted = adapter(hidden, qwen_audio_input)
    loss = adapted.float().square().mean()
    loss.backward()

    assert adapted.shape == qwen_audio_input.shape
    assert torch.isfinite(adapted).all()
    assert 0.0 < float(adapter.scale.detach()) < adapter.max_scale
    assert adapter.output.weight.grad is not None
    assert torch.isfinite(adapter.output.weight.grad).all()
    assert float(adapter.output.weight.grad.norm()) > 0.0
    assert qwen_audio_input.grad is not None


def test_adapter_rejects_hidden_or_audio_frame_mismatch():
    adapter = FrameResidualAdapter(input_dim=1536, output_dim=2048, bottleneck_dim=256)
    hidden = torch.randn(1, 3, 1536)
    qwen_audio_input = torch.randn(1, 4, 2048)

    with pytest.raises(ValueError, match="batch/frame dimensions"):
        adapter(hidden, qwen_audio_input)


def test_response_state_follows_qwen_cursor_and_holds_only_after_text():
    hidden = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    first, first_tail = response_state_at_cursor(hidden, 0)
    last, last_tail = response_state_at_cursor(hidden, 3)
    held, held_tail = response_state_at_cursor(hidden, 7)
    assert first.tolist() == [[[0.0]]]
    assert not first_tail
    assert last.tolist() == [[[3.0]]]
    assert not last_tail
    assert held.tolist() == [[[3.0]]]
    assert held_tail


def test_zero_adapter_adds_no_qwen_query_residual_and_reports_tail_hold():
    adapter = FrameResidualAdapter()
    hidden = torch.randn(1, 2, 1536)
    base = torch.randn(1, 1, 2048)
    result, tail_hold = apply_response_state_residual(adapter, hidden, base, 4)
    torch.testing.assert_close(result, base, rtol=0, atol=0)
    assert tail_hold


import pytest
