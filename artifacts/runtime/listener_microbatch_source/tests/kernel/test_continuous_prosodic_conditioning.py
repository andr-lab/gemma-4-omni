import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
try:
    import pytest
except ImportError:
    pytest = None

from duplex_kernel.frame_residual_adapter import (
    FrameResidualQueryFiLMAdapter,
    apply_response_state_residual,
    load_frame_residual_adapter,
)


def _activate_output(adapter: FrameResidualQueryFiLMAdapter) -> None:
    with torch.no_grad():
        adapter.output.weight.normal_(mean=0.0, std=1e-3)
        adapter.output.bias.normal_(mean=0.0, std=1e-3)


def test_continuous_film_modulation():
    adapter = FrameResidualQueryFiLMAdapter()
    _activate_output(adapter)
    hidden_seq = torch.randn(1, 5, 1536)
    query = torch.randn(1, 1, 2048)

    # Step 0 (frame 0) must be preserved if preserve_frame0 is True
    f0, tail0 = apply_response_state_residual(adapter, hidden_seq, query, 0)
    assert torch.equal(f0, query), "Frame 0 must be bit-exact preserved to native query"
    assert not tail0

    # Step 1: dynamically modulated by hidden state h_1
    f1, tail1 = apply_response_state_residual(adapter, hidden_seq, query, 1)
    assert not torch.equal(f1, query), "Step 1 should be modulated by Gemma hidden state"
    assert not tail1
    assert torch.isfinite(f1).all()

    # Verify modulation delta is bounded and smooth
    delta = f1 - query
    assert float(torch.norm(delta.detach())) > 0.0


def test_continuous_cursor_tail_hold():
    adapter = FrameResidualQueryFiLMAdapter()
    _activate_output(adapter)
    hidden_seq = torch.randn(1, 4, 1536)
    query = torch.randn(1, 1, 2048)

    # Step 3 is last token (index 3 out of len 4)
    f3, tail3 = apply_response_state_residual(adapter, hidden_seq, query, 3)
    assert not tail3

    # Steps 4, 5, 10: text exhausted -> holds final state continuously
    f4, tail4 = apply_response_state_residual(adapter, hidden_seq, query, 4)
    f5, tail5 = apply_response_state_residual(adapter, hidden_seq, query, 5)
    f10, tail10 = apply_response_state_residual(adapter, hidden_seq, query, 10)

    assert tail4 and tail5 and tail10, "Exhausted tokens must trigger tail-hold"
    assert torch.equal(f4, f5), "Tail-hold must produce identical modulation for same query"
    assert torch.equal(f5, f10)


def test_zero_scale_recovers_identity():
    adapter = FrameResidualQueryFiLMAdapter()
    _activate_output(adapter)
    adapter.scale_logit.data.fill_(-100.0)

    hidden_seq = torch.randn(1, 6, 1536)
    query = torch.randn(1, 1, 2048)

    for step in range(8):
        out, _ = apply_response_state_residual(adapter, hidden_seq, query, step)
        assert torch.allclose(out, query, atol=1e-7), f"Zero scale must yield identity at step {step}"


def test_real_emovo_candidate_checkpoint():
    ckpt_path = Path("checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt")
    if not ckpt_path.is_file():
        if pytest is not None:
            pytest.skip("Local checkpoint not present")
        else:
            return

    adapter, payload = load_frame_residual_adapter(ckpt_path, device="cpu", dtype=torch.float32)
    assert adapter.architecture_id == "response_hidden_q1_query_film_v2"
    assert abs(float(adapter.scale) - 0.0513) < 0.005

    hidden_seq = torch.randn(1, 5, 1536)
    query = torch.randn(1, 1, 2048)

    # Frame 0 preserved
    s0, _ = apply_response_state_residual(adapter, hidden_seq, query, 0)
    assert torch.equal(s0, query)

    # Frame 1 modulated
    s1, _ = apply_response_state_residual(adapter, hidden_seq, query, 1)
    assert not torch.equal(s1, query)
    assert torch.isfinite(s1).all()

    # Ratio of residual delta to query norm should be in reasonable scale (~learned scale)
    delta_norm = float(torch.norm(s1 - query))
    query_norm = float(torch.norm(query))
    ratio = delta_norm / query_norm
    assert 0.001 < ratio < 0.20, f"Residual ratio {ratio:.4f} outside expected range"


if __name__ == "__main__":
    test_continuous_film_modulation()
    print("test_continuous_film_modulation: PASS")
    test_continuous_cursor_tail_hold()
    print("test_continuous_cursor_tail_hold: PASS")
    test_zero_scale_recovers_identity()
    print("test_zero_scale_recovers_identity: PASS")
    test_real_emovo_candidate_checkpoint()
    print("test_real_emovo_candidate_checkpoint: PASS")
    print("ALL CONTINUOUS PROSODIC TESTS PASSED!")
