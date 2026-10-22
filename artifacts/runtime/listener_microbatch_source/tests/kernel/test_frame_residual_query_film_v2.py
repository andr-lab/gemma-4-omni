import tempfile
from pathlib import Path

import torch
import pytest

from duplex_kernel.frame_residual_adapter import (
    FrameResidualAdapter,
    FrameResidualQueryFiLMAdapter,
    apply_response_state_residual,
    load_frame_residual_adapter,
)
def _artifact_payload(
    adapter: FrameResidualQueryFiLMAdapter,
    train: tuple = (),
    validation: tuple = (),
    initial_checkpoint: str | None = None,
    result: dict | None = None,
) -> dict:
    return {
        "schema_version": adapter.schema_version,
        "architecture": adapter.architecture,
        "architecture_id": adapter.architecture_id,
        "state_dict": adapter.state_dict(),
        "training_contract": {
            "query_conditioning": "exact current frozen-Qwen base audio-query embedding",
        },
        "result": result or {},
    }


def _activate_output(adapter: FrameResidualQueryFiLMAdapter) -> None:
    with torch.no_grad():
        adapter.output.weight.normal_(mean=0.0, std=1e-3)
        adapter.output.bias.normal_(mean=0.0, std=1e-3)


def test_v2_zero_output_is_exact_qwen_baseline_and_has_bounded_scale():
    torch.manual_seed(101)
    adapter = FrameResidualQueryFiLMAdapter()
    hidden = torch.randn(1, 4, 1536)
    query = torch.randn(1, 4, 2048)

    assert torch.equal(adapter(hidden, query), query)
    assert float(adapter.scale.detach()) == pytest.approx(0.05, abs=1e-6)
    with torch.no_grad():
        adapter.scale_logit.fill_(12.0)
    assert 0.0 < float(adapter.scale.detach()) < 0.25


def test_v2_query_changes_film_residual_and_both_inputs_receive_gradient():
    torch.manual_seed(102)
    adapter = FrameResidualQueryFiLMAdapter()
    _activate_output(adapter)
    hidden = torch.randn(1, 3, 1536, requires_grad=True)
    query = torch.randn(1, 3, 2048, requires_grad=True)
    other_query = torch.randn(1, 3, 2048)

    residual = adapter.residual(hidden, query)
    other_residual = adapter.residual(hidden.detach(), other_query)
    assert residual.shape == query.shape
    assert torch.isfinite(residual).all()
    assert not torch.equal(residual, other_residual)
    base_rms = torch.sqrt(query.detach().float().square().mean(dim=-1))
    residual_rms = torch.sqrt(residual.float().square().mean(dim=-1))
    assert torch.all(residual_rms <= adapter.scale.detach() * base_rms + 1e-6)

    residual.float().square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert float(hidden.grad.norm()) > 0.0
    assert float(query.grad.norm()) > 0.0


def test_v2_runtime_keeps_frame_zero_plain_but_conditions_later_q1_queries():
    torch.manual_seed(103)
    adapter = FrameResidualQueryFiLMAdapter()
    _activate_output(adapter)
    hidden = torch.randn(1, 2, 1536)
    query = torch.randn(1, 1, 2048)

    frame_zero, zero_tail = apply_response_state_residual(adapter, hidden, query, 0)
    frame_one, one_tail = apply_response_state_residual(adapter, hidden, query, 1)

    assert torch.equal(frame_zero, query)
    assert not zero_tail and not one_tail
    assert not torch.equal(frame_one, query)


def test_loader_keeps_v1_rollback_and_loads_only_matching_v2_schema():
    contract = {"runtime_route": "plain native Qwen Q1, unchanged text/BOS/positions/KV"}
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        for adapter in (FrameResidualAdapter(), FrameResidualQueryFiLMAdapter()):
            payload = {
                "schema_version": adapter.schema_version,
                "architecture": adapter.architecture,
                "architecture_id": adapter.architecture_id,
                "state_dict": adapter.state_dict(),
                "training_contract": contract,
            }
            path = temp_path / f"schema{adapter.schema_version}.pt"
            torch.save(payload, path)
            loaded, loaded_payload = load_frame_residual_adapter(
                path, device="cpu", dtype=torch.float32
            )
            assert type(loaded) is type(adapter)
            assert loaded_payload["architecture_id"] == adapter.architecture_id
            assert not loaded.training
            assert all(not parameter.requires_grad for parameter in loaded.parameters())

        v2_payload = _artifact_payload(
            FrameResidualQueryFiLMAdapter(), train=(), validation=(),
            initial_checkpoint=None,
            result={"history_alignment": "dtw", "query_residual_distill_weight": 1.0},
        )
        assert v2_payload["schema_version"] == 2
        assert v2_payload["architecture_id"] == "response_hidden_q1_query_film_v2"
        assert "exact current frozen-Qwen base audio-query embedding" in v2_payload["training_contract"]["query_conditioning"]

        bad_payload = {
            "schema_version": 2,
            "architecture": FrameResidualQueryFiLMAdapter.architecture,
            "architecture_id": FrameResidualQueryFiLMAdapter.architecture_id,
            "state_dict": FrameResidualAdapter().state_dict(),
            "training_contract": contract,
        }
        bad_path = temp_path / "mismatched_v2.pt"
        torch.save(bad_payload, bad_path)
        with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
            load_frame_residual_adapter(bad_path, device="cpu", dtype=torch.float32)
