from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM, _StaticPaddingWorkspace


class _TinyLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.layers = torch.nn.ModuleList()
        self.proj = torch.nn.Linear(8, 8, bias=False)


def test_shape_stable_listener_keeps_the_original_language_model_and_weight_storage() -> None:
    language_model = _TinyLanguageModel()
    before = {name: parameter.data_ptr() for name, parameter in language_model.named_parameters()}

    listener = ShapeStableGemmaListenerLM(language_model, target_length=16)

    assert listener.language_model is language_model
    report = listener.weight_sharing_report()
    assert report["listener_parameter_count"] == 0
    assert report["shared_weight_storage"] is True
    assert report["source_parameter_data_ptrs"] == before


def test_static_workspace_keeps_simultaneously_live_key_and_value_buffers_disjoint() -> None:
    workspace = _StaticPaddingWorkspace(target_length=4)
    key = torch.ones((1, 2, 1, 3), dtype=torch.bfloat16)
    value = torch.full_like(key, 2)

    padded_key = workspace.padded(key, axis=-2, slot="attention_key")
    padded_value = workspace.padded(value, axis=-2, slot="attention_value")

    assert padded_key.data_ptr() != padded_value.data_ptr()
    assert torch.equal(padded_key[..., :1, :], key)
    assert torch.equal(padded_value[..., :1, :], value)


@pytest.mark.parametrize("mask_kind", ["boolean", "additive"])
def test_static_attention_mask_clears_stale_tail_after_long_to_short_refresh(mask_kind: str) -> None:
    workspace = _StaticPaddingWorkspace(target_length=8)
    query = torch.tensor([[[[0.2, -0.3, 0.7, 0.1]], [[-0.1, 0.4, 0.3, -0.5]]]])
    long_key = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 4) / 7
    long_value = torch.flip(long_key, dims=(-1,))
    short_key = long_key[..., :3, :].contiguous()
    short_value = long_value[..., :3, :].contiguous()

    def mask(length: int) -> torch.Tensor:
        if mask_kind == "boolean":
            return torch.ones((1, 1, 1, length), dtype=torch.bool)
        return torch.zeros((1, 1, 1, length), dtype=torch.float32)

    def run(target: _StaticPaddingWorkspace, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        key_padded = target.padded(key, axis=-2, slot="attention_key")
        value_padded = target.padded(value, axis=-2, slot="attention_value")
        attention_mask = target.padded_mask(mask(int(key.shape[-2])), key_length=int(key.shape[-2]))
        return torch.nn.functional.scaled_dot_product_attention(
            query, key_padded, value_padded, attn_mask=attention_mask,
        )

    # Prime the reused buffers with a longer, all-valid cache before the short
    # prefix call; the latter must mask every stale K/V tail element.
    run(workspace, long_key, long_value)
    refreshed = run(workspace, short_key, short_value)
    fresh = run(_StaticPaddingWorkspace(target_length=8), short_key, short_value)

    assert torch.equal(refreshed, fresh)


def test_production_listener_sources_do_not_mutate_module_forwards_or_attention_registries() -> None:
    root = Path(__file__).resolve().parents[2]
    production_sources = (
        root / "duplex_kernel" / "listener.py",
        root / "duplex_kernel" / "shape_stable_listener.py",
    )
    forbidden = (
        "module.forward =",
        "ALL_ATTENTION_FUNCTIONS[",
        "ALL_ATTENTION_FUNCTIONS._global_mapping",
    )

    for source in production_sources:
        content = source.read_text(encoding="utf-8")
        assert all(token not in content for token in forbidden), source
