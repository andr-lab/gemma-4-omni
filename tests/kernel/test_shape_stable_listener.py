from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

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
