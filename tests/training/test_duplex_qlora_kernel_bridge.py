from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from duplex_kernel.duplex_control6 import merge_gemma_adapter_into_thinker


class _GemmaInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    def get_audio_features(self, *args, **kwargs):
        return None


class _BaseModel(nn.Module):
    def __init__(self, base_path):
        super().__init__()
        self.model = _GemmaInner()
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self._gemma4_checkpoint_path = str(base_path)


class _Thinker:
    def __init__(self, model):
        self.model = model


def _peft_passthrough(monkeypatch):
    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter_dir, *, is_trainable):
            assert is_trainable is False
            assert adapter_dir

            class Wrapper:
                def merge_and_unload(self):
                    return model

            return Wrapper()

    module = types.ModuleType("peft")
    module.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "peft", module)


def test_adapter_only_kernel_merge_accepts_real_bf16_topology_before_listener(tmp_path, monkeypatch):
    _peft_passthrough(monkeypatch)
    base_path = tmp_path / "gemma"
    base_path.mkdir()
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    model = _BaseModel(base_path)
    thinker = _Thinker(model)

    loaded_base = merge_gemma_adapter_into_thinker(thinker, adapter_path)

    assert loaded_base == str(base_path.resolve())
    assert thinker.model is model
    assert thinker.model.model.language_model._gemma4_checkpoint_path == loaded_base


def test_adapter_only_kernel_merge_rejects_wrong_dtype_before_pointer_binding(tmp_path, monkeypatch):
    _peft_passthrough(monkeypatch)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    base_path = tmp_path / "gemma"
    base_path.mkdir()
    model = _BaseModel(base_path)
    model.weight.data = model.weight.data.float()
    thinker = _Thinker(model)

    with pytest.raises(RuntimeError, match="BF16"):
        merge_gemma_adapter_into_thinker(thinker, adapter_path)
