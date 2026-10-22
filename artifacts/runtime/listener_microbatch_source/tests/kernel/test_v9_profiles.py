from __future__ import annotations

from pathlib import Path
import inspect

import pytest
import torch

from duplex_kernel import config
from duplex_kernel.engine import PipelineEngine


def test_canonical_engine_default_is_causal_l0() -> None:
    assert inspect.signature(PipelineEngine.__init__).parameters["v9_profile"].default == config.V9_PROFILE_CAUSAL_L0


def test_v9_quality_oracle_profile_resolves_final() -> None:
    spec = config.v9_profile_spec(config.V9_PROFILE_QUALITY_FINAL_ORACLE)

    assert Path(spec["path"]).name == "v9_prefix_final.pt"
    assert spec["recipe"] == config.V9_QUALITY_RECIPE
    assert spec["sha256"] == config.EXPECTED_QUALITY_V9_SHA256


def test_v9_causal_profile_resolves_l0() -> None:
    spec = config.v9_profile_spec(config.V9_PROFILE_CAUSAL_L0)

    assert Path(spec["path"]).name == "v9_causal_l0_final.pt"
    assert spec["recipe"] == config.V9_CAUSAL_L0_RECIPE
    assert spec["sha256"] == config.EXPECTED_CAUSAL_V9_SHA256


def test_quality_profile_requires_pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXPECTED_QUALITY_V9_SHA256", "<PIN>")

    with pytest.raises(RuntimeError, match="pinned"):
        config.v9_profile_spec(config.V9_PROFILE_QUALITY_FINAL_ORACLE)


class _IdentityCausalConv:
    def __init__(self) -> None:
        self.inputs: torch.Tensor | None = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs = value.detach().clone()
        return value


class _RecordingProjector:
    def __init__(self) -> None:
        self.raw: torch.Tensor | None = None
        self.text: torch.Tensor | None = None

    def __call__(self, raw: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        self.raw = raw.detach().clone()
        self.text = text.detach().clone()
        return torch.zeros((*raw.shape[:-1], 2048), dtype=raw.dtype)


def _engine_for_v9() -> PipelineEngine:
    engine = object.__new__(PipelineEngine)
    engine.dtype = torch.float32
    engine.v9_profile = config.V9_PROFILE_QUALITY_FINAL_ORACLE
    engine.v9_recipe = config.V9_QUALITY_RECIPE
    engine.causal_conv = _IdentityCausalConv()
    engine.v9 = _RecordingProjector()
    engine.v9_stream = None
    return engine


def test_quality_batch_recipe_uses_mean_and_sixteen_frames() -> None:
    engine = _engine_for_v9()
    hidden = torch.arange(1 * 5 * 1536, dtype=torch.float32).reshape(1, 5, 1536)

    output = engine.v9_prefix(hidden, prompt_len=1)

    assert tuple(output.shape) == (1, 16, 2048)
    projector = engine.v9
    assert torch.equal(projector.text, hidden[:, 1:, :].mean(dim=1, keepdim=True))
    assert tuple(projector.raw.shape) == (1, 16, 1536)
    assert tuple(engine.causal_conv.inputs.shape) == (1, 16, 1536)


def test_quality_batch_recipe_is_not_available_on_causal_profile() -> None:
    engine = _engine_for_v9()
    engine.v9_profile = config.V9_PROFILE_CAUSAL_L0
    engine.v9_recipe = config.V9_CAUSAL_L0_RECIPE

    with pytest.raises(RuntimeError, match="V9 incrementale"):
        engine.v9_prefix(torch.zeros(1, 8, 1536), prompt_len=0)


def test_stateful_recipe_rejects_quality_profile() -> None:
    engine = _engine_for_v9()

    with pytest.raises(RuntimeError, match="V9_PROFILE_RECIPE_MISMATCH"):
        engine.v9_prefix_stateful(torch.zeros(1, 1, 1536), prompt_len=0)
