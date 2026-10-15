from pathlib import Path

import pytest
import torch

from duplex_kernel import config
from duplex_kernel.frame_residual_adapter import FrameResidualAdapter
from duplex_kernel.launch import (
    _load_query_film_v2_adapter_cpu,
    _require_query_film_v2_checkpoint,
)
from duplex_kernel.server import _normalize_chat_modalities


def test_training_voice_defaults_to_pinned_workspace_reference():
    assert Path(config.SPEAKER_REFERENCE_WAV) == (
        Path(config.WORKSPACE) / "artifacts" / "voices" / "reference.wav"
    )
    assert config.SPEAKER_REFERENCE_SHA256 == "480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c"


def test_production_checkpoint_gate_rejects_v1_adapter(tmp_path):
    adapter = FrameResidualAdapter()
    checkpoint = tmp_path / "legacy-v1.pt"
    torch.save(
        {
            "schema_version": FrameResidualAdapter.schema_version,
            "architecture": FrameResidualAdapter.architecture,
            "architecture_id": FrameResidualAdapter.architecture_id,
            "state_dict": adapter.state_dict(),
            "training_contract": {"runtime_route": "plain native Qwen Q1"},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="Query FiLM v2"):
        _load_query_film_v2_adapter_cpu(checkpoint)


def test_serving_checkpoint_path_is_required_and_resolved(tmp_path):
    with pytest.raises(FileNotFoundError, match="required"):
        _require_query_film_v2_checkpoint(None, workspace=tmp_path)

    with pytest.raises(FileNotFoundError, match="missing"):
        _require_query_film_v2_checkpoint("missing.pt", workspace=tmp_path)

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"present")
    assert _require_query_film_v2_checkpoint("candidate.pt", workspace=tmp_path) == checkpoint.resolve()


def test_chat_modalities_default_to_text_and_audio_and_reject_unhashable_values():
    assert _normalize_chat_modalities(None) == ["text", "audio"]
    assert _normalize_chat_modalities(["text"]) == ["text"]

    for invalid in ([{}], [[]], [], "audio"):
        with pytest.raises(ValueError):
            _normalize_chat_modalities(invalid)
