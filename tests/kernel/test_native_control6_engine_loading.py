from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

import duplex_kernel.engine as engine_module
import duplex_kernel.duplex_control6 as control6_module
import duplex_kernel.listener as listener_module
from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.engine import PipelineEngine
import training.duplex_qlora.native_control6 as native_control6_module


def _engine_that_stops_at_thinker_load(monkeypatch, tmp_path, **kwargs) -> tuple[PipelineEngine, list[bool]]:
    monkeypatch.setenv("DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX", "0")
    checkpoint = tmp_path / "native_control6.pt"
    checkpoint.write_bytes(b"test checkpoint placeholder")
    adapter_dir = tmp_path / "action_adapter"
    adapter_dir.mkdir()
    engine = PipelineEngine(device="cpu", dtype=torch.float32, **kwargs)
    reached_thinker_load: list[bool] = []

    def stop_at_thinker_load(*_args, **_kwargs):
        reached_thinker_load.append(True)
        raise RuntimeError("test stopped before model allocation")

    monkeypatch.setattr(engine_module, "load_thinker", stop_at_thinker_load)
    engine.native_control6_adapter_path = str(checkpoint)
    engine.gemma_qlora_adapter_dir = str(adapter_dir)
    return engine, reached_thinker_load


def test_native_control6_accepts_action_qlora_without_response_lora_ablation(monkeypatch, tmp_path) -> None:
    engine, reached = _engine_that_stops_at_thinker_load(
        monkeypatch,
        tmp_path,
        enable_gemma_native_control6_policy=True,
        native_control6_adapter_path="native_control6.pt",
        gemma_qlora_adapter_dir="action_adapter",
    )

    engine.load(workspace=str(tmp_path))

    assert reached == [True]


def test_native_control6_allows_response_lora_ablation_without_response_bundle(monkeypatch, tmp_path) -> None:
    engine, reached = _engine_that_stops_at_thinker_load(
        monkeypatch,
        tmp_path,
        enable_gemma_native_control6_policy=True,
        native_control6_adapter_path="native_control6.pt",
        gemma_qlora_adapter_dir="action_adapter",
        ablate_response_lora=True,
    )

    engine.load(workspace=str(tmp_path))

    assert reached == [True]


def test_native_control6_loads_action_qlora_with_bundle_lora_ablated(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX", "0")
    monkeypatch.setenv("DUPLEX_TALKER_NATIVE_AUDIO_Q1", "1")
    native_checkpoint = tmp_path / "native_control6.pt"
    native_checkpoint.write_bytes(b"test checkpoint placeholder")
    bundle_dir = tmp_path / "response_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    action_adapter = tmp_path / "action_adapter"
    action_adapter.mkdir()
    engine = PipelineEngine(
        device="cpu",
        dtype=torch.float32,
        enable_gemma_native_control6_policy=True,
        native_control6_adapter_path=str(native_checkpoint),
        gemma_qlora_adapter_dir=str(action_adapter),
        duplex_response_bundle_dir=str(bundle_dir),
        ablate_response_lora=True,
    )

    class StopAfterCompatibilityValidation(RuntimeError):
        pass

    reached: list[bool] = []

    def stop_at_thinker_load(*_args, **_kwargs):
        reached.append(True)
        raise StopAfterCompatibilityValidation("reached model allocation boundary")

    monkeypatch.setattr(engine_module, "load_thinker", stop_at_thinker_load)
    monkeypatch.setattr(
        control6_module,
        "read_duplex_response_bundle",
        lambda _path: {"manifest": {"response_protocol": AUDIO_RESPONSE_PROTOCOL_V2}},
    )

    engine.load(workspace=str(tmp_path))
    assert reached == [True]


@pytest.mark.parametrize(
    ("native_control6_enabled", "response_lora_ablated"),
    [(False, True), (True, False), (False, False)],
)
def test_response_bundle_and_action_qlora_reject_unsupported_policy_modes(
    native_control6_enabled: bool,
    response_lora_ablated: bool,
) -> None:
    with pytest.raises(ValueError, match="only with native Control6"):
        engine_module._validate_response_bundle_qlora_compatibility(
            bundle_enabled=True,
            gemma_qlora_enabled=True,
            native_control6_enabled=native_control6_enabled,
            response_lora_ablated=response_lora_ablated,
        )


def test_response_bundle_action_qlora_native_control6_merge_order(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX", "0")
    monkeypatch.setenv("DUPLEX_TALKER_NATIVE_AUDIO_Q1", "1")
    native_checkpoint = tmp_path / "native_control6.pt"
    native_checkpoint.write_bytes(b"test checkpoint placeholder")
    bundle_dir = tmp_path / "response_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    lexicon_path = tmp_path / "backchannel_lexicon.json"
    lexicon_path.write_text(
        json.dumps({"schema_version": 1, "language": "it", "source_split": "train", "phrases": ["bene"]}),
        encoding="utf-8",
    )
    action_adapter = tmp_path / "action_adapter"
    action_adapter.mkdir()
    engine = PipelineEngine(
        device="cpu",
        dtype=torch.float32,
        enable_gemma_native_control6_policy=True,
        native_control6_adapter_path=str(native_checkpoint),
        gemma_qlora_adapter_dir=str(action_adapter),
        duplex_response_bundle_dir=str(bundle_dir),
        ablate_response_lora=True,
    )
    order: list[str] = []
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros((2, 2))))
    thinker = SimpleNamespace(
        model=model,
        tokenizer=SimpleNamespace(eos_token_id=2, encode=lambda *_a, **_k: [5]),
    )

    def stop_at_listener(*_args, **_kwargs):
        order.append("listener")
        raise RuntimeError("stop after adapter binding")

    class NativeRuntime(torch.nn.Linear):
        rank = 16

    monkeypatch.setattr(engine_module, "load_thinker", lambda *_a, **_k: thinker)
    monkeypatch.setattr(
        control6_module,
        "read_duplex_response_bundle",
        lambda _path: {"manifest": {"response_protocol": AUDIO_RESPONSE_PROTOCOL_V2}},
    )

    def load_bundle(_thinker, *, bundle_dir, device, apply_response_lora):
        order.append(f"bundle:{apply_response_lora}")
        return {
            "bundle_id": "test-bundle", "adapter_path": str(bundle_dir),
            "control_head": "test-head", "backchannel_lexicon_path": str(lexicon_path),
        }

    def merge_action(_thinker, adapter_dir):
        order.append("action_qlora")
        assert Path(adapter_dir).resolve() == action_adapter.resolve()
        return "action-qlora-merged"

    def load_native(_path, *, base_rows, device):
        order.append("native_control6")
        assert tuple(base_rows.shape) == (1, 1536)
        return NativeRuntime(1, 1), {"loaded": True}

    monkeypatch.setattr(control6_module, "load_duplex_response_bundle_runtime", load_bundle)
    monkeypatch.setattr(control6_module, "merge_gemma_adapter_into_thinker", merge_action)
    monkeypatch.setattr(native_control6_module, "select_tied_output_rows", lambda _weight: torch.zeros((1, 1536)))
    monkeypatch.setattr(native_control6_module, "load_native_control6_adapter", load_native)
    monkeypatch.setattr(listener_module.GemmaListener, "from_thinker", stop_at_listener)

    engine.load(workspace=str(tmp_path))

    assert order == ["bundle:False", "action_qlora", "native_control6", "listener"]
    assert engine.gemma_qlora_adapter_loaded == str(action_adapter.resolve())
