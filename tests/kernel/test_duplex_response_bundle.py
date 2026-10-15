from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest
import torch

from duplex_kernel import duplex_control6
from duplex_kernel.duplex_control6 import ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT
from training.duplex_qlora.control import ControlHead
from training.duplex_qlora.features import FEATURE_CONTRACT
from training.duplex_qlora.schema import ACTIONS


def _bundle(root: Path) -> dict:
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
    ControlHead(architecture="linear").save(root / "control_head.pt")
    (root / "backchannel_lexicon.json").write_text(json.dumps({
        "schema_version": 1,
        "language": "it",
        "source_split": "train",
        "phrases": ["Sì.", "Capisco."],
    }), encoding="utf-8")
    return {
        "bundle_format": "gemma4_duplex_response",
        "schema_version": 1,
        "bundle_id": "candidate-lm-head-v5",
        "model": {
            "base_model_id": "google/gemma-4-E2B-it",
            "adapter_dir": "adapter",
            "adapter_format": "peft",
        },
        "policy": {
            "id": "duplex_control6",
            "head_path": "control_head.pt",
            "actions": list(ACTIONS),
            "feature_contract": dict(FEATURE_CONTRACT),
            "abi": {
                "hidden_dim": 1536, "query_index": 3, "classes": list(ACTIONS),
                "sample_rate_hz": 16000, "channels": 1, "tick_ms": 80,
            },
            "feature_source": {
                "base_model_id": "google/gemma-4-E2B-it", "adapter_dir": "adapter",
            },
        },
        "backchannel": {"lexicon_path": "backchannel_lexicon.json", "language": "it"},
    }


def _write_bundle(root: Path, manifest: dict) -> Path:
    path = root / "bundle_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_new_bundle_uses_only_runtime_abi_and_resolves_its_own_artifacts(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    path = _write_bundle(tmp_path, manifest)

    resolved = duplex_control6.read_duplex_response_bundle(tmp_path)

    assert resolved["bundle_id"] == "candidate-lm-head-v5"
    assert resolved["adapter_dir"] == (tmp_path / "adapter").resolve()
    assert resolved["head_path"] == (tmp_path / "control_head.pt").resolve()
    assert resolved["lexicon_path"] == (tmp_path / "backchannel_lexicon.json").resolve()
    assert resolved["base_model_id"] == "google/gemma-4-E2B-it"
    assert resolved["feature_source"]["adapter_dir"] == "adapter"
    assert resolved["runtime_abi"]["query_index"] == 3
    assert resolved["manifest_path"] == path.resolve()
    assert "run_id" not in resolved


def test_new_bundle_rejects_paths_outside_its_root(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    manifest["policy"]["head_path"] = "../control_head.pt"
    _write_bundle(tmp_path, manifest)

    with pytest.raises(ValueError, match="inside bundle"):
        duplex_control6.read_duplex_response_bundle(tmp_path)


def test_new_bundle_requires_exact_policy_feature_abi(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    manifest["policy"]["actions"] = list(reversed(ACTIONS))
    _write_bundle(tmp_path, manifest)

    with pytest.raises(ValueError, match="action contract"):
        duplex_control6.read_duplex_response_bundle(tmp_path)


def test_new_bundle_accepts_only_the_canonical_optional_1540_acoustic_feature_contract(
    tmp_path: Path,
) -> None:
    manifest = _bundle(tmp_path)
    manifest["policy"]["feature_contract"] = copy.deepcopy(ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT)
    _write_bundle(tmp_path, manifest)

    resolved = duplex_control6.read_duplex_response_bundle(tmp_path)

    assert resolved["feature_contract"] == ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT

    manifest["policy"]["feature_contract"]["acoustic_phase"] = {
        **manifest["policy"]["feature_contract"]["acoustic_phase"],
        "names": list(reversed(manifest["policy"]["feature_contract"]["acoustic_phase"]["names"])),
    }
    _write_bundle(tmp_path, manifest)
    with pytest.raises(ValueError, match="acoustic-phase 1540"):
        duplex_control6.read_duplex_response_bundle(tmp_path)


def test_new_bundle_validates_optional_six_action_margin_calibration(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    manifest["policy"]["calibration"] = {
        "schema_version": 1,
        "kind": "shared_non_idle_logit_margin",
        "threshold": 1.365189790725708,
        "baseline_actions": ["LISTEN", "CONTINUE"],
        "eligible_actions": ["SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "source": "validation",
    }
    _write_bundle(tmp_path, manifest)

    resolved = duplex_control6.read_duplex_response_bundle(tmp_path)

    assert resolved["calibration"]["threshold"] == pytest.approx(1.365189790725708)
    assert resolved["calibration"]["baseline_actions"] == ["LISTEN", "CONTINUE"]


def test_new_bundle_rejects_nonfinite_calibration_threshold(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    manifest["policy"]["calibration"] = {
        "schema_version": 1,
        "kind": "shared_non_idle_logit_margin",
        "threshold": float("nan"),
        "baseline_actions": ["LISTEN", "CONTINUE"],
        "eligible_actions": ["SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "source": "validation",
    }
    _write_bundle(tmp_path, manifest)

    with pytest.raises(ValueError, match="finito e numerico"):
        duplex_control6.read_duplex_response_bundle(tmp_path)


def test_new_bundle_runtime_merges_its_adapter_and_loads_its_head_without_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path, _bundle(tmp_path))
    thinker = object()
    merged: list[tuple[object, Path]] = []

    def merge(th: object, adapter_dir: str | Path) -> str:
        merged.append((th, Path(adapter_dir)))
        return "runtime-base-path"

    monkeypatch.setattr(duplex_control6, "merge_gemma_adapter_into_thinker", merge)
    runtime = duplex_control6.load_duplex_response_bundle_runtime(
        thinker, bundle_dir=tmp_path, device="cpu"
    )

    assert merged == [(thinker, (tmp_path / "adapter").resolve())]
    assert runtime["bundle_id"] == "candidate-lm-head-v5"
    assert runtime["response_lora_applied"] is True
    assert runtime["adapter_path"] == str((tmp_path / "adapter").resolve())
    assert runtime["control_head"] == str((tmp_path / "control_head.pt").resolve())
    assert runtime["backchannel_lexicon_path"] == str((tmp_path / "backchannel_lexicon.json").resolve())
    assert isinstance(runtime["head"], ControlHead)
    assert runtime["actions"] == list(ACTIONS)
    assert runtime["feature_contract"] == FEATURE_CONTRACT
    assert runtime["base_model_id"] == "google/gemma-4-E2B-it"
    assert runtime["runtime_abi"]["sample_rate_hz"] == 16000
    assert runtime["calibration"] is None
    assert "run_id" not in runtime


def test_response_lora_ablation_keeps_bundle_control_head_without_merging_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path, _bundle(tmp_path))
    thinker = object()
    merge_calls: list[tuple[object, Path]] = []

    def merge(th: object, adapter_dir: str | Path) -> str:
        merge_calls.append((th, Path(adapter_dir)))
        return "unexpected-merged-base"

    monkeypatch.setattr(duplex_control6, "merge_gemma_adapter_into_thinker", merge)
    runtime = duplex_control6.load_duplex_response_bundle_runtime(
        thinker, bundle_dir=tmp_path, device="cpu", apply_response_lora=False
    )

    assert merge_calls == []
    assert runtime["response_lora_applied"] is False
    assert runtime["base_path"] == "base_model_response_lora_ablation"
    assert isinstance(runtime["head"], ControlHead)
    assert runtime["control_head"] == str((tmp_path / "control_head.pt").resolve())
    assert runtime["adapter_path"] == str((tmp_path / "adapter").resolve())
