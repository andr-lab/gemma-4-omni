"""Explicit six-action policy and in-memory PEFT merge for Gemma kernel."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from training.duplex_qlora.control import (
    ACTIONS,
    HEAD_ARCHITECTURES,
    ControlHead,
    load_control_head,
)
from training.duplex_qlora.features import FEATURE_CONTRACT, last_audio_query
from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput
from duplex_kernel.acoustic_phase_features import (
    ACOUSTIC_PHASE_FEATURE_CONTRACT,
    ACOUSTIC_PHASE_FEATURE_NAMES,
    acoustic_phase_features,
)

ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT = {
    **FEATURE_CONTRACT,
    "input_dim": FEATURE_CONTRACT["input_dim"] + len(ACOUSTIC_PHASE_FEATURE_NAMES),
    "acoustic_phase": dict(ACOUSTIC_PHASE_FEATURE_CONTRACT),
}


def legal_control_actions(
    *,
    active_response: bool,
    response_pending: bool,
    user_turn_pending: bool,
    peer_turn_active: bool,
    backchannel_available: bool,
    renderer_plan_pending: bool,
) -> tuple[str, ...]:
    """Return the six actions legal in the current scheduler state."""
    active = bool(active_response)
    pending = bool(response_pending)
    allowed = ["LISTEN"]
    if active:
        allowed.extend(("CONTINUE", "YIELD"))
    if active or bool(renderer_plan_pending):
        allowed.append("REPLAN")
    if (
        bool(peer_turn_active)
        and bool(backchannel_available)
        and not active
        and not pending
    ):
        allowed.append("BACKCHANNEL")
    if bool(user_turn_pending) and not bool(peer_turn_active) and not active and not pending:
        allowed.append("SPEAK")
    return tuple(allowed)


class DuplexControl6Policy:
    policy_id = "duplex_control6"

    def __init__(self, runtime: dict[str, Any]) -> None:
        required = {"adapter_path", "control_head", "actions", "feature_contract", "head"}
        missing = required - set(runtime)
        if missing:
            raise ValueError(f"duplex_control6 runtime incompleto: {sorted(missing)}")
        if tuple(runtime["actions"]) != ACTIONS:
            raise ValueError("duplex_control6 runtime action/feature contract non compatibile")
        self.feature_contract = _resolve_runtime_feature_contract(
            runtime["feature_contract"], where="runtime.feature_contract"
        )
        if not isinstance(runtime["head"], ControlHead):
            raise TypeError("duplex_control6 richiede il proprio ControlHead a sei classi")
        self.runtime = runtime
        self.head = runtime["head"]
        self.input_dim = int(self.feature_contract["input_dim"])
        if int(getattr(self.head, "input_dim", -1)) != self.input_dim:
            raise ValueError(
                "duplex_control6 feature/head width mismatch: "
                f"contract={self.input_dim}, head={getattr(self.head, 'input_dim', None)}"
            )
        self.requires_acoustic_phase = self.input_dim == ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT["input_dim"]
        self.calibration = _validate_response_policy_calibration(runtime.get("calibration"))

    def step(self, state: PolicyInput) -> PolicyDecision:
        if state.listener_audio_queries is None or state.listener_frame_id is None:
            raise RuntimeError(
                "duplex_control6 richiede le quattro query Gemma native del tick corrente; "
                "non è consentito il fallback alla feature V9/2048"
            )
        query_origin_frame_id = int(state.listener_frame_id)
        query_age_frames = int(state.frame_id) - query_origin_frame_id
        if query_age_frames < 0 or (
            query_age_frames != 0 and not state.listener_query_reused
        ) or (state.listener_query_reused and query_age_frames <= 0):
            raise RuntimeError(
                "duplex_control6 accetta una query precedente solo sul tick esplicito di silence-skip: "
                f"origin={query_origin_frame_id} current={state.frame_id} reused={state.listener_query_reused}"
            )
        feature = last_audio_query(state.listener_audio_queries)
        parameter = next(self.head.parameters())
        feature = feature.detach().to(device=parameter.device, dtype=parameter.dtype).reshape(1, 1536)
        if self.requires_acoustic_phase:
            if state.user_pcm_history is None:
                raise RuntimeError("duplex_control6 acoustic phase richiede history PCM causale")
            acoustic = acoustic_phase_features(
                state.user_pcm_history,
                state.assistant_reference_pcm,
            )
            acoustic_tensor = torch.as_tensor(
                acoustic,
                device=parameter.device,
                dtype=parameter.dtype,
            ).reshape(1, len(ACOUSTIC_PHASE_FEATURE_NAMES))
            feature = torch.cat((feature, acoustic_tensor), dim=-1)
        if feature.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"duplex_control6 policy feature width {feature.shape[-1]} != {self.input_dim}"
            )
        with torch.inference_mode():
            logits = self.head(feature)
            probability = torch.softmax(logits.float(), dim=-1)[0]
            native_index = int(logits[0].argmax().item())
            native_confidence = float(probability[native_index].item())
            selected_index = native_index
            margin = None
            calibration_applied = False
            if self.calibration is not None and ACTIONS[native_index] in self.calibration["eligible_actions"]:
                baseline_indices = [ACTIONS.index(name) for name in self.calibration["baseline_actions"]]
                baseline_index = max(baseline_indices, key=lambda index: float(logits[0, index].item()))
                margin = float(logits[0, native_index].item()) - float(logits[0, baseline_index].item())
                if margin < float(self.calibration["threshold"]):
                    selected_index = baseline_index
                    calibration_applied = True
            confidence = float(probability[selected_index].item())
            native_probabilities = [float(value) for value in probability.detach().cpu().tolist()]
        native_action = ControlAction(ACTIONS[native_index])
        action = ControlAction(ACTIONS[selected_index])
        return PolicyDecision(
            action=action,
            frame_id=int(state.frame_id),
            confidence=confidence,
            reason="duplex_control6_calibrated" if calibration_applied else "duplex_control6",
            generation_id=int(state.generation_id),
            listener_frame_id=query_origin_frame_id,
            source=(
                "native_gemma_listener:last_audio_query+causal_acoustic_phase"
                if self.requires_acoustic_phase
                else "native_gemma_listener:last_audio_query"
            ),
            session=str(state.session_id),
            diagnostics={
                "native_action": native_action.value,
                "native_action_probability": native_confidence,
                "native_action_probabilities": native_probabilities,
                "selected_action": action.value,
                "selected_action_probability": confidence,
                "calibration_applied": calibration_applied,
                "calibration_margin": margin,
                "calibration_threshold": (
                    float(self.calibration["threshold"]) if self.calibration is not None else None
                ),
                "feature_contract_input_dim": self.input_dim,
                "acoustic_phase_features_used": self.requires_acoustic_phase,
                "listener_query_reused": bool(state.listener_query_reused),
                "listener_query_origin_frame_id": query_origin_frame_id,
                "listener_query_age_frames": query_age_frames,
            },
        )

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        pass


def _resolve_runtime_feature_contract(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"duplex_control6 {where} deve essere un oggetto feature contract")
    if value == FEATURE_CONTRACT:
        return dict(FEATURE_CONTRACT)
    if value == ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT:
        return {
            **FEATURE_CONTRACT,
            "input_dim": int(ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT["input_dim"]),
            "acoustic_phase": dict(ACOUSTIC_PHASE_FEATURE_CONTRACT),
        }
    raise ValueError(f"duplex_control6 {where} non compatibile con ABI 1536 o acoustic-phase 1540")


def _read_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"duplex_control6 training manifest mancante: {source}")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("duplex_control6 training manifest schema non compatibile")
    contract = manifest.get("control_head")
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("duplex_control6 manifest control_head deve essere un oggetto")
    if isinstance(contract, dict):
        expected_fields = {
            "pooling": "last_audio_query",
            "tick_ms": 80,
            "actions": list(ACTIONS),
        }
        for field, expected in expected_fields.items():
            if field in contract and contract[field] != expected:
                raise ValueError(
                    "duplex_control6 manifest ABI incompatibile: "
                    f"{field}: actual={contract[field]!r}, expected={expected!r}"
                )
        head_input_dim = contract.get("input_dim", 1536)
        if isinstance(head_input_dim, bool) or head_input_dim not in {1536, 1540}:
            raise ValueError("duplex_control6 manifest control_head.input_dim deve essere 1536 o 1540")
        architecture = contract.get("architecture")
        if architecture is not None and architecture not in HEAD_ARCHITECTURES:
            raise ValueError(
                "duplex_control6 manifest architecture non supportata: "
                f"actual={architecture!r}, expected one of {HEAD_ARCHITECTURES!r}"
            )
        _validate_optional_feature_contract(contract.get("feature_contract"), where="control_head.feature_contract")
    _validate_optional_feature_contract(manifest.get("feature_contract"), where="feature_contract")
    return manifest


def _bundle_file(root: Path, value: Any, *, label: str, must_be_dir: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"response bundle {label} deve essere un percorso relativo non vuoto")
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"response bundle {label} deve restare dentro il bundle")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"response bundle {label} deve restare inside bundle: {value!r}")
    if must_be_dir:
        if not resolved.is_dir():
            raise FileNotFoundError(f"response bundle {label} non trovato: {resolved}")
    elif not resolved.is_file():
        raise FileNotFoundError(f"response bundle {label} non trovato: {resolved}")
    return resolved


def read_duplex_response_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Resolve a self-contained model+policy bundle using only its runtime ABI.

    The manifest deliberately has no training run ID or historical base-model
    attestation. The PEFT adapter is merged into the configured Gemma base at
    runtime; the head's own serialized ABI is checked by ``load_control_head``.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"duplex response bundle manifest mancante: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"duplex response bundle manifest illeggibile: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("duplex response bundle manifest deve essere un object")
    if manifest.get("bundle_format") != "gemma4_duplex_response" or manifest.get("schema_version") != 1:
        raise ValueError("duplex response bundle format/schema non compatibile")
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise ValueError("duplex response bundle bundle_id deve essere una stringa non vuota")

    model = manifest.get("model")
    policy = manifest.get("policy")
    backchannel = manifest.get("backchannel")
    if not isinstance(model, dict) or not isinstance(policy, dict) or not isinstance(backchannel, dict):
        raise ValueError("duplex response bundle richiede model, policy e backchannel object")
    if model.get("adapter_format") != "peft":
        raise ValueError("duplex response bundle model.adapter_format deve essere 'peft'")
    base_model_id = model.get("base_model_id")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError("duplex response bundle model.base_model_id deve identificare il modello base")
    if policy.get("id") != "duplex_control6":
        raise ValueError("duplex response bundle policy.id deve essere 'duplex_control6'")
    if policy.get("actions") != list(ACTIONS):
        raise ValueError("duplex response bundle action contract non compatibile")
    feature_contract = _resolve_runtime_feature_contract(
        policy.get("feature_contract"), where="policy.feature_contract"
    )
    calibration = _validate_response_policy_calibration(policy.get("calibration"))
    expected_abi = {
        "hidden_dim": 1536,
        "query_index": 3,
        "classes": list(ACTIONS),
        "sample_rate_hz": 16000,
        "channels": 1,
        "tick_ms": 80,
    }
    if policy.get("abi") != expected_abi:
        raise ValueError(f"duplex response bundle runtime ABI non compatibile: expected {expected_abi!r}")
    feature_source = policy.get("feature_source")
    if not isinstance(feature_source, dict):
        raise ValueError("duplex response bundle policy.feature_source deve descrivere il modello delle feature")
    if feature_source.get("base_model_id") != base_model_id:
        raise ValueError("duplex response bundle head feature base non coincide con il modello del bundle")
    if feature_source.get("adapter_dir") != model.get("adapter_dir"):
        raise ValueError("duplex response bundle head feature adapter non coincide con l'adapter del bundle")
    if backchannel.get("language") != "it":
        raise ValueError("duplex response bundle backchannel.language deve essere 'it'")

    adapter_dir = _bundle_file(root, model.get("adapter_dir"), label="model.adapter_dir", must_be_dir=True)
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        raise FileNotFoundError(
            f"response bundle PEFT adapter incompleto: richiesti {adapter_config.name} e {adapter_weights.name}"
        )
    try:
        peft_config = json.loads(adapter_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"response bundle PEFT adapter_config non leggibile: {exc}") from exc
    if not isinstance(peft_config, dict) or str(peft_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("response bundle adapter_config deve dichiarare peft_type='LORA'")

    head_path = _bundle_file(root, policy.get("head_path"), label="policy.head_path")
    lexicon_path = _bundle_file(root, backchannel.get("lexicon_path"), label="backchannel.lexicon_path")
    try:
        lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"response bundle backchannel lexicon illeggibile: {exc}") from exc
    if (
        not isinstance(lexicon, dict)
        or lexicon.get("schema_version") != 1
        or lexicon.get("language") != "it"
        or lexicon.get("source_split", "train") != "train"
        or not isinstance(lexicon.get("phrases"), list)
        or not lexicon["phrases"]
        or any(not isinstance(phrase, str) or not phrase.strip() for phrase in lexicon["phrases"])
    ):
        raise ValueError("response bundle backchannel lexicon deve contenere frasi italiane di train valide")
    return {
        "bundle_id": bundle_id.strip(),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "root": root,
        "adapter_dir": adapter_dir,
        "base_model_id": base_model_id.strip(),
        "feature_source": dict(feature_source),
        "runtime_abi": expected_abi,
        "head_path": head_path,
        "lexicon_path": lexicon_path,
        "actions": list(ACTIONS),
        "feature_contract": feature_contract,
        "calibration": calibration,
    }


def load_duplex_response_bundle_runtime(
    thinker: Any,
    *,
    bundle_dir: str | Path,
    device: str | torch.device,
    apply_response_lora: bool = True,
) -> dict[str, Any]:
    """Load a bundle's control head/lexicon, optionally applying its response LoRA."""
    if not isinstance(apply_response_lora, bool):
        raise TypeError("apply_response_lora must be a bool")
    bundle = read_duplex_response_bundle(bundle_dir)
    if apply_response_lora:
        base_path = merge_gemma_adapter_into_thinker(thinker, bundle["adapter_dir"])
    else:
        base_path = "base_model_response_lora_ablation"
    head = load_control_head(bundle["head_path"], device=device)
    feature_contract = bundle["feature_contract"]
    if int(getattr(head, "input_dim", -1)) != int(feature_contract["input_dim"]):
        raise ValueError(
            "duplex response bundle feature/head width mismatch: "
            f"feature_contract={feature_contract['input_dim']}, head={getattr(head, 'input_dim', None)}"
        )
    return {
        "bundle_id": bundle["bundle_id"],
        "bundle_root": str(bundle["root"]),
        "bundle_manifest": str(bundle["manifest_path"]),
        "base_path": base_path,
        "response_lora_applied": apply_response_lora,
        "base_model_id": bundle["base_model_id"],
        "feature_source": dict(bundle["feature_source"]),
        "runtime_abi": dict(bundle["runtime_abi"]),
        "adapter_path": str(bundle["adapter_dir"]),
        "control_head": str(bundle["head_path"]),
        "backchannel_lexicon_path": str(bundle["lexicon_path"]),
        "actions": list(bundle["actions"]),
        "feature_contract": dict(bundle["feature_contract"]),
        "calibration": bundle["calibration"],
        "head": head,
    }


def _validate_response_policy_calibration(value: Any) -> dict[str, Any] | None:
    """Validate the optional experimental six-class event margin contract."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("duplex response bundle policy.calibration deve essere un oggetto")
    expected = {
        "schema_version": 1,
        "kind": "shared_non_idle_logit_margin",
        "baseline_actions": ["LISTEN", "CONTINUE"],
        "eligible_actions": ["SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "source": "validation",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(
                "duplex response bundle calibration incompatibile: "
                f"{field}: actual={value.get(field)!r}, expected={expected_value!r}"
            )
    threshold = value.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("duplex response bundle calibration.threshold deve essere finito e numerico")
    return {
        **expected,
        "threshold": float(threshold),
    }


def _validate_optional_feature_contract(value: Any, *, where: str) -> None:
    """Reject contradictory declarations without requiring redundant metadata."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"duplex_control6 manifest {where} deve essere un oggetto")
    expected_fields = {
        "queries_per_tick": 4,
        "pooling": "last_audio_query",
        "tick_ms": 80,
        "source": "native_gemma_listener",
    }
    for field, expected in expected_fields.items():
        if field in value and value[field] != expected:
            raise ValueError(
                "duplex_control6 manifest feature ABI incompatibile: "
                f"{where}.{field}: actual={value[field]!r}, expected={expected!r}"
            )
    input_dim = value.get("input_dim", 1536)
    if isinstance(input_dim, bool) or input_dim not in {1536, 1540}:
        raise ValueError(f"duplex_control6 manifest {where}.input_dim deve essere 1536 o 1540")
    if input_dim == 1540:
        if value.get("acoustic_phase") != ACOUSTIC_PHASE_FEATURE_CONTRACT:
            raise ValueError(
                f"duplex_control6 manifest {where}.acoustic_phase deve corrispondere al contract canonico"
            )
    elif "acoustic_phase" in value:
        raise ValueError(f"duplex_control6 manifest {where}.acoustic_phase richiede input_dim=1540")


def _validate_loaded_head_architecture(manifest: dict[str, Any], head: ControlHead) -> None:
    contract = manifest.get("control_head")
    if not isinstance(contract, dict):
        return
    declared = contract.get("architecture")
    if declared is None:
        return
    actual = getattr(head, "architecture", None)
    if actual != declared:
        raise ValueError(
            "duplex_control6 architettura dichiarata diversa dalla head caricata: "
            f"manifest={declared!r}, head={actual!r}"
        )


def merge_gemma_adapter_into_thinker(thinker: Any, adapter_dir: str | Path) -> str:
    """Merge a QLoRA adapter into the BF16 base before native pointer binding.

    This adapter-only path is used for the audio/Qwen smoke before the
    six-action head exists. It does not create or activate a turn policy.
    """
    adapter = Path(adapter_dir).resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"Gemma QLoRA adapter mancante: {adapter}")
    base_model = getattr(thinker, "model", None)
    if base_model is None:
        raise TypeError("Thinker wrapper privo del modello Gemma BF16")
    actual_base = str(
        getattr(base_model, "_gemma4_checkpoint_path", "")
        or getattr(base_model, "name_or_path", "")
    )
    floating = [parameter for parameter in base_model.parameters() if parameter.is_floating_point()]
    if not floating or any(parameter.dtype != torch.bfloat16 for parameter in floating):
        raise RuntimeError("merge QLoRA kernel richiede il modello base BF16 prima del binding")
    try:
        from peft import PeftModel

        merged = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=False).merge_and_unload()
    except Exception as exc:
        raise RuntimeError(f"merge PEFT Gemma fallito: {type(exc).__name__}: {exc}") from exc
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("merge_and_unload ha lasciato LoRA attive prima del listener nativo")
    nested = getattr(merged, "model", None)
    if (
        nested is None
        or not callable(getattr(nested, "get_audio_features", None))
        or not hasattr(nested, "language_model")
    ):
        raise RuntimeError("Gemma PEFT merged non supporta il listener multimodale richiesto dal kernel")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        for parameter in merged.parameters()
    ):
        raise RuntimeError("merge QLoRA kernel deve mantenere tutti i pesi floating in BF16")
    for module in (merged, nested, nested.language_model):
        setattr(module, "_gemma4_checkpoint_path", actual_base)
    merged.eval()
    thinker.model = merged
    return actual_base


def load_control6_runtime(
    thinker: Any,
    *,
    adapter_dir: str | Path,
    head_path: str | Path | None,
    manifest_path: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Merge onto the native BF16 base before a listener can bind data_ptrs."""
    manifest = _read_manifest(manifest_path)
    adapter = Path(adapter_dir).resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"duplex_control6 adapter LoRA mancante: {adapter}")
    declared_adapter = Path(str(manifest.get("adapter_dir", ""))).resolve()
    if adapter != declared_adapter:
        raise ValueError(f"adapter runtime != manifest: {adapter} != {declared_adapter}")
    declared_head = manifest.get("control_head_path")
    chosen_head = Path(head_path).resolve() if head_path else Path(str(declared_head)).resolve()
    if not declared_head or chosen_head != Path(str(declared_head)).resolve():
        raise ValueError("control head runtime deve corrispondere al percorso nel manifest")
    if not chosen_head.is_file():
        raise FileNotFoundError(f"duplex_control6 control head mancante: {chosen_head}")

    base_model = getattr(thinker, "model", None)
    if base_model is None:
        raise TypeError("Thinker wrapper privo del modello Gemma BF16")
    declared_base = str(manifest.get("base_model_path", manifest.get("base_model_id", "")))
    actual_base = str(getattr(base_model, "_gemma4_checkpoint_path", "") or getattr(base_model, "name_or_path", ""))
    if not actual_base or not declared_base:
        raise ValueError("Gemma base path non identificabile per duplex_control6")
    declared_path = Path(declared_base)
    if declared_path.exists():
        if Path(actual_base).resolve() != declared_path.resolve():
            raise ValueError(f"Gemma base diverso dal training manifest: {actual_base} != {declared_base}")
    elif actual_base != declared_base and getattr(getattr(base_model, "config", None), "_name_or_path", None) != declared_base:
        raise ValueError(f"Gemma base id diverso dal training manifest: {actual_base} != {declared_base}")

    merge_gemma_adapter_into_thinker(thinker, adapter)
    head = load_control_head(chosen_head, device=device)
    _validate_loaded_head_architecture(manifest, head)
    head_contract = manifest.get("control_head")
    feature_contract_value = (
        manifest.get("feature_contract")
        or (head_contract.get("feature_contract") if isinstance(head_contract, dict) else None)
        or FEATURE_CONTRACT
    )
    feature_contract = _resolve_runtime_feature_contract(
        feature_contract_value,
        where="manifest.feature_contract",
    )
    if int(getattr(head, "input_dim", -1)) != int(feature_contract["input_dim"]):
        raise ValueError(
            "duplex_control6 manifest/head width mismatch: "
            f"feature_contract={feature_contract['input_dim']}, head={getattr(head, 'input_dim', None)}"
        )
    return {
        "run_id": str(manifest.get("run_id", "")),
        "base_path": actual_base,
        "adapter_path": str(adapter),
        "control_head": str(chosen_head),
        "actions": list(ACTIONS),
        "feature_contract": feature_contract,
        "head": head,
    }


__all__ = [
    "DuplexControl6Policy",
    "load_control6_runtime",
    "load_duplex_response_bundle_runtime",
    "read_duplex_response_bundle",
    "merge_gemma_adapter_into_thinker",
]
