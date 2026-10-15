#!/usr/bin/env python3
"""Run an evidence-oriented E2E evaluation against an already-running duplex server.

This tool never loads a Gemma checkpoint. It sends cached 16 kHz user WAVs to a
running server, captures its raw events/audio, and checks that the requested
checkpoint and control policy are the ones the server says it loaded.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter, deque
import json
import math
import re
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dialogue_eval.audio_metrics import compute_physical_audio_metrics
from dialogue_eval.evaluator import calculate_wer
from dialogue_eval.scenarios import ALL_SCENARIOS, DialogueScenario, ScenarioStep
from duplex_kernel.acoustic_phase_features import ACOUSTIC_PHASE_FEATURE_CONTRACT

ACTIONS = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
ACTION_ORDER = ACTIONS
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/gemma4_duplex_qlora")
DEFAULT_SCENARIO_AUDIO_DIR = Path("artifacts/scenario_audio_qwen3tts")
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
FRAME_MS = 80
FRAME_SAMPLES = INPUT_RATE * FRAME_MS // 1000
_DECISION_FIELDS = ("action", "confidence", "frame_id", "listener_frame_id", "session", "source")


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"training manifest {label} must be a non-empty string")
    return value.strip()


def _resolve_checkpoint_path(root: Path, raw: Any, label: str) -> Path:
    value = _required_string(raw, label)
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def load_checkpoint_manifest(checkpoint_root: str | Path) -> dict[str, Any]:
    """Load either a legacy training identity or a self-contained runtime bundle."""
    root = Path(checkpoint_root).expanduser().resolve()
    response_manifest_path = root / "bundle_manifest.json"
    if response_manifest_path.is_file():
        try:
            bundle = json.loads(response_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read response bundle manifest {response_manifest_path}: {exc}") from exc
        if not isinstance(bundle, dict):
            raise ValueError("response bundle manifest root must be an object")
        if bundle.get("bundle_format") != "gemma4_duplex_response" or bundle.get("schema_version") != 1:
            raise ValueError("response bundle format/schema is not supported")
        bundle_id = _required_string(bundle.get("bundle_id"), "bundle_id")
        model, policy, backchannel = bundle.get("model"), bundle.get("policy"), bundle.get("backchannel")
        if not isinstance(model, dict) or not isinstance(policy, dict) or not isinstance(backchannel, dict):
            raise ValueError("response bundle requires model, policy and backchannel objects")
        if model.get("adapter_format") != "peft" or policy.get("id") != "duplex_control6":
            raise ValueError("response bundle adapter/policy ABI is not supported")
        base_model_id = _required_string(model.get("base_model_id"), "model.base_model_id")
        if policy.get("actions") != list(ACTION_ORDER):
            raise ValueError(f"response bundle actions must match {list(ACTION_ORDER)}")
        expected_feature_base = {
            "queries_per_tick": 4, "input_dim": 1536, "pooling": "last_audio_query",
            "tick_ms": FRAME_MS, "source": "native_gemma_listener",
        }
        feature_contract = policy.get("feature_contract")
        legacy_feature = expected_feature_base
        phase_feature = {
            **expected_feature_base,
            "input_dim": 1540,
            "acoustic_phase": ACOUSTIC_PHASE_FEATURE_CONTRACT,
        }
        if feature_contract == legacy_feature:
            expected_feature = legacy_feature
        elif feature_contract == phase_feature:
            expected_feature = phase_feature
        else:
            raise ValueError("response bundle native listener feature contract is not supported")
        expected_abi = {
            "hidden_dim": 1536, "query_index": 3, "classes": list(ACTION_ORDER),
            "sample_rate_hz": INPUT_RATE, "channels": 1, "tick_ms": FRAME_MS,
        }
        if policy.get("abi") != expected_abi:
            raise ValueError("response bundle runtime ABI is not supported")
        feature_source = policy.get("feature_source")
        if not isinstance(feature_source, dict):
            raise ValueError("response bundle policy.feature_source must describe listener features")
        if feature_source.get("base_model_id") != base_model_id or feature_source.get("adapter_dir") != model.get("adapter_dir"):
            raise ValueError("response bundle head feature provenance does not match its model adapter")
        if backchannel.get("language") != "it":
            raise ValueError("response bundle backchannel language must be it")

        def resolve_bundle_path(raw: Any, label: str, *, directory: bool = False) -> Path:
            value = _required_string(raw, label)
            relative = Path(value).expanduser()
            if relative.is_absolute():
                raise ValueError(f"response bundle {label} must be relative to its bundle")
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"response bundle {label} must stay inside the bundle")
            if directory and not resolved.is_dir():
                raise ValueError(f"response bundle {label} directory is missing: {resolved}")
            if not directory and not resolved.is_file():
                raise ValueError(f"response bundle {label} file is missing: {resolved}")
            return resolved

        adapter_path = resolve_bundle_path(model.get("adapter_dir"), "model.adapter_dir", directory=True)
        if not (adapter_path / "adapter_config.json").is_file() or not (adapter_path / "adapter_model.safetensors").is_file():
            raise ValueError("response bundle PEFT adapter files are incomplete")
        control_head_path = resolve_bundle_path(policy.get("head_path"), "policy.head_path")
        lexicon_path = resolve_bundle_path(backchannel.get("lexicon_path"), "backchannel.lexicon_path")
        return {
            **bundle,
            "bundle_id": bundle_id,
            "base_model_id": base_model_id,
            "feature_source": dict(feature_source),
            "runtime_abi": expected_abi,
            "_response_bundle": True,
            "_checkpoint_root": root,
            "_manifest_path": response_manifest_path.resolve(),
            "adapter_path": adapter_path,
            "control_head_path_resolved": control_head_path,
            "control_head_path": control_head_path,
            "backchannel_lexicon_path": lexicon_path,
            "control_head": {
                "input_dim": expected_feature["input_dim"], "pooling": "last_audio_query", "tick_ms": FRAME_MS,
                "actions": list(ACTION_ORDER), "feature_contract": expected_feature,
            },
        }

    manifest_path = root / "training_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"training manifest not found: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("training manifest root must be an object")
    run_id = _required_string(manifest.get("run_id"), "run_id")
    base_model_id = _required_string(manifest.get("base_model_id"), "base_model_id")
    # base_model_path can be omitted when the base is addressed by model id.
    base_model_path = manifest.get("base_model_path")
    if base_model_path is not None:
        base_model_path = _required_string(base_model_path, "base_model_path")
    adapter_path = _resolve_checkpoint_path(root, manifest.get("adapter_dir"), "adapter_dir")
    control_head_path = _resolve_checkpoint_path(root, manifest.get("control_head_path"), "control_head_path")
    control = manifest.get("control_head")
    if not isinstance(control, dict):
        raise ValueError("training manifest control_head must be an object")
    if control.get("input_dim") != 1536:
        raise ValueError("control_head.input_dim must be 1536")
    if control.get("pooling") != "last_audio_query":
        raise ValueError("control_head.pooling must be 'last_audio_query'")
    if control.get("tick_ms") != FRAME_MS:
        raise ValueError("control_head.tick_ms must be 80")
    if control.get("actions") != list(ACTION_ORDER):
        raise ValueError(f"control_head.actions must match the ordered contract {list(ACTION_ORDER)}")
    return {
        **manifest,
        "run_id": run_id,
        "base_model_id": base_model_id,
        "base_model_path": base_model_path,
        "_checkpoint_root": root,
        "_manifest_path": manifest_path.resolve(),
        "adapter_path": adapter_path,
        "control_head_path": control_head_path,
        "control_head_path_resolved": control_head_path,
    }


def _event(row: dict[str, Any]) -> dict[str, Any]:
    event = row.get("event", row)
    return event if isinstance(event, dict) else {}


def _canonical_location(value: Any, root: Path) -> str | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    text = str(value).strip()
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve()).replace("\\", "/").casefold()
    rooted = (root / path).resolve()
    return str(rooted).replace("\\", "/").casefold()


def _huggingface_repo_id(value: Any) -> str | None:
    """Recover `owner/repo` from a Hugging Face cache path when the manifest stores an ID."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    for part in Path(str(value)).expanduser().parts:
        normalized = part.replace("\\", "/")
        if normalized.startswith("models--"):
            repo_id = normalized[len("models--"):].replace("--", "/").casefold()
            if repo_id.count("/") == 1 and all(repo_id.split("/")):
                return repo_id
    return None


def _valid_decision(event: dict[str, Any], policy_name: str) -> bool:
    if event.get("type") != "duplex_control6.decision" or any(field not in event for field in _DECISION_FIELDS):
        return False
    # `source` identifies the feature producer (currently the native listener),
    # while the policy is identified by the event type and the matching
    # `model_loaded.policy_id` checked before decisions are accepted.
    if event.get("action") not in ACTION_ORDER:
        return False
    if not isinstance(event.get("source"), str) or not event["source"].strip():
        return False
    confidence = event.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        return False
    if "accepted" in event and not isinstance(event["accepted"], bool):
        return False
    for field in ("frame_id", "listener_frame_id"):
        if isinstance(event.get(field), bool) or not isinstance(event.get(field), int) or event[field] < 0:
            return False
    return isinstance(event.get("session"), str) and bool(event["session"].strip())


def validate_model_loaded_identity(
    manifest: dict[str, Any], rows: Iterable[dict[str, Any]], policy_name: str
) -> dict[str, Any]:
    """Compare every live model_loaded event with the requested runtime before sending audio."""
    materialized = list(rows)
    loaded_events = [_event(row) for row in materialized if _event(row).get("type") == "model_loaded"]
    if not loaded_events:
        return {"status": "unavailable", "reason": "server emitted no model_loaded event", "mismatches": []}
    root = Path(manifest["_checkpoint_root"])
    if manifest.get("_response_bundle"):
        expected = {
            "bundle_id": manifest["bundle_id"],
            "base_model_id": manifest["base_model_id"],
            "feature_source": manifest["feature_source"],
            "runtime_abi": manifest["runtime_abi"],
            "adapter_path": _canonical_location(manifest["adapter_path"], root),
            "control_head": _canonical_location(manifest["control_head_path_resolved"], root),
            "actions": list(ACTION_ORDER),
        }
        mismatches: list[str] = []
        for event in loaded_events:
            candidate = {
                "bundle_id": event.get("bundle_id"),
                "base_model_id": event.get("base_model_id"),
                "feature_source": event.get("feature_source"),
                "runtime_abi": event.get("runtime_abi"),
                "adapter_path": _canonical_location(event.get("adapter_path"), root),
                "control_head": _canonical_location(event.get("control_head"), root),
                "actions": event.get("actions"),
            }
            mismatches.extend(key for key in expected if candidate[key] != expected[key])
            feature = event.get("feature_contract")
            if not isinstance(feature, dict):
                mismatches.append("feature_contract")
            else:
                for key in ("input_dim", "pooling", "tick_ms"):
                    if feature.get(key) != manifest["control_head"][key]:
                        mismatches.append(f"feature_contract.{key}")
            event_policy = event.get("policy_id", event.get("policy"))
            if event_policy != policy_name:
                mismatches.append("policy")
        if mismatches:
            return {"status": "mismatch", "reason": "loaded runtime does not match response bundle",
                    "mismatches": sorted(set(mismatches)), "loaded_events": len(loaded_events)}
        return {"status": "matched", "reason": None, "mismatches": [], "loaded_event": loaded_events[0],
                "loaded_events": len(loaded_events)}

    expected_base = manifest.get("base_model_path") or manifest["base_model_id"]
    expected = {
        "run_id": manifest["run_id"],
        "base_path": _canonical_location(expected_base, root),
        "adapter_path": _canonical_location(manifest["adapter_path"], root),
        "control_head": _canonical_location(manifest["control_head_path_resolved"], root),
        "actions": list(ACTION_ORDER),
    }
    mismatches: list[str] = []
    for event in loaded_events:
        feature = event.get("feature_contract")
        candidate = {
            "run_id": event.get("run_id"),
            "base_path": _canonical_location(event.get("base_path"), root),
            "adapter_path": _canonical_location(event.get("adapter_path"), root),
            "control_head": _canonical_location(event.get("control_head"), root),
            "actions": event.get("actions"),
        }
        event_mismatch = [key for key in expected if candidate[key] != expected[key]]
        if (
            "base_path" in event_mismatch
            and _huggingface_repo_id(event.get("base_path")) == manifest["base_model_id"].casefold()
        ):
            event_mismatch.remove("base_path")
        if not isinstance(feature, dict):
            event_mismatch.append("feature_contract")
        else:
            # The serving event carries the action-logit order at top-level. Do not
            # require a duplicated feature_contract.actions field: current server
            # versions omit it while preserving the canonical top-level order.
            for key in ("input_dim", "pooling", "tick_ms"):
                expected_value = manifest["control_head"][key]
                if feature.get(key) != expected_value:
                    event_mismatch.append(f"feature_contract.{key}")
        event_policy = event.get("policy_id", event.get("policy"))
        if event_policy != policy_name:
            event_mismatch.append("policy")
        mismatches.extend(event_mismatch)
    if mismatches:
        return {"status": "mismatch", "reason": "model_loaded identity does not match the requested checkpoint",
                "mismatches": sorted(set(mismatches)), "loaded_events": len(loaded_events)}
    return {"status": "matched", "reason": None, "mismatches": [], "loaded_event": loaded_events[0],
            "loaded_events": len(loaded_events)}


def validate_loaded_identity(
    manifest: dict[str, Any], rows: Iterable[dict[str, Any]], policy_name: str
) -> dict[str, Any]:
    """Require matching live checkpoint identity plus a decision from the selected policy."""
    materialized = list(rows)
    identity = validate_model_loaded_identity(manifest, materialized, policy_name)
    if identity["status"] != "matched":
        return identity
    decisions = [
        _event(row) for row in materialized
        if _valid_decision(_event(row), policy_name)
    ]
    if not decisions:
        return {**identity, "status": "unavailable", "reason": f"no complete decision event from {policy_name}",
                "decision_events": 0}
    return {**identity, "decision_events": len(decisions)}


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Character error rate with Unicode text normalization matching the WER helper."""
    ref = re.findall(r"\w", reference.casefold())
    hyp = re.findall(r"\w", hypothesis.casefold())
    if not ref:
        return 0.0 if not hyp else 1.0
    distance = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, start=1):
        current = [i]
        for j, right in enumerate(hyp, start=1):
            current.append(min(current[-1] + 1, distance[j] + 1, distance[j - 1] + (left != right)))
        distance = current
    return round(min(1.0, distance[-1] / len(ref)), 4)


def compare_asr_metrics(
    input_reference: str,
    input_asr: str,
    output_text: str,
    output_asr: str,
    *,
    configured: bool,
    error: str | None = None,
) -> dict[str, Any]:
    """Return WER/CER only when configured independent ASR succeeded on both captured WAVs."""
    result: dict[str, Any] = {
        "status": "unavailable", "reason": None, "input_wer": None, "input_cer": None,
        "output_wer": None, "output_cer": None,
    }
    if not configured:
        result["reason"] = "no local ASR model configured"
        return result
    if error:
        result["reason"] = f"local ASR invocation failed: {error}"
        return result
    if not all(isinstance(text, str) and text.strip() for text in (input_reference, input_asr, output_text, output_asr)):
        result["reason"] = "ASR or response text missing for input/output comparison"
        return result
    result.update({
        "status": "available", "reason": None,
        "input_wer": calculate_wer(input_reference, input_asr),
        "input_cer": calculate_cer(input_reference, input_asr),
        "output_wer": calculate_wer(output_text, output_asr),
        "output_cer": calculate_cer(output_text, output_asr),
    })
    return result


def audio_quality(
    pcm: np.ndarray, sample_rate: int, packet_boundary_indices: Iterable[int] | None = None
) -> dict[str, Any]:
    array = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if array.size == 0:
        return {"status": "unavailable", "reason": "no captured samples", "metrics": None}
    metrics = compute_physical_audio_metrics(
        array, sample_rate, packet_boundary_indices=list(packet_boundary_indices or [])
    )
    frame_samples = max(1, round(sample_rate * 0.04))
    silence_threshold = 0.005
    frames = [array[start:start + frame_samples] for start in range(0, array.size, frame_samples)]
    silent_frames = 0
    for frame in frames:
        centered = frame - float(np.mean(frame))
        if float(np.sqrt(np.mean(centered ** 2))) < silence_threshold:
            silent_frames += 1
    silence_ratio = silent_frames / len(frames)
    metrics["silence_frame_ratio"] = round(silence_ratio, 4)
    issues = []
    if not metrics["finite"]:
        issues.append("captured audio contains non-finite samples")
        status = "invalid"
    elif silence_ratio >= 0.95 or metrics["rms"] < silence_threshold:
        issues.append("captured audio is silent or nearly silent")
        status = "silent"
    elif metrics["clipping_pct"] > 0.1:
        issues.append(f"clipping {metrics['clipping_pct']}% exceeds the 0.1% audio gate")
        status = "degraded"
    else:
        status = "available"
    return {
        "status": status,
        "reason": "; ".join(issues) or None,
        "metrics": metrics,
    }


def action_coverage(
    rows: Iterable[dict[str, Any]], actions: Iterable[str], policy_name: str, *, accepted_only: bool = False
) -> dict[str, Any]:
    expected = list(actions)
    counts = Counter(
        _event(row)["action"] for row in rows
        if _valid_decision(_event(row), policy_name)
        and (not accepted_only or _event(row).get("accepted") is True)
    )
    observed = sorted(counts, key=lambda action: expected.index(action) if action in expected else len(expected))
    missing = [action for action in expected if action not in observed]
    return {"expected": expected, "observed": observed, "missing": missing,
            "decision_count": sum(counts.values()),
            "count_by_action": {action: counts.get(action, 0) for action in expected},
            "status": "complete" if not missing else "incomplete",
            "accepted_only": bool(accepted_only)}


def action_lifecycle_coverage(
    rows: Iterable[dict[str, Any]], actions: Iterable[str], policy_name: str
) -> dict[str, Any]:
    """Keep deferred listener intents separate from accepted decisions and created responses."""
    expected = list(actions)
    materialized = list(rows)
    deferred_reasons = {"deferred_speak_dispatched", "deferred_until_backchannel_done"}
    deferred_counts: Counter[str] = Counter()
    deferred_by_reason: Counter[str] = Counter()
    executed_counts: Counter[str] = Counter()
    unattributed_created = 0

    for row in materialized:
        event = _event(row)
        if (
            _valid_decision(event, policy_name)
            and event.get("accepted") is False
            and event.get("rejection_reason") in deferred_reasons
        ):
            deferred_counts[str(event["action"])] += 1
            deferred_by_reason[str(event["rejection_reason"])] += 1
        if event.get("type") != "response.created":
            continue
        metadata = event.get("metadata")
        candidates = (
            event.get("effective_action"),
            metadata.get("action") if isinstance(metadata, dict) else None,
            event.get("action"),
        )
        action = next(
            (value for value in candidates if isinstance(value, str) and value in expected),
            None,
        )
        if action is None:
            unattributed_created += 1
        else:
            executed_counts[action] += 1

    def summary(counts: Counter[str]) -> dict[str, Any]:
        observed = sorted(counts, key=lambda action: expected.index(action) if action in expected else len(expected))
        return {
            "expected": expected,
            "observed": observed,
            "missing": [action for action in expected if action not in observed],
            "count": sum(counts.values()),
            "count_by_action": {action: counts.get(action, 0) for action in expected},
        }

    return {
        "deferred_decisions": {
            **summary(deferred_counts),
            "count_by_reason": {
                reason: deferred_by_reason.get(reason, 0)
                for reason in sorted(deferred_reasons)
            },
        },
        "executed_responses": {
            **summary(executed_counts),
            "unattributed_response_count": unattributed_created,
        },
    }


def live_policy_semantics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed action/ack/cancel timing without inferring intent from text.

    These counters make the desired CONTINUE, BACKCHANNEL and REPLAN interactions
    visible in the real-audio report. They are observations, not model-quality
    certification gates.
    """
    materialized = list(rows)

    def owner(row: dict[str, Any], event: dict[str, Any]) -> str:
        return str(row.get("session_id") or row.get("session") or event.get("session") or "")

    consumed: set[tuple[str, int]] = set()
    input_appends: dict[str, list[int]] = defaultdict(list)
    event_rows: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for index, row in enumerate(materialized):
        event = _event(row)
        kind = str(event.get("type", ""))
        session = owner(row, event)
        timestamp = int(row.get("received_at_monotonic_ns", row.get("sent_at_monotonic_ns", 0)) or 0)
        if kind == "input_audio_buffer.consumed" and isinstance(event.get("frame_id"), int):
            consumed.add((session, int(event["frame_id"])))
        if row.get("direction") == "client" and kind == "input_audio_buffer.append":
            input_appends[session].append(timestamp)
        event_rows.append((index, session, event, row))

    decisions = [
        (index, session, event, row)
        for index, session, event, row in event_rows
        if _valid_decision(event, "duplex_control6") and event.get("accepted") is True
    ]
    continues = [item for item in decisions if item[2].get("action") == "CONTINUE"]
    replans = [item for item in decisions if item[2].get("action") == "REPLAN"]
    backchannels = [item for item in decisions if item[2].get("action") == "BACKCHANNEL"]
    continue_ack_matches = sum(
        (session, int(event["frame_id"])) in consumed
        for _, session, event, _ in continues
    )

    created_backchannels: list[tuple[int, str, int, str]] = []
    for index, session, event, row in event_rows:
        if event.get("type") == "response.created" and event.get("effective_action", event.get("action")) == "BACKCHANNEL":
            created_backchannels.append((index, session, int(row.get("received_at_monotonic_ns", 0) or 0), str(event.get("response_id", ""))))
    backchannel_overlaps = 0
    for _, session, created_ns, response_id in created_backchannels:
        terminal_ns = next((
            int(row.get("received_at_monotonic_ns", 0) or 0)
            for _, candidate_session, event, row in event_rows
            if candidate_session == session and event.get("response_id") == response_id
            and event.get("type") in {"response.done", "response.cancelled"}
            and int(row.get("received_at_monotonic_ns", 0) or 0) >= created_ns
        ), None)
        if terminal_ns is not None and any(created_ns < stamp < terminal_ns for stamp in input_appends.get(session, [])):
            backchannel_overlaps += 1

    replan_cancel_resume = 0
    for decision_index, session, decision, _ in replans:
        decision_frame = int(decision["frame_id"])
        cancel_index = None
        for index, candidate_session, event, _ in event_rows:
            if index <= decision_index or candidate_session != session:
                continue
            if event.get("type") == "response.cancelled" and str(event.get("reason", "")).lower() == "replan":
                cancel_index = index
                break
        if cancel_index is None:
            continue
        resumed = any(
            index > cancel_index and candidate_session == session
            and event.get("type") == "response.created"
            and event.get("effective_action", event.get("action")) in {"SPEAK", "REPLAN"}
            and isinstance(event.get("intent_endpoint_frame_id"), int)
            and int(event["intent_endpoint_frame_id"]) >= decision_frame
            for index, candidate_session, event, _ in event_rows
        )
        replan_cancel_resume += int(resumed)

    return {
        "continue": {
            "accepted_decisions": len(continues),
            "same_frame_listener_consumed_ack": continue_ack_matches,
            "ack_without_same_frame_replan": sum(
                (session, int(event["frame_id"])) in consumed
                and not any(other_session == session and int(other["frame_id"]) == int(event["frame_id"])
                            for _, other_session, other, _ in replans)
                for _, session, event, _ in continues
            ),
        },
        "backchannel": {
            "accepted_decisions": len(backchannels),
            "responses_created": len(created_backchannels),
            "responses_with_user_audio_continuing_during_playout": backchannel_overlaps,
        },
        "replan": {
            "accepted_decisions": len(replans),
            "cancel_then_endpoint_scoped_resume": replan_cancel_resume,
        },
    }


def session_isolation_evidence(rows: Iterable[dict[str, Any]], policy_name: str = "duplex_control6") -> dict[str, Any]:
    by_session: dict[str, dict[str, Any]] = defaultdict(lambda: {"responses": set(), "decisions": 0})
    response_owners: dict[str, set[str]] = defaultdict(set)
    violations: list[str] = []
    for row in rows:
        session_id = row.get("session_id") or row.get("session")
        event = _event(row)
        event_session = event.get("session")
        if event.get("type") not in {"duplex_control6.decision", "response.created"}:
            continue
        if session_id and event_session and str(event_session) != str(session_id):
            violations.append(f"event session {event_session!r} differs from connection {session_id!r}")
        owner = str(session_id or event_session or "")
        if not owner:
            continue
        if event.get("type") == "response.created" and event.get("response_id"):
            by_session[owner]["responses"].add(str(event["response_id"]))
            response_owners[str(event["response_id"])].add(owner)
        if _valid_decision(event, policy_name):
            by_session[owner]["decisions"] += 1
    duplicated = sorted(response_id for response_id, owners in response_owners.items() if len(owners) > 1)
    sessions = sorted(by_session)
    complete = len(sessions) >= 2 and all(
        by_session[session]["responses"] and by_session[session]["decisions"] > 0 for session in sessions[:2]
    )
    if duplicated:
        violations.append(f"response IDs shared across sessions: {duplicated}")
    status = "mismatch" if violations else ("observed" if complete else "unavailable")
    return {"status": status, "sessions": sessions,
            "response_count_by_session": {key: len(value["responses"]) for key, value in by_session.items()},
            "decision_count_by_session": {key: value["decisions"] for key, value in by_session.items()},
            "violations": violations}


def _row_event(row: dict[str, Any]) -> dict[str, Any]:
    return _event(row)


def compute_event_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only observed client sends/server events and monotonic timestamps."""
    materialized = list(rows)
    event_counts: dict[str, int] = defaultdict(int)
    input_frames = input_samples = ack_count = errors = created_count = completed = cancelled = output_samples = 0
    output_rate = None
    commit_by_session: dict[str, int] = {}
    last_input_by_session: dict[str, int] = {}
    start_latencies: list[float] = []
    first_audio_latencies: list[float] = []
    e2e_latencies: list[float] = []
    response_samples: dict[str, int] = defaultdict(int)
    response_rate: dict[str, int] = {}
    response_start: dict[str, int] = {}
    response_commit: dict[str, int] = {}
    response_first_audio: dict[str, int] = {}
    response_end: dict[str, int] = {}
    for row in materialized:
        direction = row.get("direction")
        if direction not in {"client", "server"}:
            continue
        event = _row_event(row)
        kind = str(event.get("type", "unknown"))
        event_counts[kind] += 1
        session = str(row.get("session_id") or row.get("session") or "")
        if direction == "client":
            sent_ns = int(row.get("sent_at_monotonic_ns", 0))
            if kind == "input_audio_buffer.append":
                input_frames += 1
                input_samples += int(row.get("audio_samples", event.get("audio_samples", FRAME_SAMPLES)))
                last_input_by_session[session] = sent_ns
            elif kind == "input_audio_buffer.commit":
                commit_by_session[session] = sent_ns
            continue
        if kind in {"input_audio_buffer.consumed", "input_audio_buffer.ack", "input_audio_buffer.acknowledged"}:
            ack_count += 1
        if kind == "error":
            errors += 1
        received_ns = int(row.get("received_at_monotonic_ns", 0))
        if kind == "response.created":
            created_count += 1
            response_id = str(event.get("response_id", f"anonymous-{created_count}"))
            response_start[response_id] = received_ns
            response_commit[response_id] = commit_by_session.get(session, last_input_by_session.get(session, received_ns))
            start_latencies.append(max(0.0, received_ns - response_commit[response_id]) / 1_000_000.0)
        elif kind == "response.audio.delta":
            response_id = str(event.get("response_id", ""))
            response_first_audio.setdefault(response_id, received_ns)
            count = int(row.get("audio_samples", event.get("audio_samples", 0)))
            rate = int(row.get("sample_rate", event.get("sample_rate", OUTPUT_RATE)))
            response_samples[response_id] += count
            response_rate[response_id] = rate
            output_samples += count
            output_rate = rate
        elif kind in {"response.done", "response.cancelled"}:
            response_id = str(event.get("response_id", ""))
            if kind == "response.done":
                completed += 1
            else:
                cancelled += 1
            response_end[response_id] = received_ns
            start_ns = response_commit.get(response_id)
            if start_ns is not None and received_ns >= start_ns:
                e2e_latencies.append((received_ns - start_ns) / 1_000_000.0)
    for response_id, first_audio_ns in response_first_audio.items():
        input_frame_ns = response_commit.get(response_id)
        if input_frame_ns is not None and first_audio_ns >= input_frame_ns:
            first_audio_latencies.append((first_audio_ns - input_frame_ns) / 1_000_000.0)
    response_rtfs = []
    for response_id, samples in response_samples.items():
        rate = response_rate.get(response_id, OUTPUT_RATE)
        elapsed_ns = response_end.get(response_id, 0) - response_start.get(response_id, 0)
        if samples and rate > 0 and elapsed_ns > 0:
            response_rtfs.append((samples / rate) / (elapsed_ns / 1_000_000_000.0))
    def mean(values: list[float]) -> float | None:
        return round(float(sum(values) / len(values)), 4) if values else None
    return {
        "event_counts": dict(sorted(event_counts.items())), "input_frames_sent": input_frames,
        "input_audio_samples_sent": input_samples, "input_ack_events": ack_count, "server_errors": errors,
        "responses_created": created_count, "responses_completed": completed, "responses_cancelled": cancelled,
        "output_audio_samples": output_samples, "output_sample_rate": output_rate,
        # Last-frame latency is transport/pipeline evidence, not annotated utterance-end latency.
        "frame_to_response_start_latency_ms": [round(value, 4) for value in start_latencies],
        "mean_frame_to_response_start_latency_ms": mean(start_latencies),
        "frame_to_first_audio_latency_ms": [round(value, 4) for value in first_audio_latencies],
        "mean_frame_to_first_audio_latency_ms": mean(first_audio_latencies),
        "frame_to_response_done_latency_ms": [round(value, 4) for value in e2e_latencies],
        "mean_frame_to_response_done_latency_ms": mean(e2e_latencies), "response_rtf": [round(value, 4) for value in response_rtfs],
        "output_rtf": mean(response_rtfs),
    }


def summarize_input_audio_transcription(
    rows: Iterable[dict[str, Any]], *, requested: bool, enabled: bool,
) -> dict[str, Any]:
    """Summarize public native-transcript snapshots without retaining transcript text."""
    if not requested:
        return {
            "status": "not_requested", "requested": False, "enabled": bool(enabled),
            "partial_events": 0, "completed_events": 0, "utterances": 0,
            "revision_monotonic": None,
            "first_partial_from_utterance_onset_ms": [], "first_partial_delivery_lag_ms": [],
        }
    if not enabled:
        return {
            "status": "not_enabled", "requested": True, "enabled": False,
            "partial_events": 0, "completed_events": 0, "utterances": 0,
            "revision_monotonic": None,
            "first_partial_from_utterance_onset_ms": [], "first_partial_delivery_lag_ms": [],
        }

    captured_rows = [row for row in rows if isinstance(row, dict)]
    frame_send_ns = {
        int(row["input_frame_id"]): int(row["sent_at_monotonic_ns"])
        for row in captured_rows
        if row.get("direction") == "client"
        and row.get("event", {}).get("type") == "input_audio_buffer.append"
        and isinstance(row.get("input_frame_id"), int)
        and isinstance(row.get("sent_at_monotonic_ns"), int)
    }
    events: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in captured_rows:
        event = row.get("event")
        if row.get("direction") == "server" and isinstance(event, dict) and event.get("type") in {
            "input_audio_transcription.partial", "input_audio_transcription.completed",
        }:
            events.append((row, event))

    partial_count = sum(event.get("type") == "input_audio_transcription.partial" for _, event in events)
    completed_count = len(events) - partial_count
    revisions: dict[str, int] = {}
    utterances: set[str] = set()
    onset_latencies_ms: list[float] = []
    delivery_lags_ms: list[float] = []
    first_partial_seen: set[str] = set()
    revision_monotonic = True
    valid = True
    for row, event in events:
        utterance_id = event.get("utterance_id")
        revision = event.get("revision")
        first_frame_id = event.get("first_frame_id")
        last_frame_id = event.get("last_frame_id")
        received_ns = row.get("received_at_monotonic_ns")
        if (
            not isinstance(utterance_id, str) or not utterance_id
            or isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0
            or isinstance(first_frame_id, bool) or not isinstance(first_frame_id, int)
            or isinstance(last_frame_id, bool) or not isinstance(last_frame_id, int)
            or first_frame_id < 0 or last_frame_id < first_frame_id
            or isinstance(received_ns, bool) or not isinstance(received_ns, int)
        ):
            valid = False
            continue
        utterances.add(utterance_id)
        previous_revision = revisions.get(utterance_id)
        if previous_revision is not None and revision <= previous_revision:
            revision_monotonic = False
        revisions[utterance_id] = revision
        if event.get("type") != "input_audio_transcription.partial" or utterance_id in first_partial_seen:
            continue
        first_partial_seen.add(utterance_id)
        onset_ns = frame_send_ns.get(first_frame_id)
        snapshot_ns = frame_send_ns.get(last_frame_id)
        if onset_ns is not None:
            onset_latencies_ms.append(round((received_ns - onset_ns) / 1_000_000, 3))
        if snapshot_ns is not None:
            delivery_lags_ms.append(round((received_ns - snapshot_ns) / 1_000_000, 3))

    if not revision_monotonic:
        valid = False
    status = "invalid_events" if not valid else "observed" if events else "enabled_no_events"
    return {
        "status": status,
        "requested": True,
        "enabled": True,
        "partial_events": partial_count,
        "completed_events": completed_count,
        "utterances": len(utterances),
        "revision_monotonic": revision_monotonic if events else None,
        "first_partial_from_utterance_onset_ms": onset_latencies_ms,
        "first_partial_delivery_lag_ms": delivery_lags_ms,
    }


def _decode_input_wav(path: Path) -> np.ndarray:
    # Reuse the established strict decoder and keep all ingress at its native 16 kHz.
    from client.duplex_dialogue_harness import read_scenario_audio_wav
    return read_scenario_audio_wav(path)


def _decode_output_packet(event: dict[str, Any]) -> tuple[np.ndarray, int]:
    from client.duplex_dialogue_harness import decode_response_audio
    packet = decode_response_audio(event)
    return np.asarray(packet.pcm, dtype=np.float32).reshape(-1), int(packet.sample_rate)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all():
        raise ValueError(f"refusing to write non-finite audio: {path}")
    int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(int16.tobytes())


def _scenario_assets(audio_dir: Path) -> tuple[list[tuple[DialogueScenario, dict[str, Path]]], list[dict[str, str]]]:
    return _scenario_assets_for(audio_dir, ALL_SCENARIOS)


QUALITATIVE_SCENARIOS = [
    DialogueScenario(
        scenario_id="qualitative_schedule_replan",
        title="Sonda qualitativa: conflitto d'orario, chiarimento e correzione",
        description=(
            "Tre turni italiani per osservare se l'assistente rileva un conflitto di agenda, chiarisce una fascia oraria "
            "e aggiorna la preferenza più recente mantenendo il contesto della conversazione."
        ),
        system_prompt_a="Sei una persona che deve organizzare una visita e sta chiarendo i propri vincoli di agenda.",
        system_prompt_b=(
            "Sei un assistente vocale italiano. Aiuta a organizzare l'appuntamento, fai domande quando un vincolo è "
            "ambiguo e considera le correzioni successive dell'utente come aggiornamenti della preferenza."
        ),
        steps=[
            ScenarioStep(
                "turn_1_user", "user",
                "Ho la visita martedì alle sedici e trenta, ma proprio a quell'ora accompagno mia figlia a nuoto. "
                "Puoi trovare un'alternativa e dirmi quali fasce sono libere?",
            ),
            ScenarioStep(
                "turn_2_user", "user",
                "Per chiarire, quando dici tardo pomeriggio intendi dopo le sedici oppure può andare bene anche dalle quattordici?",
            ),
            ScenarioStep(
                "turn_3_user", "user",
                "Correggo la mia preferenza: martedì non va bene. Se c'è posto, venerdì dopo le quindici va bene.",
            ),
        ],
        max_turns=3,
    ),
]


def _evaluation_scenarios(include_qualitative: bool) -> tuple[list[DialogueScenario], list[DialogueScenario]]:
    """Return canonical coverage and optional human-review scenarios separately."""
    return list(ALL_SCENARIOS), list(QUALITATIVE_SCENARIOS) if include_qualitative else []


def _select_canonical_scenarios(
    scenarios: list[DialogueScenario], requested_ids: list[str] | None,
) -> list[DialogueScenario]:
    """Select targeted canonical scenarios while preserving their declared order."""
    if not requested_ids:
        return list(scenarios)
    available_ids = {scenario.scenario_id for scenario in scenarios}
    requested = {str(value) for value in requested_ids}
    unknown = requested - available_ids
    if unknown:
        raise ValueError(f"unknown scenario id(s): {sorted(unknown)}")
    return [scenario for scenario in scenarios if scenario.scenario_id in requested]


def _scenario_assets_for(
    audio_dir: Path, scenarios: list[DialogueScenario]
) -> tuple[list[tuple[DialogueScenario, dict[str, Path]]], list[dict[str, str]]]:
    available: list[tuple[DialogueScenario, dict[str, Path]]] = []
    skipped: list[dict[str, str]] = []
    for scenario in scenarios:
        users = [step for step in scenario.steps if step.speaker == "user"]
        assets = {step.step_id: audio_dir / f"{scenario.scenario_id}__{step.step_id}.wav" for step in users}
        missing = [path.name for path in assets.values() if not path.is_file()]
        if missing:
            skipped.append({"scenario_id": scenario.scenario_id, "reason": "missing cached user WAV", "missing": ", ".join(missing)})
            continue
        # Validate all files before opening any network connection.
        try:
            for path in assets.values():
                _decode_input_wav(path)
        except Exception as exc:
            skipped.append({"scenario_id": scenario.scenario_id, "reason": f"invalid cached WAV: {exc}", "missing": ""})
            continue
        available.append((scenario, assets))
    return available, skipped


class _PCMPlaybackSimulator:
    """Consume captured response PCM through one headless, 80 ms paced playout queue.

    This models a client endpoint and its response-scoped clear behavior. It does
    not write to a physical audio device or measure device latency.
    """

    def __init__(
        self,
        *,
        label: str,
        trace_rows: list[dict[str, Any]],
        sample_rate: int = OUTPUT_RATE,
        frame_ms: int = FRAME_MS,
    ) -> None:
        if sample_rate <= 0 or frame_ms <= 0 or sample_rate * frame_ms % 1000:
            raise ValueError("playback sample rate/frame duration must form an integer frame")
        self.label = label
        self.trace_rows = trace_rows
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        self._queue: deque[tuple[str, np.ndarray, int, int]] = deque()
        self._pending: dict[str, tuple[np.ndarray, int]] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._played_parts: dict[str, list[np.ndarray]] = defaultdict(list)
        self._cancelled_entries: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self._played_timeline: list[tuple[int, int, str, int]] = []
        self._wake: asyncio.Event | None = None
        self._idle: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._active: dict[str, Any] | None = None

    def _response(self, response_id: str) -> dict[str, Any]:
        state = self._responses.get(response_id)
        if state is None:
            state = {
                "response_id": response_id,
                "first_audio_received_ns": None,
                "last_audio_received_ns": None,
                "terminal_event": None,
                "terminal_ns": None,
                "yield_ns": None,
                "audio_samples_received": 0,
                "audio_samples_received_after_yield": 0,
                "audio_samples_consumed": 0,
                "audio_samples_after_yield_consumed": 0,
                "audio_samples_cancelled": 0,
                "queue_cleared_samples": 0,
                "queued_samples": 0,
                "queued_samples_at_terminal": None,
                "done_before_playback_complete": None,
                "playback_start_ns": None,
                "playback_end_ns": None,
                "playback_tail_end_after_yield_ns": None,
                "late_audio_after_terminal_samples": 0,
                "queue_wait_ms_total": 0.0,
            }
            self._responses[response_id] = state
        return state

    def is_response_playout_active(self, response_id: str) -> bool:
        """True only after a response has entered simulated speaker playout."""
        response_id = str(response_id)
        state = self._responses.get(response_id)
        if state is None or state["playback_start_ns"] is None or state["queued_samples"] <= 0:
            return False
        return bool(self._active and self._active.get("response_id") == response_id) or state["queued_samples"] > 0

    def _timeline(self, event_type: str, timestamp_ns: int, **details: Any) -> None:
        self.trace_rows.append({
            "direction": "playback_simulator",
            "session_id": self.label,
            "at_monotonic_ns": int(timestamp_ns),
            "event": {"type": event_type, **details},
        })

    async def start(self) -> None:
        if self._task is not None:
            return
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task = asyncio.create_task(self._run(), name=f"pcm-playout-{self.label}")

    def mark_yield(self, response_id: str, timestamp_ns: int) -> None:
        if not response_id:
            self._timeline("playback_sim.yield_unmatched", timestamp_ns)
            return
        state = self._response(response_id)
        if state["yield_ns"] is None:
            state["yield_ns"] = int(timestamp_ns)
        self._timeline("playback_sim.yield", timestamp_ns, response_id=response_id)

    def _record_cancelled(self, response_id: str, pcm: np.ndarray, sample_offset: int) -> None:
        if pcm.size:
            self._cancelled_entries[response_id].append((int(sample_offset), pcm.copy()))

    def _enqueue_frame(self, response_id: str, pcm: np.ndarray, timestamp_ns: int, sample_offset: int) -> None:
        if pcm.size == 0:
            return
        assert self._wake is not None and self._idle is not None
        frame = np.asarray(pcm, dtype=np.float32).reshape(-1).copy()
        self._queue.append((response_id, frame, int(timestamp_ns), int(sample_offset)))
        self._idle.clear()
        self._wake.set()
        self._timeline(
            "playback_sim.frame_queued", timestamp_ns,
            response_id=response_id, samples=int(frame.size), queue_frames=len(self._queue),
        )

    def enqueue(self, response_id: str, pcm: np.ndarray, sample_rate: int, timestamp_ns: int) -> None:
        response_id = str(response_id)
        if not response_id:
            raise ValueError("response.audio.delta requires a response_id for simulated playout")
        if int(sample_rate) != self.sample_rate:
            raise ValueError(f"playout expected {self.sample_rate} Hz PCM, received {sample_rate} Hz")
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not array.size:
            return
        state = self._response(response_id)
        state["first_audio_received_ns"] = state["first_audio_received_ns"] or int(timestamp_ns)
        state["last_audio_received_ns"] = int(timestamp_ns)
        state["audio_samples_received"] += int(array.size)
        if state["yield_ns"] is not None and timestamp_ns >= state["yield_ns"]:
            state["audio_samples_received_after_yield"] += int(array.size)
        if state["terminal_event"] is not None:
            state["late_audio_after_terminal_samples"] += int(array.size)
            if state["terminal_event"] == "response.cancelled":
                state["audio_samples_cancelled"] += int(array.size)
                self._record_cancelled(response_id, array, state["audio_samples_received"] - array.size)
            self._timeline(
                "playback_sim.audio_after_terminal", timestamp_ns,
                response_id=response_id, samples=int(array.size),
                terminal_event=state["terminal_event"],
            )
            return

        state["queued_samples"] += int(array.size)
        prior, prior_offset = self._pending.pop(
            response_id, (np.empty(0, dtype=np.float32), int(state["audio_samples_received"] - array.size))
        )
        combined = np.concatenate((prior, array)) if prior.size else array
        full_samples = (combined.size // self.frame_samples) * self.frame_samples
        for offset in range(0, full_samples, self.frame_samples):
            self._enqueue_frame(
                response_id, combined[offset:offset + self.frame_samples], timestamp_ns, prior_offset + offset
            )
        if full_samples < combined.size:
            self._pending[response_id] = (combined[full_samples:].copy(), prior_offset + full_samples)

    def finish_response(self, response_id: str, terminal_event: str, timestamp_ns: int) -> None:
        response_id = str(response_id)
        if not response_id:
            return
        if terminal_event not in {"response.done", "response.cancelled"}:
            raise ValueError(f"unsupported response terminal event: {terminal_event}")
        state = self._response(response_id)
        if state["terminal_event"] is not None:
            self._timeline(
                "playback_sim.duplicate_terminal", timestamp_ns,
                response_id=response_id, terminal_event=terminal_event,
            )
            return
        state["terminal_event"] = terminal_event
        state["terminal_ns"] = int(timestamp_ns)
        if terminal_event == "response.done":
            tail, tail_offset = self._pending.pop(
                response_id, (np.empty(0, dtype=np.float32), int(state["audio_samples_received"]))
            )
            if tail.size:
                self._enqueue_frame(response_id, tail, timestamp_ns, tail_offset)
            state["queued_samples_at_terminal"] = int(state["queued_samples"])
            state["done_before_playback_complete"] = state["queued_samples"] > 0
            if state["queued_samples"] == 0:
                state["playback_end_ns"] = int(timestamp_ns)
        else:
            state["queued_samples_at_terminal"] = int(state["queued_samples"])
            state["done_before_playback_complete"] = state["queued_samples"] > 0
            tail, tail_offset = self._pending.pop(
                response_id, (np.empty(0, dtype=np.float32), int(state["audio_samples_received"]))
            )
            removed_samples = int(tail.size)
            if tail.size:
                self._record_cancelled(response_id, tail, tail_offset)
                state["audio_samples_cancelled"] += int(tail.size)
            kept: deque[tuple[str, np.ndarray, int, int]] = deque()
            while self._queue:
                queued_id, frame, enqueued_ns, sample_offset = self._queue.popleft()
                if queued_id == response_id:
                    removed_samples += int(frame.size)
                    self._record_cancelled(response_id, frame, sample_offset)
                    state["audio_samples_cancelled"] += int(frame.size)
                else:
                    kept.append((queued_id, frame, enqueued_ns, sample_offset))
            self._queue = kept
            state["queue_cleared_samples"] = removed_samples
            state["queued_samples"] -= removed_samples
            active = self._active
            if active is not None and active["response_id"] == response_id:
                active["cancelled_at_ns"] = int(timestamp_ns)
                active["cancel_event"].set()
            elif state["queued_samples"] == 0:
                state["playback_end_ns"] = int(timestamp_ns)
            self._timeline(
                "playback_sim.response_queue_cleared", timestamp_ns,
                response_id=response_id, queued_samples_cleared=removed_samples,
                active_frame_interrupted=bool(active is not None and active["response_id"] == response_id),
            )
        self._timeline(
            "playback_sim.response_terminal", timestamp_ns,
            response_id=response_id, terminal_event=terminal_event,
            queued_samples=int(state["queued_samples_at_terminal"] or 0),
        )

    async def _run(self) -> None:
        assert self._wake is not None and self._idle is not None
        while True:
            if not self._queue:
                self._idle.set()
                if self._closed:
                    return
                self._wake.clear()
                await self._wake.wait()
                continue

            response_id, pcm, enqueued_ns, sample_offset = self._queue.popleft()
            state = self._response(response_id)
            start_ns = time.monotonic_ns()
            cancel_event = asyncio.Event()
            active: dict[str, Any] = {
                "response_id": response_id,
                "pcm": pcm,
                "sample_offset": sample_offset,
                "start_ns": start_ns,
                "cancel_event": cancel_event,
                "cancelled_at_ns": None,
            }
            self._active = active
            self._idle.clear()
            state["playback_start_ns"] = state["playback_start_ns"] or start_ns
            state["queue_wait_ms_total"] += max(0, start_ns - enqueued_ns) / 1_000_000.0
            self._timeline(
                "playback_sim.frame_started", start_ns,
                response_id=response_id, samples=int(pcm.size),
                queue_frames=len(self._queue), queue_wait_ms=max(0, start_ns - enqueued_ns) / 1_000_000.0,
            )
            duration_s = pcm.size / self.sample_rate
            interrupted = False
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=duration_s)
                interrupted = True
            except asyncio.TimeoutError:
                pass

            if interrupted:
                end_ns = int(active["cancelled_at_ns"] or time.monotonic_ns())
                elapsed_ns = max(0, end_ns - start_ns)
                consumed_samples = min(int(pcm.size), int(elapsed_ns * self.sample_rate // 1_000_000_000))
            else:
                end_ns = time.monotonic_ns()
                consumed_samples = int(pcm.size)
            consumed = pcm[:consumed_samples]
            discarded = pcm[consumed_samples:]
            if consumed.size:
                self._played_parts[response_id].append(consumed.copy())
                self._played_timeline.append((start_ns, end_ns, response_id, int(consumed.size)))
                state["audio_samples_consumed"] += int(consumed.size)
                yield_ns = state["yield_ns"]
                if yield_ns is not None:
                    before_yield_samples = min(
                        int(pcm.size), max(0, int(math.ceil((yield_ns - start_ns) * self.sample_rate / 1_000_000_000)))
                    )
                    after_yield = max(0, consumed_samples - before_yield_samples)
                    state["audio_samples_after_yield_consumed"] += after_yield
                    if after_yield:
                        state["playback_tail_end_after_yield_ns"] = max(
                            int(state["playback_tail_end_after_yield_ns"] or 0), end_ns
                        )
            if discarded.size:
                self._record_cancelled(response_id, discarded, sample_offset + consumed_samples)
                state["audio_samples_cancelled"] += int(discarded.size)
            state["queued_samples"] -= int(pcm.size)
            self._timeline(
                "playback_sim.frame_interrupted" if interrupted else "playback_sim.frame_consumed",
                end_ns, response_id=response_id, consumed_samples=int(consumed.size),
                cancelled_samples=int(discarded.size),
            )
            self._active = None
            if state["terminal_event"] is not None and state["queued_samples"] <= 0:
                state["queued_samples"] = 0
                state["playback_end_ns"] = end_ns
            if not self._queue:
                self._idle.set()

    async def finish(self, *, timeout_s: float = 180.0) -> dict[str, Any]:
        if self._task is None:
            return self.snapshot()
        assert self._idle is not None and self._wake is not None
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            timestamp_ns = time.monotonic_ns()
            self._timeline("playback_sim.drain_timeout", timestamp_ns, timeout_s=timeout_s)
            raise TimeoutError(f"simulated playback did not drain within {timeout_s:.1f}s")
        self._closed = True
        self._wake.set()
        await self._task
        self._task = None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        summaries: dict[str, dict[str, Any]] = {}
        for response_id, state in self._responses.items():
            duration_ms = lambda sample_count: round(sample_count / self.sample_rate * 1000.0, 3)
            terminal_ns = state["terminal_ns"]
            playback_end_ns = state["playback_end_ns"]
            yield_ns = state["yield_ns"]
            summaries[response_id] = {
                **state,
                "audio_duration_received_ms": duration_ms(state["audio_samples_received"]),
                "audio_duration_consumed_ms": duration_ms(state["audio_samples_consumed"]),
                "audio_duration_cancelled_ms": duration_ms(state["audio_samples_cancelled"]),
                "queue_cleared_duration_ms": duration_ms(state["queue_cleared_samples"]),
                "playback_pending_at_terminal_ms": duration_ms(state["queued_samples_at_terminal"] or 0),
                "terminal_to_playback_end_ms": (
                    round((playback_end_ns - terminal_ns) / 1_000_000.0, 3)
                    if terminal_ns is not None and playback_end_ns is not None else None
                ),
                "audio_duration_received_after_yield_ms": duration_ms(state["audio_samples_received_after_yield"]),
                "audio_duration_consumed_after_yield_ms": duration_ms(state["audio_samples_after_yield_consumed"]),
                "playback_tail_after_yield_ms": (
                    round((state["playback_tail_end_after_yield_ns"] - yield_ns) / 1_000_000.0, 3)
                    if yield_ns is not None and state["playback_tail_end_after_yield_ns"] is not None else 0.0
                ),
                "active_queue_at_end_samples": int(state["queued_samples"]),
                "queue_wait_ms_total": round(float(state["queue_wait_ms_total"]), 3),
            }
        return {
            "endpoint": "simulated_headless_pcm_playout",
            "sample_rate_hz": self.sample_rate,
            "frame_ms": self.frame_ms,
            "physical_device_playback": False,
            "responses": summaries,
            "played_pcm_by_response": {
                response_id: np.concatenate(parts) for response_id, parts in self._played_parts.items() if parts
            },
            "cancelled_pcm_by_response": {
                response_id: np.concatenate([part for _, part in sorted(entries, key=lambda entry: entry[0])])
                for response_id, entries in self._cancelled_entries.items() if entries
            },
        }


class _SessionCollector:
    def __init__(self, label: str, output_dir: Path, trace_rows: list[dict[str, Any]]) -> None:
        self.label = label
        self.output_dir = output_dir
        self.trace_rows = trace_rows
        self.session_id = label
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.server_events: list[dict[str, Any]] = []
        self.output_parts: list[np.ndarray] = []
        self.output_parts_by_response: dict[str, list[np.ndarray]] = defaultdict(list)
        self.output_packet_boundaries: list[int] = []
        self.output_sample_count = 0
        self.output_rate: int | None = None
        self.text_by_response: dict[str, list[str]] = defaultdict(list)
        self.active_responses: set[str] = set()
        self.current_response_id: str | None = None
        self.response_actions: dict[str, str | None] = {}
        self._pending_response_actions: deque[str] = deque()
        self.last_input_append_ns: int | None = None
        self.input_frames_sent = 0
        self.input_audio_stream_parts: list[np.ndarray] = []
        self.state_changed = asyncio.Event()
        self.playout = _PCMPlaybackSimulator(label=label, trace_rows=trace_rows)
        self.closed = asyncio.Event()
        self.receive_error: str | None = None

    def record(self, direction: str, event: dict[str, Any], timestamp: int, **extra: Any) -> dict[str, Any]:
        row = {"direction": direction, "event": event, "session_id": self.session_id, **extra}
        if direction == "server":
            row["received_at_monotonic_ns"] = timestamp
            self.server_events.append(row)
            kind = event.get("type")
            if kind == "duplex_control6.decision" and event.get("accepted") is True:
                action = str(event.get("action", ""))
                if action in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
                    self._pending_response_actions.append(action)
            if kind == "response.created":
                response_id = str(event.get("response_id", ""))
                self.active_responses.add(response_id)
                self.current_response_id = response_id
                metadata = event.get("metadata")
                metadata_action = metadata.get("action") if isinstance(metadata, dict) else None
                action_candidates = (
                    event.get("effective_action"), metadata_action, event.get("action"),
                )
                action = next(
                    (candidate for candidate in action_candidates
                     if isinstance(candidate, str) and candidate in {"SPEAK", "BACKCHANNEL", "REPLAN"}),
                    None,
                )
                if self._pending_response_actions:
                    pending_action = self._pending_response_actions.popleft()
                    if action is None:
                        action = pending_action
                self.response_actions[response_id] = action
            elif kind in {"response.done", "response.cancelled"}:
                response_id = str(event.get("response_id", ""))
                self.playout.finish_response(response_id, str(kind), timestamp)
                self.active_responses.discard(response_id)
                if self.current_response_id == response_id:
                    self.current_response_id = next(iter(self.active_responses), None)
            elif kind == "response.output_text.delta":
                self.text_by_response[str(event.get("response_id", ""))].append(str(event.get("delta", "")))
            elif (
                kind == "duplex_control6.decision"
                and event.get("action") == "YIELD"
                and event.get("accepted") is True
            ):
                response_id = str(event.get("response_id") or self.current_response_id or "")
                self.playout.mark_yield(response_id, timestamp)
            elif kind == "response.audio.delta":
                try:
                    pcm, rate = _decode_output_packet(event)
                    if self.output_rate is not None and self.output_rate != rate:
                        raise ValueError(f"response sample rate changed from {self.output_rate} to {rate}")
                    self.output_rate = rate
                    if self.output_sample_count:
                        self.output_packet_boundaries.append(self.output_sample_count)
                    self.output_parts.append(pcm)
                    self.output_parts_by_response[str(event.get("response_id", ""))].append(pcm)
                    self.output_sample_count += int(pcm.size)
                    row["audio_samples"] = int(pcm.size)
                    row["sample_rate"] = rate
                    self.playout.enqueue(str(event.get("response_id", "")), pcm, rate, timestamp)
                except Exception as exc:
                    row["audio_decode_error"] = f"{type(exc).__name__}: {exc}"
            elif kind == "error":
                response_id = str(event.get("response_id", ""))
                if response_id:
                    self.active_responses.discard(response_id)
                    if self.current_response_id == response_id:
                        self.current_response_id = next(iter(self.active_responses), None)
                if self._pending_response_actions:
                    self._pending_response_actions.popleft()
            self.queue.put_nowait(event)
            self.state_changed.set()
        else:
            row["sent_at_monotonic_ns"] = timestamp
            if event.get("type") == "input_audio_buffer.append":
                self.last_input_append_ns = int(timestamp)
        self.trace_rows.append(row)
        return row

    async def receive_loop(self, ws: Any) -> None:
        try:
            while True:
                raw = await ws.recv()
                received_ns = time.monotonic_ns()
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError("WebSocket event must be a JSON object")
                if event.get("type") == "session.created" and event.get("session_id"):
                    self.session_id = str(event["session_id"])
                self.record("server", event, received_ns)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.receive_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.closed.set()

    async def wait_for(self, types: set[str], timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"waiting for {sorted(types)} timed out")
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"waiting for {sorted(types)} timed out") from exc
            if event.get("type") in types:
                return event
            if event.get("type") == "error":
                raise RuntimeError(f"server error while waiting for {sorted(types)}: {event.get('error')!r}")

    def output_pcm(self) -> np.ndarray:
        return np.concatenate(self.output_parts) if self.output_parts else np.empty(0, dtype=np.float32)


async def _send(ws: Any, collector: _SessionCollector, event: dict[str, Any], **extra: Any) -> None:
    timestamp = time.monotonic_ns()
    await ws.send(json.dumps(event, ensure_ascii=False))
    collector.record("client", event, timestamp, **extra)


async def _setup(
    ws: Any,
    collector: _SessionCollector,
    instructions: str,
    policy: str,
    *,
    input_audio_transcription: bool = False,
) -> bool:
    await collector.wait_for({"session.created"}, 15.0)
    await _send(ws, collector, {"type": "session.update", "session": {
        "voice": "female_it", "instructions": instructions, "auto_response": True,
        "turn_policy": policy, "input_audio_buffer_acknowledgements": True,
        "input_audio_transcription": bool(input_audio_transcription),
    }})
    updated = await collector.wait_for({"session.updated"}, 15.0)
    if updated.get("session_id"):
        collector.session_id = str(updated["session_id"])
    enabled = updated.get("input_audio_transcription_enabled")
    if enabled is None and not input_audio_transcription:
        return False
    if not isinstance(enabled, bool):
        raise RuntimeError("session.updated omitted input_audio_transcription_enabled")
    if enabled != bool(input_audio_transcription):
        if input_audio_transcription:
            raise RuntimeError("server did not enable native input transcription")
        raise RuntimeError("server enabled native input transcription without an explicit request")
    return enabled



async def _append_input_frame(
    ws: Any, collector: _SessionCollector, frame: np.ndarray, *, pace: bool,
    real_samples: int, silence_frame: bool = False,
) -> int:
    from client.duplex_dialogue_harness import float_to_pcm16_bytes
    audio_frame = np.asarray(frame, dtype=np.float32).reshape(-1)
    if audio_frame.size != FRAME_SAMPLES:
        raise ValueError(f"input frame must contain exactly {FRAME_SAMPLES} samples")
    previous_ns = collector.last_input_append_ns
    if pace and previous_ns is not None:
        remaining_s = FRAME_MS / 1000.0 - (time.monotonic_ns() - previous_ns) / 1_000_000_000
        if remaining_s > 0:
            await asyncio.sleep(remaining_s)
    encoded = base64.b64encode(float_to_pcm16_bytes(audio_frame)).decode("ascii")
    timestamp = time.monotonic_ns()
    input_frame_id = collector.input_frames_sent
    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": encoded}))
    collector.input_frames_sent += 1
    collector.record(
        "client", {"type": "input_audio_buffer.append", "audio_samples": FRAME_SAMPLES}, timestamp,
        audio_samples=int(real_samples), padded_samples=int(FRAME_SAMPLES - real_samples),
        silence_frame=bool(silence_frame), input_frame_id=input_frame_id,
    )
    collector.input_audio_stream_parts.append(audio_frame.copy())
    return timestamp


async def _stream_user_audio(
    ws: Any, collector: _SessionCollector, pcm: np.ndarray, *, pace: bool = True,
    on_first_frame: Callable[[int], None] | None = None,
) -> None:
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    for offset in range(0, audio.size, FRAME_SAMPLES):
        real_frame = audio[offset:offset + FRAME_SAMPLES]
        frame = real_frame
        if frame.size < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - frame.size))
        timestamp = await _append_input_frame(
            ws, collector, frame, pace=pace, real_samples=int(real_frame.size),
        )
        if on_first_frame is not None and offset == 0:
            on_first_frame(timestamp)


def _last_voiced_frame_offset(pcm: np.ndarray) -> int | None:
    """Return the last frame with speech energy, or None for an all-silence clip."""
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    last_voiced: int | None = None
    for frame_id, start in enumerate(range(0, audio.size, FRAME_SAMPLES)):
        frame = audio[start:start + FRAME_SAMPLES]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        peak = float(np.max(np.abs(frame)))
        if rms >= 0.001 or peak >= 0.005:
            last_voiced = frame_id
    return last_voiced


async def _stream_endpoint_silence(
    ws: Any, collector: _SessionCollector, *, pace: bool, frames: int = 10,
) -> None:
    """Keep the microphone clock live for the configured 800 ms SPEAK endpoint."""
    if frames <= 0:
        return
    zero_frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    for _ in range(int(frames)):
        await _append_input_frame(
            ws, collector, zero_frame, pace=pace,
            real_samples=FRAME_SAMPLES, silence_frame=True,
        )


async def _input_silence_pump(ws: Any, collector: _SessionCollector, *, pace: bool) -> None:
    zero_frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    while True:
        await _append_input_frame(
            ws, collector, zero_frame, pace=pace, real_samples=FRAME_SAMPLES, silence_frame=True,
        )
        if not pace:
            await asyncio.sleep(0)


async def _wait_response_with_silence(
    ws: Any, collector: _SessionCollector, *, need_audio: bool = False,
    timeout_s: float = 180.0, pace: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    created: dict[str, Any] | None = None
    silence_task = asyncio.create_task(_input_silence_pump(ws, collector, pace=pace))
    try:
        created = await collector.wait_for({"response.created"}, min(45.0, timeout_s))
        if need_audio:
            await collector.wait_for({"response.audio.delta"}, min(45.0, timeout_s))
        end = await collector.wait_for({"response.done", "response.cancelled"}, timeout_s)
        return created, end
    except (TimeoutError, RuntimeError) as exc:
        collector.trace_rows.append({
            "direction": "local", "session_id": collector.session_id,
            "event": {"type": "wait_failure", "message": str(exc), "timeout_s": timeout_s},
            "at_monotonic_ns": time.monotonic_ns(),
        })
        return created, None
    finally:
        silence_task.cancel()
        await asyncio.gather(silence_task, return_exceptions=True)


def _active_speak_response(collector: _SessionCollector) -> dict[str, Any] | None:
    for row in reversed(collector.server_events):
        event = row.get("event", {})
        if event.get("type") != "response.created":
            continue
        response_id = str(event.get("response_id", ""))
        if (
            response_id
            and collector.response_actions.get(response_id) == "SPEAK"
            and collector.playout.is_response_playout_active(response_id)
        ):
            return event
    return None


async def _wait_for_speak_playout_active(
    ws: Any, collector: _SessionCollector, *, timeout_s: float = 45.0, pace: bool = True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    silence_task = asyncio.create_task(_input_silence_pump(ws, collector, pace=pace))
    try:
        while True:
            response = _active_speak_response(collector)
            if response is not None:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no SPEAK response entered PCM playout before the barge-in timeout")
            collector.state_changed.clear()
            try:
                await asyncio.wait_for(collector.state_changed.wait(), timeout=min(0.01, remaining))
            except asyncio.TimeoutError:
                pass
    finally:
        silence_task.cancel()
        await asyncio.gather(silence_task, return_exceptions=True)


async def _wait_response_terminal_with_silence(
    ws: Any, collector: _SessionCollector, response_id: str, *, timeout_s: float = 180.0,
    pace: bool = True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    silence_task = asyncio.create_task(_input_silence_pump(ws, collector, pace=pace))
    try:
        while True:
            for row in collector.server_events:
                event = row.get("event", {})
                if event.get("response_id") == response_id and event.get("type") in {
                    "response.done", "response.cancelled", "error",
                }:
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"response {response_id} did not reach a terminal event in {timeout_s:.1f}s")
            collector.state_changed.clear()
            try:
                await asyncio.wait_for(collector.state_changed.wait(), timeout=min(FRAME_MS / 1000.0, remaining))
            except asyncio.TimeoutError:
                pass
    finally:
        silence_task.cancel()
        await asyncio.gather(silence_task, return_exceptions=True)


async def _wait_for_speak_with_silence(
    ws: Any, collector: _SessionCollector, *, after_event_index: int,
    timeout_s: float = 180.0, pace: bool = True,
    last_voiced_input_frame_id: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Wait for a causally complete response, keeping microphone PCM live.

    BACKCHANNEL and other response terminals do not satisfy a user turn. Deferred
    SPEAK may be dispatched after a BACKCHANNEL finishes, so continue pumping
    microphone silence until that SPEAK reaches a terminal event or this explicit
    timeout expires. If the WAV has a known final voiced frame, ignore responses
    whose endpoint predates it so an intra-utterance pause cannot end the turn.
    """
    deadline = time.monotonic() + timeout_s
    silence_task = asyncio.create_task(_input_silence_pump(ws, collector, pace=pace))
    ignored_response_ids: set[str] = set()
    try:
        while True:
            collector.state_changed.clear()
            rows = collector.server_events[after_event_index:]
            selected_created: dict[str, Any] | None = None
            selected_id = ""
            selected_action: str | None = None
            selected_endpoint: int | None = None
            for row in rows:
                event = row.get("event", {})
                if event.get("type") != "response.created":
                    continue
                response_id = str(event.get("response_id", ""))
                action = collector.response_actions.get(response_id)
                if not response_id or action not in {"SPEAK", "REPLAN"}:
                    continue
                endpoint = event.get("intent_endpoint_frame_id")
                endpoint_id = endpoint if isinstance(endpoint, int) and not isinstance(endpoint, bool) else None
                if (
                    last_voiced_input_frame_id is not None
                    and (endpoint_id is None or endpoint_id < last_voiced_input_frame_id)
                ):
                    if response_id not in ignored_response_ids:
                        ignored_response_ids.add(response_id)
                        collector.trace_rows.append({
                            "direction": "local", "session_id": collector.session_id,
                            "event": {
                                "type": "turn_completion_candidate_ignored",
                                "response_id": response_id, "action": action,
                                "intent_endpoint_frame_id": endpoint_id,
                                "last_voiced_input_frame_id": last_voiced_input_frame_id,
                            },
                            "at_monotonic_ns": time.monotonic_ns(),
                        })
                    continue
                selected_created = event
                selected_id = response_id
                selected_action = action
                selected_endpoint = endpoint_id
                break
            if selected_created is not None:
                for row in rows:
                    event = row.get("event", {})
                    if event.get("response_id") == selected_id and event.get("type") in {
                        "response.done", "response.cancelled", "error",
                    }:
                        if last_voiced_input_frame_id is not None:
                            collector.trace_rows.append({
                                "direction": "local", "session_id": collector.session_id,
                                "event": {
                                    "type": "turn_completion_gate",
                                    "response_id": selected_id, "action": selected_action,
                                    "last_voiced_input_frame_id": last_voiced_input_frame_id,
                                    "intent_endpoint_frame_id": selected_endpoint,
                                    "terminal_type": event.get("type"),
                                },
                                "at_monotonic_ns": time.monotonic_ns(),
                            })
                        return selected_created, event

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                collector.trace_rows.append({
                    "direction": "local", "session_id": collector.session_id,
                    "event": {
                        "type": "wait_failure",
                        "message": f"no terminal SPEAK response within {timeout_s:.1f}s",
                        "timeout_s": timeout_s,
                    },
                    "at_monotonic_ns": time.monotonic_ns(),
                })
                return selected_created, None
            # The event is cleared before inspecting state. If a new event arrives
            # after this scan, it sets the event and the wait returns immediately.
            # In particular, an already-seen SPEAK must still await its terminal;
            # spinning on that persistent created row would starve the PCM task.
            try:
                await asyncio.wait_for(collector.state_changed.wait(), timeout=min(0.05, remaining))
            except asyncio.TimeoutError:
                pass
    finally:
        silence_task.cancel()
        await asyncio.gather(silence_task, return_exceptions=True)


async def _wait_response(
    ws: Any, collector: _SessionCollector, *, need_audio: bool = False, pace: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return await _wait_response_with_silence(
        ws, collector, need_audio=need_audio, timeout_s=180.0, pace=pace,
    )


async def _drain_active_responses_with_silence(
    ws: Any, collector: _SessionCollector, *, pace: bool, timeout_s: float = 180.0,
) -> list[str]:
    failures: list[str] = []
    deadline = time.monotonic() + timeout_s
    while collector.active_responses:
        for response_id in list(collector.active_responses):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                terminal = await _wait_response_terminal_with_silence(
                    ws, collector, response_id, timeout_s=remaining, pace=pace,
                )
                if terminal.get("type") == "error":
                    code = terminal.get("error", {}).get("code", "unknown")
                    failures.append(f"response {response_id} ended with server error {code}")
            except TimeoutError as exc:
                failures.append(str(exc))
                collector.trace_rows.append({
                    "direction": "local", "session_id": collector.session_id,
                    "event": {"type": "wait_failure", "message": str(exc), "response_id": response_id,
                              "timeout_s": timeout_s},
                    "at_monotonic_ns": time.monotonic_ns(),
                })
                return failures
    if collector.active_responses:
        for response_id in sorted(collector.active_responses):
            message = f"response {response_id} did not reach a terminal event before drain timeout"
            failures.append(message)
            collector.trace_rows.append({
                "direction": "local", "session_id": collector.session_id,
                "event": {"type": "wait_failure", "message": message, "response_id": response_id,
                          "timeout_s": timeout_s},
                "at_monotonic_ns": time.monotonic_ns(),
            })
    return failures


async def _run_session(
    *, scenario: DialogueScenario, assets: dict[str, Path], label: str, ws_url: str,
    policy: str, manifest: dict[str, Any], output_dir: Path, trace_rows: list[dict[str, Any]],
    pace: bool = True, special_barge: bool = False,
    input_audio_transcription: bool = False,
) -> dict[str, Any]:
    import websockets

    inputs: list[np.ndarray] = []
    references: list[str] = []
    collector = _SessionCollector(label, output_dir, trace_rows)
    session_result: dict[str, Any] = {"label": label, "scenario_id": scenario.scenario_id,
                                      "session_id": label, "responses": [], "errors": [],
                                      "input_audio_transcription_requested": bool(input_audio_transcription)}
    user_steps = [step for step in scenario.steps if step.speaker == "user"]
    assistant_steps = [step for step in scenario.steps if step.speaker == "assistant"]
    try:
        async with websockets.connect(ws_url, max_size=50_000_000, ping_interval=20, ping_timeout=20) as ws:
            await collector.playout.start()
            receiver = asyncio.create_task(collector.receive_loop(ws))
            try:
                transcription_enabled = await _setup(
                    ws, collector, scenario.system_prompt_b, policy,
                    input_audio_transcription=input_audio_transcription,
                )
                session_result["input_audio_transcription_enabled"] = transcription_enabled
                session_result["session_id"] = collector.session_id
                live_identity = validate_model_loaded_identity(manifest, collector.server_events, policy)
                if live_identity["status"] != "matched":
                    raise RuntimeError(
                        "live model_loaded identity unavailable or mismatched; refusing to send scenario audio: "
                        f"{live_identity.get('reason')} {live_identity.get('mismatches', [])}"
                    )
                if special_barge and len(user_steps) >= 2:
                    first_pcm = _decode_input_wav(assets[user_steps[0].step_id])
                    inputs.append(first_pcm)
                    references.append(user_steps[0].expected_text)
                    await _stream_user_audio(ws, collector, first_pcm, pace=pace)
                    try:
                        first_created = await _wait_for_speak_playout_active(
                            ws, collector, timeout_s=90.0, pace=pace,
                        )
                    except TimeoutError as exc:
                        first_created = None
                        session_result["barge_trigger_status"] = "timeout"
                        session_result["errors"].append(str(exc))
                        session_result["errors"].extend(
                            await _drain_active_responses_with_silence(ws, collector, pace=pace)
                        )
                    first_id = str((first_created or {}).get("response_id", ""))
                    if first_id:
                        session_result["responses"].append(first_id)
                    if first_id:
                        session_result["barge_trigger_status"] = "speak_playout_active"
                        session_result["barge_trigger_action"] = collector.response_actions.get(first_id)
                        session_result["barge_response_id"] = first_id
                        barge_pcm = _decode_input_wav(assets[user_steps[1].step_id])
                        inputs.append(barge_pcm)
                        references.append(user_steps[1].expected_text)

                        def record_barge_start(timestamp: int) -> None:
                            session_result["barge_in_start_ns"] = timestamp
                            session_result["overlap_at_barge_start"] = bool(
                                first_id in collector.active_responses
                                and collector.playout.is_response_playout_active(first_id)
                            )

                        await _stream_user_audio(
                            ws, collector, barge_pcm, pace=pace, on_first_frame=record_barge_start,
                        )
                        session_result["barge_input_completed_ns"] = time.monotonic_ns()
                        try:
                            first_end = await _wait_response_terminal_with_silence(
                                ws, collector, first_id, timeout_s=45.0, pace=pace,
                            )
                        except TimeoutError as exc:
                            first_end = None
                            session_result["errors"].append(str(exc))
                        first_end_row = next((row for row in collector.server_events
                                              if row["event"].get("response_id") == first_id
                                              and row["event"].get("type") in {"response.cancelled", "response.done"}), None)
                        session_result["barge_response_end_ns"] = (
                            first_end_row.get("received_at_monotonic_ns") if first_end_row else None
                        )
                        session_result["barge_cancel_observed"] = bool(
                            first_end and first_end.get("type") == "response.cancelled"
                        )
                        post_interrupt_audio_ns = [
                            row["received_at_monotonic_ns"] for row in collector.server_events
                            if row["event"].get("type") == "response.audio.delta"
                            and row["event"].get("response_id") == first_id
                            and row["received_at_monotonic_ns"] >= session_result["barge_in_start_ns"]
                        ]
                        session_result["barge_last_audio_ns"] = max(post_interrupt_audio_ns, default=None)
                        try:
                            second_created = await _wait_for_speak_playout_active(
                                ws, collector, timeout_s=90.0, pace=pace,
                            )
                            second_id = str(second_created.get("response_id", ""))
                            if second_id:
                                session_result["responses"].append(second_id)
                            session_result["recovery_response_created"] = bool(second_created)
                            session_result["recovery_response_action"] = collector.response_actions.get(second_id)
                            session_result["recovery_response_end"] = await _wait_response_terminal_with_silence(
                                ws, collector, second_id, timeout_s=180.0, pace=pace,
                            )
                        except TimeoutError as exc:
                            session_result["recovery_response_created"] = False
                            session_result["recovery_response_end"] = None
                            session_result["errors"].append(str(exc))
                        session_result["errors"].extend(
                            await _drain_active_responses_with_silence(ws, collector, pace=pace)
                        )
                    else:
                        session_result["recovery_response_created"] = False
                else:
                    for step in user_steps:
                        pcm = _decode_input_wav(assets[step.step_id])
                        inputs.append(pcm)
                        references.append(step.expected_text)
                        event_start_index = len(collector.server_events)
                        first_input_frame_id = collector.input_frames_sent
                        last_voiced_offset = _last_voiced_frame_offset(pcm)
                        last_voiced_input_frame_id = (
                            first_input_frame_id + last_voiced_offset
                            if last_voiced_offset is not None else None
                        )
                        await _stream_user_audio(ws, collector, pcm, pace=pace)
                        await _stream_endpoint_silence(ws, collector, pace=pace)
                        created, end = await _wait_for_speak_with_silence(
                            ws, collector, after_event_index=event_start_index,
                            timeout_s=180.0, pace=pace,
                            last_voiced_input_frame_id=last_voiced_input_frame_id,
                        )
                        if created:
                            session_result["responses"].append(str(created.get("response_id", "")))
                        if end is None:
                            session_result["errors"].append(
                                "SPEAK response did not reach a terminal event within the wait window"
                            )
                        elif end.get("type") == "error":
                            code = end.get("error", {}).get("code", "unknown")
                            session_result["errors"].append(
                                f"SPEAK response {created.get('response_id') if created else ''} ended with server error {code}"
                            )
                        session_result["errors"].extend(
                            await _drain_active_responses_with_silence(ws, collector, pace=pace)
                        )
                        if end is None:
                            break
                session_result["errors"].extend(
                    await _drain_active_responses_with_silence(ws, collector, pace=pace)
                )
                created_ids = [
                    str(row["event"].get("response_id", "")) for row in collector.server_events
                    if row["event"].get("type") == "response.created" and row["event"].get("response_id")
                ]
                session_result["responses"] = list(dict.fromkeys(
                    [*session_result["responses"], *created_ids]
                ))
                await asyncio.sleep(0.1)
            finally:
                try:
                    session_result["playout_simulation"] = await collector.playout.finish(timeout_s=180.0)
                except Exception as exc:
                    session_result["errors"].append(
                        f"simulated playback drain failed: {type(exc).__name__}: {exc}"
                    )
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    except Exception as exc:
        session_result["errors"].append(f"{type(exc).__name__}: {exc}")
        trace_rows.append({"direction": "local", "session_id": collector.session_id,
                           "event": {"type": "session_failure", "message": f"{type(exc).__name__}: {exc}"},
                           "at_monotonic_ns": time.monotonic_ns()})
    session_result["session_id"] = collector.session_id
    session_result["inputs"] = inputs
    session_result["input_references"] = references
    session_result["input_audio_stream"] = collector.input_audio_stream_parts
    session_result["output_parts"] = collector.output_parts
    session_result["output_parts_by_response"] = dict(collector.output_parts_by_response)
    session_result["output_packet_boundaries"] = collector.output_packet_boundaries
    session_result["output_rate"] = collector.output_rate
    session_result["text_by_response"] = {key: "".join(parts) for key, parts in collector.text_by_response.items()}
    session_result["receive_error"] = collector.receive_error
    session_result["input_audio_transcription"] = summarize_input_audio_transcription(
        [row for row in trace_rows if row.get("session_id") == collector.session_id],
        requested=input_audio_transcription,
        enabled=bool(session_result.get("input_audio_transcription_enabled", False)),
    )
    session_result.setdefault("playout_simulation", collector.playout.snapshot())
    return session_result


def _resolve_asr_backend(model_path: str | Path, backend: str) -> str:
    """Select a local ASR loader from the checkpoint's on-disk format."""
    path = Path(model_path).expanduser()
    has_transformers_weights = (path / "config.json").is_file() and any(path.glob("*.safetensors"))
    has_faster_whisper_weights = (path / "model.bin").is_file()
    if backend == "auto":
        if has_transformers_weights:
            return "transformers"
        if has_faster_whisper_weights:
            return "faster-whisper"
        raise ValueError(
            "local ASR directory must contain either config.json plus safetensors "
            "or a Faster-Whisper model.bin"
        )
    if backend == "transformers":
        if not has_transformers_weights:
            raise ValueError("local Transformers Whisper checkpoint requires config.json and safetensors weights")
        return backend
    if backend == "faster-whisper":
        if not has_faster_whisper_weights:
            raise ValueError("local Faster-Whisper checkpoint requires model.bin")
        return backend
    raise ValueError(f"unsupported ASR backend: {backend!r}")


def _load_local_asr(
    model_path: str | None, device: str, backend: str = "auto"
) -> tuple[Callable[[np.ndarray, int], str] | None, str | None, str | None]:
    if not model_path:
        return None, "no local ASR model configured", None
    path = Path(model_path).expanduser()
    if not path.exists():
        return None, f"ASR model path does not exist: {path}", None
    try:
        selected_backend = _resolve_asr_backend(path, backend)
        if selected_backend == "transformers":
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA ASR requested but no CUDA device is available")
            asr_device = torch.device("cuda:0" if device == "cuda" else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(path), torch_dtype=dtype, low_cpu_mem_usage=True,
                use_safetensors=True, local_files_only=True,
            ).to(asr_device)
            processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
            model.eval()
            recognizer = pipeline(
                "automatic-speech-recognition", model=model,
                tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                dtype=dtype, device=asr_device,
            )

            def transcribe(pcm: np.ndarray, sample_rate: int) -> str:
                audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
                if sample_rate != INPUT_RATE:
                    try:
                        from scipy.signal import resample_poly
                        from math import gcd
                        divisor = gcd(sample_rate, INPUT_RATE)
                        audio = resample_poly(audio, INPUT_RATE // divisor, sample_rate // divisor).astype(np.float32)
                    except ImportError as exc:
                        raise RuntimeError("scipy is required to resample captured output audio for ASR") from exc
                result = recognizer(
                    {"raw": audio, "sampling_rate": INPUT_RATE},
                    generate_kwargs={"language": "italian", "task": "transcribe", "num_beams": 1,
                                     "max_new_tokens": 256},
                )
                return str(result.get("text", "")).strip()

            return transcribe, None, selected_backend

        from faster_whisper import WhisperModel
        model = WhisperModel(str(path), device=device, compute_type="int8" if device == "cpu" else "float16")
    except Exception as exc:
        return None, f"could not initialize local {backend} ASR model: {type(exc).__name__}: {exc}", None

    def transcribe(pcm: np.ndarray, sample_rate: int) -> str:
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if sample_rate != INPUT_RATE:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                divisor = gcd(sample_rate, INPUT_RATE)
                audio = resample_poly(audio, INPUT_RATE // divisor, sample_rate // divisor).astype(np.float32)
            except ImportError as exc:
                raise RuntimeError("scipy is required to resample captured output audio for ASR") from exc
        segments, _ = model.transcribe(audio, language="it", beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    return transcribe, None, selected_backend


def _read_captured_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read the mono PCM16 WAVs emitted by this harness for offline ASR."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"captured WAV must be mono PCM16: {path}")
        sample_rate = int(wav.getframerate())
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate <= 0 or not pcm.size:
        raise ValueError(f"captured WAV is empty or has an invalid sample rate: {path}")
    return pcm, sample_rate


def _resolve_capture_artifact(raw_path: Any, report_path: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None
    candidates = (Path.cwd() / path, report_path.parent / path, report_path.parent / path.name)
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def enrich_report_with_offline_asr(
    report: dict[str, Any],
    report_path: str | Path,
    *,
    model_path: str,
    device: str,
    backend: str,
) -> dict[str, Any]:
    """Add local ASR transcripts/metrics to an already captured report, without a server connection."""
    report_file = Path(report_path).expanduser().resolve()
    transcribe, init_error, selected_backend = _load_local_asr(model_path, device, backend)
    asr_error = init_error

    def transcribe_artifact(raw_path: Any) -> tuple[str | None, str | None]:
        if transcribe is None:
            return None, asr_error or "ASR model was not initialized"
        resolved = _resolve_capture_artifact(raw_path, report_file)
        if resolved is None:
            return None, f"captured WAV not found: {raw_path!r}"
        try:
            pcm, sample_rate = _read_captured_wav(resolved)
            text = transcribe(pcm, sample_rate)
            return (text if text.strip() else None), (None if text.strip() else "ASR returned empty text")
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    session_audio = report.get("artifacts", {}).get("session_audio", {})
    canonical_turns = report.get("behavior", {}).get("turns", [])
    asr_states: list[str] = []
    for turn in canonical_turns:
        paths = session_audio.get(turn.get("session_id"), {})
        input_text, input_error = transcribe_artifact(paths.get("input_wav"))
        output_text, output_error = transcribe_artifact(paths.get("output_wav"))
        turn["asr_input_text"] = input_text
        turn["asr_output_text"] = output_text
        errors = [error for error in (input_error, output_error) if error]
        turn["asr_metrics"] = compare_asr_metrics(
            str(turn.get("reference_text", "")), input_text or "",
            str(turn.get("response_text", "")), output_text or "",
            configured=True, error="; ".join(errors) or None,
        )
        asr_states.append(turn["asr_metrics"]["status"])

    for probe in report.get("qualitative_probes", {}).get("probes", []):
        for turn in probe.get("turns", []):
            input_text, input_error = transcribe_artifact(turn.get("input_wav"))
            output_text, output_error = transcribe_artifact(turn.get("response_wav"))
            turn["input_asr_text"] = input_text
            turn["response_asr_text"] = output_text
            errors = [error for error in (input_error, output_error) if error]
            turn["asr_metrics"] = compare_asr_metrics(
                str(turn.get("input_reference_text", "")), input_text or "",
                str(turn.get("response_text", "")), output_text or "",
                configured=True, error="; ".join(errors) or None,
            )
            turn["asr_error"] = "; ".join(errors) or None

    available = bool(asr_states) and all(status == "available" for status in asr_states)
    reason = None if available else (
        asr_error or "offline ASR did not successfully transcribe both captured WAVs for every canonical scenario"
    )
    report.setdefault("behavior", {})["asr"] = {
        "status": "available" if available else "unavailable",
        "reason": reason,
        "configured_model_path": str(Path(model_path).expanduser().resolve()),
        "backend": selected_backend,
        "device": device,
        "execution": "offline_after_server_stop",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report.setdefault("asr_enrichment", {}).update({
        "status": "available" if available else "incomplete",
        "execution": "offline_after_server_stop",
        "report_source": str(report_file),
    })
    return report


def _save_playout_simulation_artifacts(
    result: dict[str, Any], output_dir: Path, safe_label: str,
) -> dict[str, Any]:
    simulation = result.get("playout_simulation")
    if not isinstance(simulation, dict):
        return {"status": "unavailable", "responses": {}}
    responses: dict[str, Any] = {}
    metrics_by_id = simulation.get("responses", {})
    played_by_id = simulation.get("played_pcm_by_response", {})
    cancelled_by_id = simulation.get("cancelled_pcm_by_response", {})
    for response_id, metrics in metrics_by_id.items():
        safe_response_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(response_id)) or "response"
        played = np.asarray(played_by_id.get(response_id, np.empty(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        cancelled = np.asarray(cancelled_by_id.get(response_id, np.empty(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        played_path = output_dir / f"{safe_label}__{safe_response_id}_playout.wav"
        cancelled_path = output_dir / f"{safe_label}__{safe_response_id}_cleared.wav"
        if played.size:
            _write_wav(played_path, played, int(simulation.get("sample_rate_hz", OUTPUT_RATE)))
        if cancelled.size:
            _write_wav(cancelled_path, cancelled, int(simulation.get("sample_rate_hz", OUTPUT_RATE)))
        responses[str(response_id)] = {
            "metrics": metrics,
            "played_wav": str(played_path) if played.size else None,
            "cancelled_wav": str(cancelled_path) if cancelled.size else None,
        }
    return {
        "status": "available",
        "endpoint": simulation.get("endpoint", "simulated_headless_pcm_playout"),
        "sample_rate_hz": simulation.get("sample_rate_hz", OUTPUT_RATE),
        "frame_ms": simulation.get("frame_ms", FRAME_MS),
        "physical_device_playback": bool(simulation.get("physical_device_playback", False)),
        "responses": responses,
    }


async def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    requested_scenario_ids = getattr(args, "scenario_ids", None)
    requested_scenario_ids = list(requested_scenario_ids) if requested_scenario_ids else None
    all_canonical_ids = [scenario.scenario_id for scenario in ALL_SCENARIOS]
    is_targeted_scope = requested_scenario_ids is not None
    report: dict[str, Any] = {
        "schema_version": 1, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested": {"checkpoint_root": str(Path(args.checkpoint_root).resolve()), "server": args.server,
                      "turn_policy": args.turn_policy, "expected_run_id": args.expected_run_id,
                      "expected_bundle_id": getattr(args, "expected_bundle_id", None),
                      "scenario_ids": requested_scenario_ids},
        "identity": {"status": "unavailable", "requested_manifest": None, "observed_model_loaded": None},
        "system": {
            "status": "not_run", "transport": None,
            "playback_simulation": {
                "status": "unavailable", "endpoint": "simulated_headless_pcm_playout",
                "sample_rate_hz": OUTPUT_RATE, "frame_ms": FRAME_MS,
                "physical_device_playback": False, "device_latency_measured": False,
                "responses": [],
            },
        },
        "behavior": {"status": "unavailable", "action_coverage": None, "session_isolation": None,
                      "turns": [], "asr": {"status": "unavailable", "reason": "not run"}},
        "scenario_coverage": {
            "scope": "targeted_subset" if is_targeted_scope else "full_canonical_six",
            "selected": requested_scenario_ids or all_canonical_ids,
            "expected": all_canonical_ids,
            "run": [], "missing": list(all_canonical_ids),
            "skipped": [], "status": "partial" if is_targeted_scope else "incomplete",
        },
        "qualitative_probes": {
            "status": "not_requested", "assessment": "manual_review_only",
            "review_guidance": [
                "Check whether the assistant notices that Tuesday 16:30 conflicts with the user's swimming commitment.",
                "Check whether it clarifies the afternoon time range without assuming a specific slot.",
                "Check whether the latest correction to Friday after 15:00 updates the earlier Tuesday preference.",
            ],
            "run": [], "missing": [], "skipped": [], "probes": [],
        },
        "artifacts": {},
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    qualitative_trace_rows: list[dict[str, Any]] = []
    report["artifacts"]["trace_jsonl"] = str(output_dir / "trace.jsonl")
    manifest = None
    try:
        manifest = load_checkpoint_manifest(args.checkpoint_root)
        if manifest.get("_response_bundle"):
            expected_bundle_id = getattr(args, "expected_bundle_id", None)
            if expected_bundle_id and manifest["bundle_id"] != expected_bundle_id:
                raise ValueError(
                    f"requested bundle_id {expected_bundle_id!r} does not match manifest {manifest['bundle_id']!r}"
                )
            report["identity"]["requested_manifest"] = {
                "path": str(manifest["_manifest_path"]), "bundle_id": manifest["bundle_id"],
                "base_model_id": manifest["base_model_id"],
                "feature_source": manifest["feature_source"], "runtime_abi": manifest["runtime_abi"],
                "adapter_path": str(manifest["adapter_path"]),
                "control_head_path": str(manifest["control_head_path_resolved"]),
                "feature_contract": {key: manifest["control_head"][key] for key in ("input_dim", "pooling", "tick_ms", "actions")},
            }
        else:
            if args.expected_run_id and manifest["run_id"] != args.expected_run_id:
                raise ValueError(f"requested run_id {args.expected_run_id!r} does not match manifest {manifest['run_id']!r}")
            report["identity"]["requested_manifest"] = {
                "path": str(manifest["_manifest_path"]), "run_id": manifest["run_id"],
                "base_model_id": manifest["base_model_id"], "base_model_path": manifest.get("base_model_path"),
                "adapter_path": str(manifest["adapter_path"]), "control_head_path": str(manifest["control_head_path_resolved"]),
                "feature_contract": {key: manifest["control_head"][key] for key in ("input_dim", "pooling", "tick_ms", "actions")},
            }
        audio_dir = Path(args.scenario_audio_dir).expanduser() if args.scenario_audio_dir else DEFAULT_SCENARIO_AUDIO_DIR
        include_qualitative = bool(getattr(args, "include_qualitative_prompts", False))
        canonical_scenarios, qualitative_scenarios = _evaluation_scenarios(include_qualitative)
        canonical_scenarios = _select_canonical_scenarios(
            canonical_scenarios, requested_scenario_ids,
        )
        report["scenario_coverage"]["selected"] = [
            scenario.scenario_id for scenario in canonical_scenarios
        ]
        available, skipped = _scenario_assets_for(audio_dir, canonical_scenarios + qualitative_scenarios)
        canonical_ids = {scenario.scenario_id for scenario in canonical_scenarios}
        canonical_available = [(s, a) for s, a in available if s.scenario_id in canonical_ids]
        qualitative_available = [(s, a) for s, a in available if s.scenario_id not in canonical_ids]
        report["scenario_coverage"]["skipped"] = [item for item in skipped if item["scenario_id"] in canonical_ids]
        report["scenario_coverage"]["execution_mode"] = "sequential_single_session"
        report["scenario_coverage"]["concurrent_sessions"] = []
        report["qualitative_probes"]["status"] = "requested" if include_qualitative else "not_requested"
        report["qualitative_probes"]["skipped"] = [item for item in skipped if item["scenario_id"] not in canonical_ids]
        report["qualitative_probes"]["missing"] = [item["scenario_id"] for item in report["qualitative_probes"]["skipped"]]
        if not available:
            report["system"]["status"] = "not_run_missing_audio"
            report["identity"]["status"] = "unavailable"
            report["identity"]["reason"] = "no complete scenario has valid cached WAV inputs"
            return report

        # Live capture must not compete with the server's GPU.  Independent ASR
        # is deliberately a separate offline pass over the WAV artifacts.
        asr = None
        asr_init_error = "offline ASR not run during live capture; use --offline-asr-report after stopping the server"
        asr_backend = None
        run_results: list[dict[str, Any]] = []
        barge_id = "05_barge_in_interruption"
        for scenario, assets in canonical_available:
            run_results.append(await _run_session(
                scenario=scenario, assets=assets,
                label="barge_primary" if scenario.scenario_id == barge_id else scenario.scenario_id,
                ws_url=args.server,
                policy=args.turn_policy, manifest=manifest, output_dir=output_dir, trace_rows=trace_rows,
                pace=getattr(args, "frame_pacing", True),
                special_barge=scenario.scenario_id == barge_id,
                input_audio_transcription=getattr(args, "input_audio_transcription", False),
            ))
        qualitative_results: list[dict[str, Any]] = []
        for scenario, assets in qualitative_available:
            try:
                qualitative_results.append(await _run_session(
                    scenario=scenario, assets=assets, label=scenario.scenario_id, ws_url=args.server,
                    policy=args.turn_policy, manifest=manifest, output_dir=output_dir,
                    trace_rows=qualitative_trace_rows, pace=getattr(args, "frame_pacing", True),
                    input_audio_transcription=getattr(args, "input_audio_transcription", False),
                ))
            except Exception as exc:
                qualitative_results.append({
                    "scenario_id": scenario.scenario_id, "label": scenario.scenario_id,
                    "session_id": scenario.scenario_id, "errors": [f"{type(exc).__name__}: {exc}"],
                    "inputs": [], "input_references": [], "responses": [], "text_by_response": {},
                    "input_audio_transcription": {
                        "status": "not_enabled" if getattr(args, "input_audio_transcription", False) else "not_requested",
                        "requested": bool(getattr(args, "input_audio_transcription", False)),
                        "enabled": False, "partial_events": 0, "completed_events": 0,
                        "utterances": 0, "revision_monotonic": None,
                        "first_partial_from_utterance_onset_ms": [],
                        "first_partial_delivery_lag_ms": [],
                    },
                })
        if include_qualitative:
            report["artifacts"]["qualitative_trace_jsonl"] = str(output_dir / "qualitative_trace.jsonl")
            report["qualitative_probes"]["run"] = [result["scenario_id"] for result in qualitative_results]
            expected_qualitative_turns = {
                scenario.scenario_id: sum(step.speaker == "user" for step in scenario.steps)
                for scenario in qualitative_scenarios
            }
            def captured_all_qualitative_audio(result: dict[str, Any]) -> bool:
                expected_turns = expected_qualitative_turns.get(result["scenario_id"], 0)
                response_ids = result.get("responses", [])
                response_text = result.get("text_by_response", {})
                response_audio = result.get("output_parts_by_response", {})
                return bool(
                    expected_turns
                    and not result.get("errors")
                    and len(response_ids) >= expected_turns
                    and all(response_text.get(response_id, "").strip() for response_id in response_ids[:expected_turns])
                    and all(response_audio.get(response_id) for response_id in response_ids[:expected_turns])
                )
            report["qualitative_probes"]["status"] = (
                "captured" if qualitative_results and all(captured_all_qualitative_audio(r) for r in qualitative_results)
                else "incomplete"
            )
        report["scenario_coverage"]["run"] = [r["scenario_id"] for r in run_results]
        expected_scenarios = [scenario.scenario_id for scenario in ALL_SCENARIOS]
        report["scenario_coverage"]["expected"] = expected_scenarios
        report["scenario_coverage"]["missing"] = [
            scenario_id for scenario_id in expected_scenarios
            if scenario_id not in report["scenario_coverage"]["run"]
        ]
        report["scenario_coverage"]["status"] = (
            "partial" if is_targeted_scope
            else "complete" if not report["scenario_coverage"]["missing"]
            else "incomplete"
        )
        # Persist all observations, retaining original server payloads and local monotonic timestamps.
        with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for row in trace_rows:
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        all_rows = list(trace_rows)
        identity = validate_loaded_identity(manifest, all_rows, args.turn_policy)
        report["identity"].update(identity)
        report["identity"]["observed_model_loaded"] = identity.get("loaded_event")
        report["system"]["status"] = "observed"
        report["system"]["transport"] = compute_event_metrics(all_rows)
        runtime_events = [
            {
                "session_id": row.get("session_id", row.get("session")),
                "event_type": _event(row).get("type"),
                "response_id": _event(row).get("response_id"),
                "metrics": _event(row)["runtime_metrics"],
            }
            for row in all_rows
            if row.get("direction") == "server"
            and _event(row).get("type") in {"response.done", "response.cancelled"}
            and isinstance(_event(row).get("runtime_metrics"), dict)
        ]
        session_runtime_events = [
            {
                "session_id": row.get("session_id", row.get("session")),
                "event_type": "session.created",
                "metrics": _event(row)["runtime_metrics"],
            }
            for row in all_rows
            if row.get("direction") == "server"
            and _event(row).get("type") == "session.created"
            and isinstance(_event(row).get("runtime_metrics"), dict)
        ]
        report["system"]["runtime_metrics"] = {
            "status": "available" if runtime_events or session_runtime_events else "unavailable",
            "response_events": runtime_events,
            "session_events": session_runtime_events,
            "final_scheduler_summary": "server log line [realtime_session_metrics] emitted on socket close",
        }
        report["artifacts"]["session_playout"] = {}
        playback_response_reports: list[dict[str, Any]] = []
        raw_coverage = action_coverage(all_rows, ACTION_ORDER, args.turn_policy)
        coverage = action_coverage(all_rows, ACTION_ORDER, args.turn_policy, accepted_only=True)
        lifecycle_coverage = action_lifecycle_coverage(all_rows, ACTION_ORDER, args.turn_policy)
        report["behavior"]["raw_action_coverage"] = raw_coverage
        report["behavior"]["action_coverage"] = coverage
        report["behavior"]["action_lifecycle_coverage"] = lifecycle_coverage
        report["behavior"]["live_policy_semantics"] = live_policy_semantics(all_rows)
        report["behavior"]["session_isolation"] = {
            "status": "not_tested",
            "reason": "canonical scenarios run sequentially to keep the single RTX 3090 session uncontended",
        }
        report["behavior"]["barge_in"] = {
            "status": "unavailable", "evidence": None,
        }
        barge = next((result for result in run_results if result["scenario_id"] == barge_id), None)
        if barge:
            start_ns = barge.get("barge_in_start_ns")
            end_ns = barge.get("barge_response_end_ns")
            barge_observed = bool(
                start_ns and barge.get("overlap_at_barge_start")
                and barge.get("barge_cancel_observed") and barge.get("recovery_response_created")
            )
            report["behavior"]["barge_in"] = {
                "status": "observed" if barge_observed else "incomplete",
                "overlap_at_barge_start": bool(barge.get("overlap_at_barge_start")),
                "cancel_observed": bool(barge.get("barge_cancel_observed")),
                "interrupt_to_cancel_latency_ms": round((end_ns - start_ns) / 1_000_000.0, 3)
                    if start_ns and end_ns and barge.get("barge_cancel_observed") else None,
                "assistant_last_audio_after_interrupt_ms": round(
                    (barge["barge_last_audio_ns"] - start_ns) / 1_000_000.0, 3
                ) if start_ns and barge.get("barge_last_audio_ns") else None,
                "recovery_response_created": bool(barge.get("recovery_response_created")),
            }
        turn_outputs: list[dict[str, Any]] = []
        for result in run_results:
            streamed_input = result.get("input_audio_stream") or result["inputs"]
            input_audio = np.concatenate(streamed_input) if streamed_input else np.empty(0, dtype=np.float32)
            output_audio = result.pop("output_pcm", None)
            if output_audio is None:
                output_audio = np.concatenate(result["output_parts"]) if result["output_parts"] else np.empty(0, dtype=np.float32)
            out_rate = int(result.get("output_rate") or OUTPUT_RATE)
            session_id = result["session_id"]
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result["label"]))
            input_path = output_dir / f"{safe_label}_input.wav"
            output_path = output_dir / f"{safe_label}_output.wav"
            if input_audio.size:
                _write_wav(input_path, input_audio, INPUT_RATE)
            if output_audio.size:
                _write_wav(output_path, output_audio, out_rate)
            playout_artifacts = _save_playout_simulation_artifacts(result, output_dir, safe_label)
            if playout_artifacts["status"] == "available":
                report["artifacts"]["session_playout"][session_id] = playout_artifacts
                playback_response_reports.extend(
                    {"session_id": session_id, "response_id": response_id, **entry["metrics"]}
                    for response_id, entry in playout_artifacts["responses"].items()
                )
            report["artifacts"].setdefault("session_audio", {})[session_id] = {
                "input_wav": str(input_path) if input_audio.size else None,
                "output_wav": str(output_path) if output_audio.size else None,
            }
            input_metrics = audio_quality(input_audio, INPUT_RATE)
            output_metrics = audio_quality(
                output_audio, out_rate, result.get("output_packet_boundaries", [])
            )
            response_ids = result.get("responses", [])
            texts_by_id = result.get("text_by_response", {})
            output_text = " ".join(texts_by_id.get(response_id, "") for response_id in response_ids).strip()
            asr_input = asr_output = ""
            asr_error = asr_init_error
            if asr is not None:
                try:
                    asr_input = await asyncio.to_thread(asr, input_audio, INPUT_RATE)
                    asr_output = await asyncio.to_thread(asr, output_audio, out_rate) if output_audio.size else ""
                except Exception as exc:
                    asr_error = f"{type(exc).__name__}: {exc}"
            reference = " ".join(result.get("input_references", []))
            turn = {
                "scenario_id": result["scenario_id"], "session_id": session_id,
                "input_audio": input_metrics, "output_audio": output_metrics,
                "playout_simulation": playout_artifacts,
                "reference_text": reference,
                "response_text": output_text,
                "asr_metrics": compare_asr_metrics(reference, asr_input, output_text, asr_output,
                                                   configured=bool(args.asr_model_path), error=asr_error),
                "session_errors": result.get("errors", []),
                "input_audio_transcription": result.get("input_audio_transcription", {
                    "status": "not_requested", "requested": False, "enabled": False,
                }),
            }
            # Scenario validators are text-only checks. Interrupted partial output is not treated as a failed full answer.
            scenario = next((s for s in ALL_SCENARIOS if s.scenario_id == result["scenario_id"]), None)
            assistant_steps = [step for step in scenario.steps if step.speaker == "assistant"] if scenario else []
            if assistant_steps:
                step_results = []
                if result["scenario_id"] == barge_id:
                    step_response_pairs = [(assistant_steps[-1], response_ids[-1:])]
                else:
                    step_response_pairs = [
                        (step, response_ids[index:index + 1])
                        for index, step in enumerate(assistant_steps)
                    ]
                for step, step_response_ids in step_response_pairs:
                    step_text = " ".join(texts_by_id.get(response_id, "") for response_id in step_response_ids).strip()
                    step_checks = [bool(check(step_text)) for check in step.deterministic_checks]
                    step_results.append({
                        "step_id": step.step_id, "response_ids": step_response_ids,
                        "status": "observed" if step_text else "unavailable", "checks": step_checks,
                    })
                turn["text_behavior_checks"] = {
                    "status": "observed" if step_results and all(item["status"] == "observed" for item in step_results) else "incomplete",
                    "steps": step_results,
                }
            turn_outputs.append(turn)
        if include_qualitative:
            scenario_by_id = {scenario.scenario_id: scenario for scenario in qualitative_scenarios}
            for result in qualitative_results:
                scenario = scenario_by_id.get(result["scenario_id"])
                response_ids = result.get("responses", [])
                text_by_id = result.get("text_by_response", {})
                out_rate = int(result.get("output_rate") or OUTPUT_RATE)
                safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.get("label", result["scenario_id"])))
                playout_artifacts = _save_playout_simulation_artifacts(result, output_dir, safe_label)
                if playout_artifacts["status"] == "available":
                    report["artifacts"]["session_playout"][result.get("session_id", safe_label)] = playout_artifacts
                    playback_response_reports.extend(
                        {"session_id": result.get("session_id", safe_label), "response_id": response_id, **entry["metrics"]}
                        for response_id, entry in playout_artifacts["responses"].items()
                    )
                turns = []
                for index, step in enumerate(step for step in (scenario.steps if scenario else []) if step.speaker == "user"):
                    input_pcm = result.get("inputs", [])[index] if index < len(result.get("inputs", [])) else np.empty(0, dtype=np.float32)
                    response_id = response_ids[index] if index < len(response_ids) else None
                    response_text = text_by_id.get(response_id, "") if response_id else ""
                    input_path = output_dir / f"{safe_label}__{step.step_id}_input.wav"
                    if input_pcm.size:
                        _write_wav(input_path, input_pcm, INPUT_RATE)
                    response_audio_parts = result.get("output_parts_by_response", {}).get(response_id, []) if response_id else []
                    response_audio = np.concatenate(response_audio_parts) if response_audio_parts else np.empty(0, dtype=np.float32)
                    response_path = output_dir / f"{safe_label}__response_{index + 1:02d}_output.wav"
                    if response_audio.size:
                        _write_wav(response_path, response_audio, out_rate)
                    response_playout = playout_artifacts.get("responses", {}).get(response_id, {}) if response_id else {}
                    input_asr = None
                    response_asr = None
                    turn_asr_error = asr_init_error if asr is None else None
                    if asr is not None and input_pcm.size:
                        try:
                            input_asr = await asyncio.to_thread(asr, input_pcm, INPUT_RATE)
                        except Exception as exc:
                            input_asr = None
                            turn_asr_error = f"{type(exc).__name__}: {exc}"
                    if asr is not None and response_audio.size:
                        try:
                            response_asr = await asyncio.to_thread(asr, response_audio, out_rate)
                        except Exception as exc:
                            turn_asr_error = f"{type(exc).__name__}: {exc}"
                    if turn_asr_error and asr is not None:
                        result.setdefault("asr_errors", []).append(turn_asr_error)
                    turns.append({
                        "user_step_id": step.step_id,
                        "input_reference_text": step.expected_text,
                        "input_asr_text": input_asr,
                        "response_id": response_id,
                        "response_text": response_text,
                        "response_asr_text": response_asr,
                        "asr_metrics": compare_asr_metrics(
                            step.expected_text, input_asr or "", response_text, response_asr or "",
                            configured=bool(args.asr_model_path), error=turn_asr_error,
                        ),
                        "input_wav": str(input_path) if input_pcm.size else None,
                        "response_wav": str(response_path) if response_audio.size else None,
                        "playout_simulation": response_playout,
                    })
                report["qualitative_probes"]["probes"].append({
                    "scenario_id": result["scenario_id"],
                    "title": scenario.title if scenario else result["scenario_id"],
                    "session_id": result.get("session_id"),
                    "assessment": "manual_review_only",
                    "capture_status": "captured" if all(
                        turn["response_text"] and turn["response_wav"] for turn in turns
                    ) and not result.get("errors") else "incomplete",
                    "turns": turns,
                    "session_errors": result.get("errors", []),
                    "asr_errors": result.get("asr_errors", []),
                    "input_audio_transcription": result.get("input_audio_transcription", {
                        "status": "not_requested", "requested": False, "enabled": False,
                    }),
                })
        report["behavior"]["turns"] = turn_outputs
        transcription_results = [
            result for result in [*run_results, *qualitative_results]
            if result.get("input_audio_transcription", {}).get("requested")
        ]
        if not getattr(args, "input_audio_transcription", False):
            transcription_status = "not_requested"
        elif any(result.get("input_audio_transcription", {}).get("status") == "invalid_events"
                 for result in transcription_results):
            transcription_status = "invalid_events"
        elif any(result.get("input_audio_transcription", {}).get("status") == "not_enabled"
                 for result in transcription_results):
            transcription_status = "not_enabled"
        elif any(result.get("input_audio_transcription", {}).get("status") == "observed"
                 for result in transcription_results):
            transcription_status = "observed"
        else:
            transcription_status = "enabled_no_events"
        report["system"]["input_audio_transcription"] = {
            "status": transcription_status,
            "requested": bool(getattr(args, "input_audio_transcription", False)),
            "sessions": [
                {
                    "session_id": result.get("session_id"),
                    **result.get("input_audio_transcription", {}),
                }
                for result in transcription_results
            ],
            "partial_events": sum(
                int(result.get("input_audio_transcription", {}).get("partial_events", 0))
                for result in transcription_results
            ),
            "completed_events": sum(
                int(result.get("input_audio_transcription", {}).get("completed_events", 0))
                for result in transcription_results
            ),
            "first_partial_from_utterance_onset_ms": [
                latency
                for result in transcription_results
                for latency in result.get("input_audio_transcription", {}).get(
                    "first_partial_from_utterance_onset_ms", []
                )
            ],
            "first_partial_delivery_lag_ms": [
                latency
                for result in transcription_results
                for latency in result.get("input_audio_transcription", {}).get(
                    "first_partial_delivery_lag_ms", []
                )
            ],
        }
        if playback_response_reports:
            report["system"]["playback_simulation"] = {
                "status": "available", "endpoint": "simulated_headless_pcm_playout",
                "sample_rate_hz": OUTPUT_RATE, "frame_ms": FRAME_MS,
                "pcm_source": "actual response.audio.delta PCM received by the harness",
                "physical_device_playback": False, "device_latency_measured": False,
                "responses": playback_response_reports,
            }
        asr_states = [item["asr_metrics"]["status"] for item in turn_outputs]
        report["behavior"]["asr"] = {
            "status": "available" if asr_states and all(state == "available" for state in asr_states) else "unavailable",
            "reason": None if asr_states and all(state == "available" for state in asr_states) else (asr_init_error or "ASR metrics require successful local ASR on both captured input and output audio"),
            "configured_model_path": str(Path(args.asr_model_path).expanduser().resolve()) if args.asr_model_path else None,
            "backend": asr_backend,
            "device": args.asr_device if asr_backend else None,
        }
        text_checks_ok = all(
            turn.get("text_behavior_checks", {}).get("status") == "observed"
            and all(check for step in turn["text_behavior_checks"].get("steps", []) for check in step.get("checks", []))
            for turn in turn_outputs
        )
        if identity["status"] != "matched":
            report["behavior"]["status"] = "unavailable"
        elif (
            coverage["missing"]
            or report["scenario_coverage"]["status"] != "complete"
            or report["behavior"]["barge_in"]["status"] != "observed"
            or any(turn["input_audio"]["status"] != "available" for turn in turn_outputs)
            or any(turn["output_audio"]["status"] != "available" for turn in turn_outputs)
            or any(turn.get("session_errors") for turn in turn_outputs)
            or not text_checks_ok
        ):
            report["behavior"]["status"] = "incomplete"
        else:
            report["behavior"]["status"] = "observed"
        if report["system"]["transport"]["server_errors"]:
            report["system"]["status"] = "failed"
        elif not report["system"]["transport"]["responses_created"]:
            report["system"]["status"] = "incomplete"
        else:
            report["system"]["status"] = "observed"
        report["artifacts"]["report_json"] = str(output_dir / "report.json")
        report["artifacts"]["report_markdown"] = str(output_dir / "report.md")
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["system"]["status"] = "not_run"
        return report
    finally:
        with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for row in trace_rows:
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if qualitative_trace_rows:
            with (output_dir / "qualitative_trace.jsonl").open("w", encoding="utf-8") as trace:
                for row in qualitative_trace_rows:
                    trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _markdown_report(report: dict[str, Any]) -> str:
    identity = report.get("identity", {})
    system = report.get("system", {})
    behavior = report.get("behavior", {})
    transport = system.get("transport") or {}
    coverage = behavior.get("action_coverage") or {}
    lines = [
        "# Gemma QLoRA full-duplex E2E evaluation",
        "",
        f"- Evaluation: **{report.get('evaluation_status', 'incomplete')}**",
        f"- System/transport: **{system.get('status', 'not_run')}**",
        f"- Model behavior: **{behavior.get('status', 'unavailable')}**",
        f"- Loaded checkpoint identity: **{identity.get('status', 'unavailable')}**",
        f"- Bundle/run ID: `{(identity.get('requested_manifest') or {}).get('bundle_id', (identity.get('requested_manifest') or {}).get('run_id', 'unavailable'))}`",
        f"- Turn policy: `{report.get('requested', {}).get('turn_policy')}`",
        f"- Scenario scope: `{(report.get('scenario_coverage') or {}).get('scope', 'full_canonical_six')}`; selected: {(report.get('scenario_coverage') or {}).get('selected', [])}",
        "",
        "Transport and server metrics describe message delivery and event timing. They do not establish conversational quality. Behavior uses live `duplex_control6.decision` events and captured outputs.",
        "",
        f"Input frames sent: {transport.get('input_frames_sent', 0)}; acknowledgements: {transport.get('input_ack_events', 0)}; server errors: {transport.get('server_errors', 0)}.",
        f"Responses created/completed/cancelled: {transport.get('responses_created', 0)}/{transport.get('responses_completed', 0)}/{transport.get('responses_cancelled', 0)}; mean frame-to-response-start: {transport.get('mean_frame_to_response_start_latency_ms')} ms; mean frame-to-first-audio: {transport.get('mean_frame_to_first_audio_latency_ms')} ms; output RTF: {transport.get('output_rtf')}.",
        f"Action coverage: {coverage.get('observed', [])}; missing: {coverage.get('missing', list(ACTION_ORDER))}.",
        f"Deferred listener decisions: {(behavior.get('action_lifecycle_coverage') or {}).get('deferred_decisions', {}).get('count_by_action', {})} by action; executed response actions: {(behavior.get('action_lifecycle_coverage') or {}).get('executed_responses', {}).get('count_by_action', {})}.",
        f"Live policy timing: CONTINUE ack matches {(behavior.get('live_policy_semantics') or {}).get('continue', {}).get('same_frame_listener_consumed_ack', 0)}; BACKCHANNELs overlapping later user frames {(behavior.get('live_policy_semantics') or {}).get('backchannel', {}).get('responses_with_user_audio_continuing_during_playout', 0)}; REPLAN cancel+endpoint resume {(behavior.get('live_policy_semantics') or {}).get('replan', {}).get('cancel_then_endpoint_scoped_resume', 0)}.",
        f"Raw action predictions (including rejected decisions): {(behavior.get('raw_action_coverage') or {}).get('observed', [])}.",
        f"Scenario coverage: {report.get('scenario_coverage', {}).get('status', 'incomplete')}; missing: {report.get('scenario_coverage', {}).get('missing', [])}.",
        f"Concurrent session isolation: {(behavior.get('session_isolation') or {}).get('status', 'unavailable')}.",
        f"Barge-in evidence: {(behavior.get('barge_in') or {}).get('status', 'unavailable')}.",
        f"Simulated playback: {(system.get('playback_simulation') or {}).get('status', 'unavailable')} (80 ms PCM consumer); physical device playback/latency measured: {(system.get('playback_simulation') or {}).get('physical_device_playback', False)}/{(system.get('playback_simulation') or {}).get('device_latency_measured', False)}.",
        f"ASR: {(behavior.get('asr') or {}).get('status', 'unavailable')} (backend: {(behavior.get('asr') or {}).get('backend') or 'none'}, device: {(behavior.get('asr') or {}).get('device') or 'none'}) — {(behavior.get('asr') or {}).get('reason') or 'no additional reason'}.",
        f"Runtime phase metrics: {(system.get('runtime_metrics') or {}).get('status', 'unavailable')} across {len((system.get('runtime_metrics') or {}).get('response_events', []))} response events; final scheduler counters are logged as `[realtime_session_metrics]` when the socket closes.",
        "",
        "WER/CER are null unless a local ASR model was explicitly configured and successfully transcribed both captured input and output audio. Cached scenario transcripts and model text are not substituted for ASR.",
        "",
        "The headless playout simulator consumes the actual received response PCM at 80 ms frames. `response.done` ends server delivery, while playback may continue through the queued tail; `response.cancelled` clears only that response's simulated queue. This does not measure a physical speaker or device latency.",
        "",
        "Artifacts: `trace.jsonl` contains raw WebSocket events, client-send/receipt timestamps, and monotonic simulated playout queue events; per-session WAVs contain received PCM, with per-response `_playout.wav` and `_cleared.wav` files for consumed and discarded audio; `report.json` preserves the done-versus-playback timeline and evidence gaps.",
    ]
    qualitative = report.get("qualitative_probes", {})
    if qualitative.get("status") != "not_requested":
        lines.extend([
            "",
            "## Qualitative follow-up probe",
            "",
            f"Status: **{qualitative.get('status', 'unavailable')}**. This section is for manual review only and does not affect the six-scenario/action evidence gates; it is not an exact-string or MOS score.",
            "",
            "Review whether the assistant identifies the Tuesday 16:30 conflict, clarifies the afternoon range, and follows the latest correction to Friday after 15:00.",
        ])
        for probe in qualitative.get("probes", []):
            lines.extend(["", f"### {probe.get('title', probe.get('scenario_id'))}", ""])
            for turn in probe.get("turns", []):
                lines.append(f"- User: {turn.get('input_reference_text', '')}")
                lines.append(f"  - Captured input WAV: `{turn.get('input_wav') or 'unavailable'}`")
                if turn.get("input_asr_text") is not None:
                    lines.append(f"  - Independent input ASR: {turn['input_asr_text']}")
                lines.append(f"  - Assistant response: {turn.get('response_text') or 'unavailable'}")
                if turn.get("response_asr_text") is not None:
                    lines.append(f"  - Independent response ASR: {turn['response_asr_text']}")
                asr_metrics = turn.get("asr_metrics", {})
                if asr_metrics.get("status") == "available":
                    lines.append(f"  - Input WER/CER: {asr_metrics.get('input_wer')}/{asr_metrics.get('input_cer')}; response WER/CER: {asr_metrics.get('output_wer')}/{asr_metrics.get('output_cer')}")
                lines.append(f"  - Captured response WAV: `{turn.get('response_wav') or 'unavailable'}`")
            if probe.get("session_errors"):
                lines.append(f"- Session errors: {probe['session_errors']}")
        if qualitative.get("missing"):
            lines.extend(["", f"Missing qualitative assets: {qualitative['missing']}."])
    if report.get("error"):
        lines.extend(["", f"Run error: `{report['error']}`"])
    lines.extend(["", "CPU-only harness tests validate report and transport bookkeeping only. Use the requested run ID, live `model_loaded` event, policy decisions, and captured audio in this report as evidence for the server run; the test fixtures do not establish GPU model quality.", ""])
    return "\n".join(lines)


def _write_reports(report: dict[str, Any], output_dir: Path) -> None:
    transport = (report.get("system", {}).get("transport") or {})
    identity = report.get("identity", {})
    behavior = report.get("behavior", {})
    coverage = behavior.get("action_coverage") or {}
    required_evidence = (
        identity.get("status") == "matched"
        and not coverage.get("missing", list(ACTION_ORDER))
        and report.get("scenario_coverage", {}).get("status") == "complete"
        and behavior.get("barge_in", {}).get("status") == "observed"
    )
    report["evaluation_status"] = "evidence_complete" if required_evidence and behavior.get("status") == "observed" else "incomplete"
    report.setdefault("artifacts", {})["report_json"] = str(output_dir / "report.json")
    report["artifacts"]["report_markdown"] = str(output_dir / "report.md")
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--server", help="WebSocket URL, e.g. ws://127.0.0.1:8000/v1/realtime?duplex=1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--turn-policy", default="duplex_control6")
    parser.add_argument("--expected-run-id", help="Run ID expected in both the local manifest and live server trace")
    parser.add_argument("--expected-bundle-id", help="Bundle ID expected in the local runtime manifest and live server trace")
    parser.add_argument("--scenario-audio-dir", type=Path, default=DEFAULT_SCENARIO_AUDIO_DIR)
    parser.add_argument(
        "--scenario-id", dest="scenario_ids", action="append",
        choices=[scenario.scenario_id for scenario in ALL_SCENARIOS],
        help="Run only this canonical scenario; repeat to select several. Reports remain scoped partial and cannot satisfy the six-scenario gate.",
    )
    parser.add_argument("--include-qualitative-prompts", action="store_true",
                        help="Run authored Italian follow-up probes; outputs are reported for human review and do not affect six-scenario coverage.")
    parser.add_argument("--asr-model-path", help="Optional local ASR model directory; no download is attempted")
    parser.add_argument("--asr-backend", default="auto", choices=("auto", "faster-whisper", "transformers"))
    parser.add_argument("--asr-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--input-audio-transcription", action="store_true",
        help="Opt in to the runtime's native partial/final input transcript events; requires server env DUPLEX_NATIVE_PARTIAL_ASR=1.",
    )
    parser.add_argument("--offline-asr-report", type=Path,
                        help="Enrich an existing report's captured WAVs with local ASR; run only after stopping the server")
    parser.add_argument("--frame-pacing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.offline_asr_report:
        if not args.asr_model_path:
            _parser().error("--offline-asr-report requires --asr-model-path")
        report_path = args.offline_asr_report.expanduser().resolve()
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            enrich_report_with_offline_asr(
                report, report_path, model_path=args.asr_model_path,
                device=args.asr_device, backend=args.asr_backend,
            )
            _write_reports(report, report_path.parent)
        except Exception as exc:
            print(json.dumps({"asr_enrichment": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
            return 2
        print(json.dumps({"asr_enrichment": report.get("asr_enrichment", {}).get("status"),
                          "asr": report.get("behavior", {}).get("asr", {}).get("status"),
                          "report": str(report_path)}, ensure_ascii=False))
        return 0 if report.get("asr_enrichment", {}).get("status") == "available" else 2
    if not args.server:
        _parser().error("live capture requires --server (or use --offline-asr-report)")
    if args.output_dir is None:
        _parser().error("live capture requires --output-dir")
    if not args.expected_run_id and not args.expected_bundle_id:
        _parser().error("live capture requires --expected-run-id or --expected-bundle-id")
    report = asyncio.run(_build_report(args))
    output_dir = Path(args.output_dir).expanduser().resolve()
    _write_reports(report, output_dir)
    print(json.dumps({"evaluation_status": report.get("evaluation_status"), "system": report.get("system", {}).get("status"),
                      "identity": report.get("identity", {}).get("status"), "behavior": report.get("behavior", {}).get("status"),
                      "report": str(output_dir / "report.json")}, ensure_ascii=False))
    return 0 if report.get("evaluation_status") == "evidence_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
