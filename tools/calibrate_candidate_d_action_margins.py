"""Validation-only coordinate search for per-action margins on frozen Candidate C.

This utility reads only a frozen 1540-wide Candidate C head and its full validation
features. It never opens the held-out test feature file and never modifies a bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.duplex_qlora.control import load_control_head
from training.duplex_qlora.schema import ACTIONS

BASELINE_ACTIONS = ("LISTEN", "CONTINUE")
ELIGIBLE_ACTIONS = ("SPEAK", "YIELD", "BACKCHANNEL", "REPLAN")
FALSE_EVENT_BUDGET = 0.10


def apply_action_margins(
    logits: np.ndarray,
    actions: Sequence[str],
    thresholds_by_action: Mapping[str, float],
) -> np.ndarray:
    """Apply runtime's raw-top-event margin rule; equality keeps the event."""
    values = np.asarray(logits)
    ordered_actions = tuple(actions)
    if values.ndim != 2 or values.shape[1] != len(ordered_actions):
        raise ValueError("logits must have shape [rows, len(actions)]")
    if ordered_actions != tuple(ACTIONS):
        raise ValueError(f"actions must match the frozen control-head order {tuple(ACTIONS)}")
    if set(thresholds_by_action) != set(ELIGIBLE_ACTIONS):
        raise ValueError(f"thresholds_by_action must contain exactly {ELIGIBLE_ACTIONS}")
    thresholds: dict[str, float] = {}
    for action in ELIGIBLE_ACTIONS:
        value = float(thresholds_by_action[action])
        if not math.isfinite(value):
            raise ValueError(f"threshold for {action} must be finite")
        thresholds[action] = value
    if not np.isfinite(values).all():
        raise ValueError("logits must all be finite")

    index = {action: i for i, action in enumerate(ordered_actions)}
    raw = np.argmax(values, axis=1)
    # The runtime baseline is the higher of LISTEN and CONTINUE; ties choose LISTEN.
    baseline = np.where(
        values[:, index["LISTEN"]] >= values[:, index["CONTINUE"]],
        index["LISTEN"],
        index["CONTINUE"],
    )
    margin = values[np.arange(len(values)), raw] - np.maximum(
        values[:, index["LISTEN"]], values[:, index["CONTINUE"]]
    )
    predicted = raw.copy()
    for action in ELIGIBLE_ACTIONS:
        class_index = index[action]
        suppress = (raw == class_index) & (margin < thresholds[action])
        predicted[suppress] = baseline[suppress]
    return predicted


def confusion_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    actions: Sequence[str],
) -> dict[str, object]:
    count = len(actions)
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    result = confusion_metrics_from_matrix(matrix, actions)
    result["examples"] = int(len(truth))
    result["accuracy"] = float(np.mean(truth == predicted)) if len(truth) else None
    return result


def confusion_metrics_from_matrix(matrix: np.ndarray, actions: Sequence[str]) -> dict[str, object]:
    count = len(actions)
    if matrix.shape != (count, count):
        raise ValueError("confusion matrix shape must match action count")
    per_class: dict[str, dict[str, float | int | None]] = {}
    present_f1: list[float] = []
    for i, action in enumerate(actions):
        support = int(matrix[i].sum())
        predicted_support = int(matrix[:, i].sum())
        tp = int(matrix[i, i])
        precision = tp / predicted_support if predicted_support else 0.0
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else (0.0 if recall is not None else None)
        if support:
            present_f1.append(float(f1))
        per_class[action] = {
            "support": support,
            "predicted_support": predicted_support,
            "true_positive": tp,
            "precision": float(precision) if support else None,
            "recall": float(recall) if recall is not None else None,
            "f1": float(f1) if f1 is not None else None,
        }
    return {
        "examples": int(matrix.sum()),
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else None,
        "macro_f1_present_classes": float(np.mean(present_f1)) if present_f1 else None,
        "class_support": {action: per_class[action]["support"] for action in actions},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def summarize_policy(
    logits: np.ndarray,
    truth: np.ndarray,
    actions: Sequence[str],
    thresholds_by_action: Mapping[str, float],
    v5_mask: np.ndarray,
    onset_mask: np.ndarray,
    background_mask: np.ndarray,
) -> dict[str, object]:
    predicted = apply_action_margins(logits, actions, thresholds_by_action)
    onset_metrics = confusion_metrics(truth[onset_mask], predicted[onset_mask], actions)
    v5_metrics = confusion_metrics(truth[v5_mask], predicted[v5_mask], actions)
    background_support = int(background_mask.sum())
    per_event: dict[str, dict[str, float | int | None]] = {}
    background_fp_count = 0
    event_recalls: list[float] = []
    for action in ELIGIBLE_ACTIONS:
        idx = tuple(actions).index(action)
        false_positive = int(np.sum(predicted[background_mask] == idx))
        background_fp_count += false_positive
        onset = onset_metrics["per_class"][action]
        per_event[action] = {
            "background_false_events": false_positive,
            "background_support": background_support,
            "background_false_event_rate": false_positive / background_support if background_support else None,
            "onset_support": onset["support"],
            "onset_true_positive": onset["true_positive"],
            "onset_recall": onset["recall"],
            "onset_precision": onset["precision"],
            "onset_f1": onset["f1"],
        }
        if onset["recall"] is not None:
            event_recalls.append(float(onset["recall"]))
    return {
        "thresholds_by_action": {action: float(thresholds_by_action[action]) for action in ELIGIBLE_ACTIONS},
        "event_onset_metrics": onset_metrics,
        "v5_full_six_class_metrics": v5_metrics,
        "background_false_event_count": background_fp_count,
        "background_support": background_support,
        "background_false_event_rate": background_fp_count / background_support if background_support else None,
        "per_event_action": per_event,
        "all_eligible_onset_recalls_nonzero": len(event_recalls) == len(ELIGIBLE_ACTIONS) and all(value > 0 for value in event_recalls),
        "false_event_budget_met": background_fp_count / background_support <= FALSE_EVENT_BUDGET if background_support else False,
    }


def _candidate_thresholds(
    logits: np.ndarray,
    actions: Sequence[str],
    eligible_v5_mask: np.ndarray,
    action: str,
    include_threshold: float,
) -> list[float]:
    action_index = tuple(actions).index(action)
    raw = np.argmax(logits, axis=1)
    index = {name: i for i, name in enumerate(actions)}
    margins = logits[:, action_index] - np.maximum(
        logits[:, index["LISTEN"]], logits[:, index["CONTINUE"]]
    )
    relevant = margins[eligible_v5_mask & (raw == action_index)]
    if len(relevant):
        values = [float(value) for value in np.unique(relevant)]
        values.append(float(np.nextafter(np.max(relevant), np.float32(np.inf))))
    else:
        values = []
    values.append(float(include_threshold))
    return sorted(set(values))


def _is_feasible(summary: Mapping[str, object]) -> bool:
    return bool(summary["false_event_budget_met"] and summary["all_eligible_onset_recalls_nonzero"])


def _rank(summary: Mapping[str, object]) -> tuple[float, int, float]:
    onset = summary["event_onset_metrics"]
    macro_f1 = float(onset["macro_f1_present_classes"] or 0.0)
    false_events = int(summary["background_false_event_count"])
    recalls = [summary["per_event_action"][action]["onset_recall"] for action in ELIGIBLE_ACTIONS]
    min_recall = min(float(value) for value in recalls if value is not None)
    # Primary: onset macro-F1. Tie-break with fewer background false events, then
    # keep the weakest event recall higher when the measured F1 is identical.
    return (macro_f1, -false_events, min_recall)


def coordinate_search(
    logits: np.ndarray,
    truth: np.ndarray,
    actions: Sequence[str],
    v5_mask: np.ndarray,
    onset_mask: np.ndarray,
    background_mask: np.ndarray,
    *,
    shared_threshold: float,
    max_passes: int = 12,
) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    thresholds = {action: float(shared_threshold) for action in ELIGIBLE_ACTIONS}
    current = summarize_policy(logits, truth, actions, thresholds, v5_mask, onset_mask, background_mask)
    if not _is_feasible(current):
        raise ValueError("shared-threshold starting policy must be feasible on validation")
    trace: list[dict[str, object]] = []
    for pass_index in range(max_passes):
        changed = False
        for action in ELIGIBLE_ACTIONS:
            selected_threshold = thresholds[action]
            selected_summary = current
            selected_rank = _rank(current)
            options = _candidate_thresholds(logits, actions, v5_mask, action, thresholds[action])
            for value in options:
                proposal = dict(thresholds)
                proposal[action] = value
                candidate = summarize_policy(logits, truth, actions, proposal, v5_mask, onset_mask, background_mask)
                if not _is_feasible(candidate):
                    continue
                rank = _rank(candidate)
                if rank > selected_rank:
                    selected_threshold, selected_summary, selected_rank = value, candidate, rank
            if selected_threshold != thresholds[action]:
                thresholds[action] = selected_threshold
                current = selected_summary
                changed = True
        trace.append({
            "pass": pass_index + 1,
            "thresholds_by_action": dict(thresholds),
            "event_onset_macro_f1": current["event_onset_metrics"]["macro_f1_present_classes"],
            "background_false_event_rate": current["background_false_event_rate"],
        })
        if not changed:
            break
    return thresholds, current, trace


def exhaustive_onset_threshold_search(
    logits: np.ndarray,
    truth: np.ndarray,
    actions: Sequence[str],
    onset_mask: np.ndarray,
    background_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, object], dict[str, object]]:
    """Search every FAR-relevant policy induced by observed validation onsets.

    For a fixed set of predicted onset rows, the strictest threshold that preserves
    the set equals its minimum retained onset margin. That setting minimizes or
    preserves background false events, so it dominates looser thresholds with the
    same onset predictions. Enumerating each action's onset-margin breakpoints and
    their combinations is therefore exact for the stated macro-F1/FAR objective.
    """
    ordered_actions = tuple(actions)
    indices = {action: i for i, action in enumerate(ordered_actions)}
    raw = np.argmax(logits, axis=1)
    baseline = np.where(
        logits[:, indices["LISTEN"]] >= logits[:, indices["CONTINUE"]],
        indices["LISTEN"],
        indices["CONTINUE"],
    )
    margins = logits[np.arange(len(logits)), raw] - np.maximum(
        logits[:, indices["LISTEN"]], logits[:, indices["CONTINUE"]]
    )
    matrix_shape = (len(ordered_actions), len(ordered_actions))
    fixed_rows = onset_mask & np.isin(raw, [indices[action] for action in BASELINE_ACTIONS])
    fixed_confusion = np.zeros(matrix_shape, dtype=np.int64)
    np.add.at(fixed_confusion, (truth[fixed_rows], raw[fixed_rows]), 1)

    options_by_action: list[list[dict[str, object]]] = []
    breakpoint_counts: dict[str, int] = {}
    for action in ELIGIBLE_ACTIONS:
        action_index = indices[action]
        onset_group = onset_mask & (raw == action_index)
        scores = np.unique(margins[onset_group])
        if not len(scores):
            raise ValueError(f"no raw-top onset margin exists for {action}; nonzero recall is impossible")
        candidate_thresholds = [float(value) for value in scores]
        candidate_thresholds.append(float(np.nextafter(np.max(scores), np.float32(np.inf))))
        background_group = background_mask & (raw == action_index)
        action_options: list[dict[str, object]] = []
        seen: dict[tuple[bytes, int], int] = {}
        for threshold in candidate_thresholds:
            group_predicted = np.where(margins[onset_group] >= threshold, action_index, baseline[onset_group])
            group_confusion = np.zeros(matrix_shape, dtype=np.int64)
            np.add.at(group_confusion, (truth[onset_group], group_predicted), 1)
            background_false_events = int(np.sum(margins[background_group] >= threshold))
            signature = (group_confusion.tobytes(), background_false_events)
            if signature in seen:
                position = seen[signature]
                action_options[position]["threshold"] = threshold
                continue
            seen[signature] = len(action_options)
            action_options.append({
                "threshold": threshold,
                "onset_confusion": group_confusion,
                "background_false_events": background_false_events,
            })
        options_by_action.append(action_options)
        breakpoint_counts[action] = len(action_options)

    best_rank: tuple[float, int, float] | None = None
    best_thresholds: dict[str, float] | None = None
    best_matrix: np.ndarray | None = None
    best_false_events = 0
    combinations_examined = 0
    tradeoff_requirements = {
        "yield_recall_at_least_0_25": {"YIELD": 0.25},
        "backchannel_recall_at_least_0_375": {"BACKCHANNEL": 0.375},
        "yield_0_25_and_backchannel_0_375": {"YIELD": 0.25, "BACKCHANNEL": 0.375},
    }
    best_tradeoff_rank: dict[str, tuple[float, int, float] | None] = {name: None for name in tradeoff_requirements}
    best_tradeoff_thresholds: dict[str, dict[str, float] | None] = {name: None for name in tradeoff_requirements}
    best_min_far_rank: dict[str, tuple[int, float, float] | None] = {name: None for name in tradeoff_requirements}
    best_min_far_thresholds: dict[str, dict[str, float] | None] = {name: None for name in tradeoff_requirements}
    background_support = int(background_mask.sum())
    for combination in itertools.product(*options_by_action):
        combinations_examined += 1
        false_events = sum(int(option["background_false_events"]) for option in combination)
        matrix = fixed_confusion.copy()
        thresholds: dict[str, float] = {}
        for action, option in zip(ELIGIBLE_ACTIONS, combination):
            matrix += option["onset_confusion"]
            thresholds[action] = float(option["threshold"])
        metrics = confusion_metrics_from_matrix(matrix, ordered_actions)
        recalls = [metrics["per_class"][action]["recall"] for action in ELIGIBLE_ACTIONS]
        if any(recall is None or float(recall) <= 0.0 for recall in recalls):
            continue
        rank = (
            float(metrics["macro_f1_present_classes"] or 0.0),
            -false_events,
            min(float(recall) for recall in recalls if recall is not None),
        )
        for name, requirements in tradeoff_requirements.items():
            if any(float(metrics["per_class"][action]["recall"] or 0.0) < required for action, required in requirements.items()):
                continue
            if false_events / background_support <= FALSE_EVENT_BUDGET:
                current_rank = best_tradeoff_rank[name]
                if current_rank is None or rank > current_rank:
                    best_tradeoff_rank[name] = rank
                    best_tradeoff_thresholds[name] = thresholds
            min_far_rank = (-false_events, rank[0], rank[2])
            current_min_far_rank = best_min_far_rank[name]
            if current_min_far_rank is None or min_far_rank > current_min_far_rank:
                best_min_far_rank[name] = min_far_rank
                best_min_far_thresholds[name] = thresholds
        if false_events / background_support > FALSE_EVENT_BUDGET:
            continue
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_thresholds = thresholds
            best_matrix = matrix
            best_false_events = false_events
    if best_thresholds is None or best_matrix is None:
        raise ValueError("no per-action threshold combination met FAR and nonzero-recall validation constraints")
    return best_thresholds, {
        "combinations_examined": combinations_examined,
        "per_action_breakpoint_options": breakpoint_counts,
        "event_onset_metrics": confusion_metrics_from_matrix(best_matrix, ordered_actions),
        "background_false_event_count": best_false_events,
        "background_support": background_support,
        "background_false_event_rate": best_false_events / background_support,
        "tradeoff_thresholds": best_tradeoff_thresholds,
        "minimum_far_tradeoff_thresholds": best_min_far_thresholds,
        "minimum_far_tradeoff_results": {
            name: {
                "background_false_event_count": -rank[0],
                "background_support": background_support,
                "background_false_event_rate": -rank[0] / background_support,
                "event_onset_macro_f1": rank[1],
                "thresholds_by_action": best_min_far_thresholds[name],
            } if rank is not None else None
            for name, rank in best_min_far_rank.items()
        },
    }, {
        "global_optimum_within_scope": True,
        "scope": "all distinct validation onset decisions induced by the observed raw-top event margins; each decision pattern uses the strictest threshold that preserves its lowest retained onset score",
        "objective": "maximum onset macro-F1 subject to global V5 background FAR <= 0.10 and nonzero recall for every eligible action",
        "tie_break": "fewer background false events, then higher minimum eligible onset recall",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    *,
    feature_dir: Path,
    head_path: Path,
    output_path: Path,
    shared_threshold: float,
    max_passes: int,
) -> dict[str, object]:
    validation_npz = feature_dir / "validation.npz"
    validation_json = feature_dir / "validation.json"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    for path in (validation_npz, validation_json, feature_manifest_path, head_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    head = load_control_head(str(head_path), device="cpu")
    if head.input_dim != 1540:
        raise ValueError(f"Candidate C phase head must consume 1540 features, found {head.input_dim}")
    with np.load(validation_npz, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        label_ids = np.asarray(data["labels"], dtype=np.int64)
    rows = json.loads(validation_json.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    actions = tuple(ACTIONS)
    if features.shape != (len(rows), head.input_dim):
        raise ValueError(f"validation feature/metadata shape mismatch: {features.shape} vs {len(rows)} rows")
    if label_ids.shape != (len(rows),):
        raise ValueError("validation labels and metadata row counts differ")
    truth_from_metadata = np.asarray([actions.index(row["action"]) for row in rows], dtype=np.int64)
    if not np.array_equal(label_ids, truth_from_metadata):
        raise ValueError("validation NPZ labels differ from source metadata action labels")

    with torch.inference_mode():
        logits = head(torch.from_numpy(features)).cpu().numpy().astype(np.float32, copy=False)
    v5_mask = np.asarray([row.get("profile") == "audio_grounded_v5" for row in rows], dtype=bool)
    onset_mask = v5_mask & np.asarray([bool(row.get("event_onset")) for row in rows], dtype=bool)
    background_mask = v5_mask & ~onset_mask & np.isin(
        truth_from_metadata,
        [actions.index(action) for action in BASELINE_ACTIONS],
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source"))
        source_counts[source] = source_counts.get(source, 0) + 1
    if int(v5_mask.sum()) != 4087 or int(onset_mask.sum()) != 48 or int(background_mask.sum()) != 3956:
        raise ValueError(
            "validation data differs from the frozen Candidate C calibration cohort: "
            f"v5={int(v5_mask.sum())}, onsets={int(onset_mask.sum())}, background={int(background_mask.sum())}"
        )

    shared = {action: float(shared_threshold) for action in ELIGIBLE_ACTIONS}
    shared_summary = summarize_policy(logits, truth_from_metadata, actions, shared, v5_mask, onset_mask, background_mask)
    raw_thresholds = {action: -1_000_000.0 for action in ELIGIBLE_ACTIONS}
    raw_argmax_summary = summarize_policy(logits, truth_from_metadata, actions, raw_thresholds, v5_mask, onset_mask, background_mask)
    coordinate_thresholds, coordinate_summary, trace = coordinate_search(
        logits,
        truth_from_metadata,
        actions,
        v5_mask,
        onset_mask,
        background_mask,
        shared_threshold=shared_threshold,
        max_passes=max_passes,
    )
    thresholds, exhaustive_summary, exhaustive_scope = exhaustive_onset_threshold_search(
        logits,
        truth_from_metadata,
        actions,
        onset_mask,
        background_mask,
    )
    selected_summary = summarize_policy(logits, truth_from_metadata, actions, thresholds, v5_mask, onset_mask, background_mask)
    for action in ELIGIBLE_ACTIONS:
        selected_summary["per_event_action"][action]["maximum_recall_under_margin_policy"] = raw_argmax_summary["per_event_action"][action]["onset_recall"]
    tradeoff_summaries = {
        name: (
            summarize_policy(logits, truth_from_metadata, actions, proposal, v5_mask, onset_mask, background_mask)
            if proposal is not None else None
        )
        for name, proposal in exhaustive_summary["tradeoff_thresholds"].items()
    }
    minimum_far_tradeoff_summaries = {
        name: (
            summarize_policy(logits, truth_from_metadata, actions, proposal, v5_mask, onset_mask, background_mask)
            if proposal is not None else None
        )
        for name, proposal in exhaustive_summary["minimum_far_tradeoff_thresholds"].items()
    }
    return {
        "schema_version": 1,
        "candidate_id": "candidate_d_action_margins_validation_proposal",
        "status": "proposal_only_not_packaged_or_served",
        "calibration_kind": "per_action_non_idle_logit_margin",
        "threshold_semantics": (
            "Only the raw top eligible event is checked. Keep it when its logit minus max(LISTEN, CONTINUE) "
            "is >= its action threshold; otherwise remap to the higher-logit baseline. Equality keeps the event."
        ),
        "threshold_source": "full_v5_validation_timeline_only",
        "validation_rows": len(rows),
        "validation_v5_rows": int(v5_mask.sum()),
        "validation_v5_onset_rows": int(onset_mask.sum()),
        "validation_v5_background_rows": int(background_mask.sum()),
        "validation_action_support": {action: int(np.sum(truth_from_metadata[v5_mask] == i)) for i, action in enumerate(actions)},
        "validation_source_counts": source_counts,
        "false_event_budget": FALSE_EVENT_BUDGET,
        "meaningful_onset_recall_required_for_each": list(ELIGIBLE_ACTIONS),
        "test_data_read": False,
        "head_path": str(head_path),
        "head_sha256": _sha256(head_path),
        "validation_npz_sha256": _sha256(validation_npz),
        "feature_manifest_identity": feature_manifest.get("identity", feature_manifest.get("source_identity")),
        "search": {
            "algorithm": "exhaustive_combinations_of_validation_onset_margin_policies",
            "initialization": "none; finite search over onset decision patterns",
            "maximum_passes": max_passes,
            "coordinate_descent_diagnostic": {
                "thresholds_by_action": coordinate_thresholds,
                "summary": coordinate_summary,
                "passes": trace,
            },
            "exhaustive_search": exhaustive_summary,
            "exhaustive_scope": exhaustive_scope,
            "tradeoff_proposals": tradeoff_summaries,
            "minimum_far_tradeoff_proposals": minimum_far_tradeoff_summaries,
            "global_optimum_within_scope": True,
            "selection_objective": "validation event-onset macro-F1 over classes with positive onset support",
            "constraints": ["global V5 true-background false-event rate <= 0.10", "nonzero onset recall for every eligible event action"],
            "tie_break": "fewer true-background false events, then higher minimum eligible onset recall",
        },
        "candidate_c_shared_threshold_reference": shared_summary,
        "raw_argmax_reference": raw_argmax_summary,
        "candidate_d_action_threshold_proposal": selected_summary,
        "candidate_d_feasible": _is_feasible(selected_summary),
        "limitations": [
            "Only 48 V5 validation onset rows support the onset calibration; YIELD, BACKCHANNEL, and REPLAN each have support 8.",
            "The exhaustive search is global only over policies that differ on these 48 validation onset rows; thresholds are then fixed to the strictest value preserving each onset pattern.",
            "This is a validation calibration proposal only; no held-out test data was read and no bundle was modified.",
        ],
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("/workspace/checkpoints/gemma4_duplex_control6_v5_fulltimeline_repair200_20261003/features_candidate_c_phase"),
    )
    parser.add_argument(
        "--head-path",
        type=Path,
        default=Path("/workspace/checkpoints/gemma4_duplex_control6_v5_fulltimeline_repair200_20261003/head_candidate_c_phase/control_head_validation_frozen.pt"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/workspace/checkpoints/gemma4_duplex_control6_v5_fulltimeline_repair200_20261003/head_candidate_c_phase/candidate_d_action_margin_validation_proposal.json"),
    )
    parser.add_argument("--shared-threshold", type=float, default=0.9217718243598939)
    parser.add_argument("--max-passes", type=int, default=12)
    args = parser.parse_args()
    report = build_report(
        feature_dir=args.feature_dir,
        head_path=args.head_path,
        output_path=args.output_path,
        shared_threshold=args.shared_threshold,
        max_passes=args.max_passes,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output_path)
    selected = report["candidate_d_action_threshold_proposal"]
    print(json.dumps({
        "output": str(args.output_path),
        "thresholds_by_action": selected["thresholds_by_action"],
        "onset_macro_f1": selected["event_onset_metrics"]["macro_f1_present_classes"],
        "background_false_event_rate": selected["background_false_event_rate"],
        "per_event_action": selected["per_event_action"],
        "feasible": report["candidate_d_feasible"],
    }, indent=2))


if __name__ == "__main__":
    main()
