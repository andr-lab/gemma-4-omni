"""Reproducible CPU phase-coordinate sweeps for a native1540 Control6 adapter.

This is a diagnostic forward evaluator, not a certification gate. It evaluates
both raw action-row groupings and the runtime speaker-active HOLD/YIELD
mapping, while keeping one phase coordinate fixed at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from training.duplex_qlora.native_control6 import ACTIONS, load_native_control6_adapter


EPSILON = 1e-6
SPEECH_RMS_FLOOR = 1e-3
RATIO_ACTIVITY_THRESHOLD = -0.15
HIDDEN_WIDTH = 1536
PHASE_WIDTH = 4


def control6_phase0_grid(*, max_observed: float) -> np.ndarray:
    """RMS-log grid: dense low-RMS interval and the requested boundary probes."""
    if not math.isfinite(max_observed):
        raise ValueError("max_observed must be finite")
    if max_observed < -6.0 or max_observed > 0.0:
        raise ValueError("native1540 observed phase0 must be in [-6, 0]")
    values: list[float] = list(np.linspace(-6.0, -3.0, 301, dtype=np.float64))
    values.extend((-6.0 - 1e-6, -6.0 + 1e-6, -3.0 - 1e-6, -3.0 + 1e-6))
    if max_observed > -3.0:
        values.extend(np.linspace(-3.0, max_observed, 20, dtype=np.float64).tolist())
    return np.asarray(sorted(set(values)), dtype=np.float64)


def control6_phase3_grid() -> np.ndarray:
    """Native ratio clip grid with exact and epsilon probes at the activity edge."""
    values = list(np.linspace(-6.0, 6.0, 101, dtype=np.float64))
    values.extend((
        RATIO_ACTIVITY_THRESHOLD - 1e-6,
        RATIO_ACTIVITY_THRESHOLD,
        RATIO_ACTIVITY_THRESHOLD + 1e-6,
    ))
    return np.asarray(sorted(set(values)), dtype=np.float64)


def speaker_active_effective_delta(
    raw_probabilities: dict[str, float],
    *,
    user_rms: float,
    assistant_log_rms: float,
    user_assistant_log_ratio: float,
    ratio_valid: bool,
    peer_active: bool,
    speech_rms_floor: float = SPEECH_RMS_FLOOR,
) -> tuple[float, float, bool]:
    """Return runtime-normalized YIELD-minus-HOLD, weight, and activity.

    The mapping follows GemmaNativeControl6Policy for an active speaker:
    LISTEN is split between HOLD and YIELD by the continuous RMS/ratio weight;
    detected activity sets that weight to one. Illegal SPEAK/BACKCHANNEL mass is
    removed and the remaining physical HOLD/YIELD mass is normalized.
    """
    if not math.isfinite(user_rms) or user_rms < 0.0:
        raise ValueError("user_rms must be finite and non-negative")
    if not math.isfinite(assistant_log_rms) or not math.isfinite(user_assistant_log_ratio):
        raise ValueError("phase values must be finite")
    if not math.isfinite(speech_rms_floor) or speech_rms_floor <= 0.0:
        raise ValueError("speech_rms_floor must be finite and positive")
    if set(raw_probabilities) != set(ACTIONS):
        raise ValueError("raw_probabilities must provide exactly the six Control6 actions")

    assistant_rms = 10.0 ** float(np.clip(assistant_log_rms, -6.0, 0.0))
    ratio_weight = 0.0
    if ratio_valid and assistant_rms > EPSILON:
        ratio_floor = math.log10(EPSILON / assistant_rms)
        ratio_span = RATIO_ACTIVITY_THRESHOLD - ratio_floor
        if ratio_span > 0.0:
            ratio_weight = min(
                1.0,
                max(0.0, (user_assistant_log_ratio - ratio_floor) / ratio_span),
            )
    ratio_activity = bool(
        ratio_valid and user_assistant_log_ratio > RATIO_ACTIVITY_THRESHOLD
    )
    activity = bool(peer_active or user_rms >= speech_rms_floor or ratio_activity)
    weight = 1.0 if activity else min(
        1.0,
        max(0.0, max(user_rms / speech_rms_floor, ratio_weight)),
    )
    hold = raw_probabilities["CONTINUE"] + (1.0 - weight) * raw_probabilities["LISTEN"]
    yield_probability = (
        raw_probabilities["YIELD"]
        + raw_probabilities["REPLAN"]
        + weight * raw_probabilities["LISTEN"]
    )
    legal_mass = hold + yield_probability
    if legal_mass <= 0.0 or not math.isfinite(legal_mass):
        return 0.0, weight, activity
    return (yield_probability - hold) / legal_mass, weight, activity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_arrays(metadata: list[dict[str, Any]], features: np.ndarray) -> dict[str, np.ndarray]:
    n_rows = len(metadata)
    frame_ids = np.asarray([int(row.get("listener_frame_id", row.get("frame_id", -1))) for row in metadata])
    assistant_refs = np.asarray([
        row.get("assistant_reference_frame_id") == int(row.get("listener_frame_id", row.get("frame_id", -1))) - 1
        for row in metadata
    ], dtype=bool)
    context_valid = np.asarray([
        bool(row.get("native_control6_context_valid", False))
        and row.get("native_control6_phase_profile") == "native1540"
        for row in metadata
    ], dtype=bool)
    speaker_active = np.asarray([
        bool(row.get("speaker_active_at_frame_start", False)) for row in metadata
    ], dtype=bool)
    peer_active = np.asarray([
        bool(row.get("peer_active_during_frame", False)) for row in metadata
    ], dtype=bool)
    phase = features[:, HIDDEN_WIDTH:HIDDEN_WIDTH + PHASE_WIDTH]
    if len(frame_ids) != n_rows or phase.shape != (n_rows, PHASE_WIDTH):
        raise ValueError("validation metadata/features do not align to the native1540 contract")
    return {
        "frame_ids": frame_ids,
        "assistant_refs": assistant_refs,
        "context_valid": context_valid,
        "speaker_active": speaker_active,
        "peer_active": peer_active,
        "phase": phase,
    }


def _forward_grid(
    features: np.ndarray,
    adapter: torch.nn.Module,
    grid: np.ndarray,
    *,
    phase_index: int,
    paired_ratio: bool = False,
    assistant_log_rms: np.ndarray | None = None,
    batch_rows: int = 64,
) -> Iterable[tuple[int, np.ndarray]]:
    """Yield [rows, grid, actions] softmaxes using the checkpoint forward math."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(features), batch_rows):
            stop = min(len(features), start + batch_rows)
            batch = torch.as_tensor(features[start:stop], dtype=torch.float32)
            hidden = batch[:, :HIDDEN_WIDTH]
            phases = batch[:, HIDDEN_WIDTH:HIDDEN_WIDTH + PHASE_WIDTH]
            base = F.linear(hidden.to(adapter.base_rows.dtype), adapter.base_rows).float()
            delta_input = torch.cat((hidden, phases), dim=-1)
            rank_base = F.linear(delta_input, adapter.lora_A)
            if paired_ratio:
                if assistant_log_rms is None:
                    raise ValueError("paired_ratio requires assistant_log_rms")
                log_user = grid_t.clamp(-6.0, 0.0)
                paired = (log_user[None, :] - torch.as_tensor(
                    assistant_log_rms[start:stop], dtype=torch.float32
                )[:, None]).clamp(-6.0, 6.0)
                coord_values = paired
                original = phases[:, 3:4]
            else:
                coord_values = grid_t.clamp(-6.0, 0.0) if phase_index == 0 else grid_t.clamp(-6.0, 6.0)
                original = phases[:, phase_index:phase_index + 1]
                coord_values = coord_values[None, :].expand(stop - start, -1)
            phase_axis = adapter.lora_A[:, HIDDEN_WIDTH + phase_index]
            if paired_ratio:
                # The paired geometry changes both user RMS and ratio with the
                # source-native relation; phase0 remains the sweep coordinate.
                phase0_axis = adapter.lora_A[:, HIDDEN_WIDTH]
                phase3_axis = adapter.lora_A[:, HIDDEN_WIDTH + 3]
                delta0 = grid_t.clamp(-6.0, 0.0)[None, :] - phases[:, 0:1]
                delta3 = coord_values - phases[:, 3:4]
                rank = (
                    rank_base[:, None, :]
                    + delta0[:, :, None] * phase0_axis[None, None, :]
                    + delta3[:, :, None] * phase3_axis[None, None, :]
                )
            else:
                rank = rank_base[:, None, :] + (
                    coord_values - original
                )[:, :, None] * phase_axis[None, None, :]
            delta = F.linear(rank, adapter.lora_B) * adapter.scaling
            logits = base[:, None, :] + delta
            yield start, torch.softmax(logits, dim=-1).cpu().numpy()


def _curve_statistics(
    values: np.ndarray,
    *,
    row_indices: np.ndarray,
    coordinates: np.ndarray,
    metadata: list[dict[str, Any]],
    exclude_pair: int | None = None,
    weights: np.ndarray | None = None,
    activities: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize signed adjacent changes and retain an actionable minimum row."""
    if values.size == 0 or values.shape[1] < 2:
        return {"row_count": int(values.shape[0]), "minimum_delta_step": None, "negative_step_count": 0}
    delta_steps = np.diff(values, axis=1)
    eligible = np.ones(delta_steps.shape[1], dtype=bool)
    if exclude_pair is not None and 0 <= exclude_pair < len(eligible):
        eligible[exclude_pair] = False
    selected = delta_steps[:, eligible]
    if selected.size == 0:
        return {"row_count": int(values.shape[0]), "minimum_delta_step": None, "negative_step_count": 0}
    flat_min = int(np.argmin(selected))
    row_local, pair_local = np.unravel_index(flat_min, selected.shape)
    actual_pairs = np.flatnonzero(eligible)
    pair_index = int(actual_pairs[pair_local])
    row_index = int(row_indices[row_local])
    meta = metadata[row_index]
    example: dict[str, Any] = {
        "row_index": row_index,
        "frame_id": int(meta.get("listener_frame_id", meta.get("frame_id", -1))),
        "conversation_id": meta.get("conversation_id"),
        "coordinate_start": float(coordinates[pair_index]),
        "coordinate_end": float(coordinates[pair_index + 1]),
        "delta_start": float(values[row_local, pair_index]),
        "delta_end": float(values[row_local, pair_index + 1]),
        "delta_step": float(delta_steps[row_local, pair_index]),
    }
    if weights is not None:
        example["weight_start"] = float(weights[row_local, pair_index])
        example["weight_end"] = float(weights[row_local, pair_index + 1])
    if activities is not None:
        example["activity_start"] = bool(activities[row_local, pair_index])
        example["activity_end"] = bool(activities[row_local, pair_index + 1])
    if probabilities is not None:
        example["raw_probabilities_start"] = {
            name: float(probabilities[row_local, pair_index, i]) for i, name in enumerate(ACTIONS)
        }
        example["raw_probabilities_end"] = {
            name: float(probabilities[row_local, pair_index + 1, i]) for i, name in enumerate(ACTIONS)
        }
    return {
        "row_count": int(values.shape[0]),
        "pair_count": int(selected.size),
        "negative_step_count": int(np.count_nonzero(selected < 0.0)),
        "minimum_delta_step": float(selected.flat[flat_min]),
        "minimum_example": example,
        "excluding_boundary_pair": exclude_pair,
    }


def _runtime_curves(
    probs: np.ndarray,
    *,
    features: np.ndarray,
    row_indices: np.ndarray,
    metadata: list[dict[str, Any]],
    phase_coordinates: np.ndarray,
    phase_index: int,
    paired_ratio: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the current speaker-active physical mapping to each sweep point."""
    phase = features[row_indices, HIDDEN_WIDTH:HIDDEN_WIDTH + PHASE_WIDTH]
    assistant_logs = phase[:, 2]
    peer = np.asarray([bool(metadata[i].get("peer_active_during_frame", False)) for i in row_indices])
    speaker_active = np.asarray([bool(metadata[i].get("speaker_active_at_frame_start", False)) for i in row_indices])
    if not np.all(speaker_active):
        raise ValueError("runtime physical sweep expects speaker_active rows only")
    assistant_refs = np.asarray([
        metadata[i].get("assistant_reference_frame_id")
        == int(metadata[i].get("listener_frame_id", metadata[i].get("frame_id", -1))) - 1
        for i in row_indices
    ], dtype=bool)
    if phase_index == 0:
        phase0_actual = np.broadcast_to(phase_coordinates[None, :], (len(row_indices), len(phase_coordinates)))
        current_rms = np.power(10.0, phase0_actual)
        phase3 = np.broadcast_to(phase[:, 3:4], current_rms.shape)
        assistant_rms = np.power(10.0, np.clip(assistant_logs, -6.0, 0.0))[:, None]
        valid = assistant_refs[:, None] & (current_rms > EPSILON) & (assistant_rms >= SPEECH_RMS_FLOOR)
    else:
        phase0_actual = np.broadcast_to(phase[:, 0:1], (len(row_indices), len(phase_coordinates)))
        current_rms = np.power(10.0, np.clip(phase0_actual, -6.0, 0.0))
        phase3 = np.broadcast_to(phase_coordinates[None, :], current_rms.shape)
        assistant_rms = np.power(10.0, np.clip(assistant_logs, -6.0, 0.0))[:, None]
        valid = assistant_refs[:, None] & (current_rms > EPSILON) & (assistant_rms >= SPEECH_RMS_FLOOR)
    if paired_ratio:
        phase3 = np.clip(phase0_actual - assistant_logs[:, None], -6.0, 6.0)
        valid = assistant_refs[:, None] & (current_rms > EPSILON) & (assistant_rms >= SPEECH_RMS_FLOOR)

    floor_values = np.log10(EPSILON / np.maximum(assistant_rms, EPSILON))
    ratio_span = RATIO_ACTIVITY_THRESHOLD - floor_values
    ratio_weight = np.divide(
        phase3 - floor_values,
        ratio_span,
        out=np.zeros_like(phase3),
        where=valid & (ratio_span > 0.0),
    )
    ratio_weight = np.clip(ratio_weight, 0.0, 1.0)
    ratio_activity = valid & (phase3 > RATIO_ACTIVITY_THRESHOLD)
    activity = peer[:, None] | (current_rms >= SPEECH_RMS_FLOOR) | ratio_activity
    raw = probs.reshape(len(row_indices), len(phase_coordinates), len(ACTIONS))
    names = {name: index for index, name in enumerate(ACTIONS)}
    weight = np.where(
        activity,
        1.0,
        np.clip(np.maximum(current_rms / SPEECH_RMS_FLOOR, ratio_weight), 0.0, 1.0),
    )
    hold = raw[:, :, names["CONTINUE"]] + (1.0 - weight) * raw[:, :, names["LISTEN"]]
    yield_probability = (
        raw[:, :, names["YIELD"]]
        + raw[:, :, names["REPLAN"]]
        + weight * raw[:, :, names["LISTEN"]]
    )
    legal_mass = hold + yield_probability
    delta = np.divide(
        yield_probability - hold,
        legal_mass,
        out=np.zeros_like(hold),
        where=legal_mass > 0.0,
    )
    return delta, weight, activity


def evaluate(
    *,
    validation_npz: Path,
    validation_json: Path,
    checkpoint: Path,
    batch_rows: int = 64,
) -> dict[str, Any]:
    with np.load(validation_npz, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"])
    metadata = json.loads(validation_json.read_text(encoding="utf-8"))
    if len(metadata) != len(features) or features.ndim != 2 or features.shape[1] != 1540:
        raise ValueError("expected aligned native1540 validation metadata and [N,1540] features")
    arrays = _metadata_arrays(metadata, features)
    adapter, load_report = load_native_control6_adapter(checkpoint, device="cpu")
    if load_report.get("native_control6_phase_profile") != "native1540":
        raise ValueError("current evaluator requires a native1540 checkpoint")
    adapter.eval()
    phase0 = arrays["phase"][:, 0]
    phase3 = arrays["phase"][:, 3]
    grid0 = control6_phase0_grid(max_observed=float(phase0.max()))
    grid3 = control6_phase3_grid()
    context_rows = np.flatnonzero(arrays["context_valid"])
    runtime_rows0 = context_rows[arrays["speaker_active"][context_rows] & ~arrays["peer_active"][context_rows]]
    runtime_rows3 = runtime_rows0[
        arrays["assistant_refs"][runtime_rows0]
        & (np.power(10.0, np.clip(phase0[runtime_rows0], -6.0, 0.0)) > EPSILON)
        & (np.power(10.0, np.clip(arrays["phase"][runtime_rows0, 2], -6.0, 0.0)) >= SPEECH_RMS_FLOOR)
    ]

    all_valid = context_rows
    raw_summaries: dict[str, Any] = {}
    runtime_summaries: dict[str, Any] = {}
    for coordinate_name, grid, phase_index, runtime_rows, paired in (
        ("phase0_user_rms_log", grid0, 0, runtime_rows0, False),
        ("phase3_user_assistant_log_ratio", grid3, 3, runtime_rows3, False),
        ("causal_paired_phase0_and_ratio", grid0, 0, runtime_rows0, True),
    ):
        n_rows = len(all_valid)
        raw_values_1 = np.empty((n_rows, len(grid)), dtype=np.float32)
        raw_values_2 = np.empty((n_rows, len(grid)), dtype=np.float32)
        runtime_values = np.empty((len(runtime_rows), len(grid)), dtype=np.float32)
        runtime_weights = np.empty_like(runtime_values)
        runtime_activities = np.empty(runtime_values.shape, dtype=bool)
        runtime_probs = np.empty((len(runtime_rows), len(grid), len(ACTIONS)), dtype=np.float32)
        runtime_lookup = {int(row): i for i, row in enumerate(runtime_rows)}
        for start, chunk_probs in _forward_grid(
            features[all_valid],
            adapter,
            grid,
            phase_index=phase_index,
            paired_ratio=paired,
            assistant_log_rms=arrays["phase"][all_valid, 2] if paired else None,
            batch_rows=batch_rows,
        ):
            end = min(n_rows, start + chunk_probs.shape[0])
            action_i = {name: ACTIONS.index(name) for name in ACTIONS}
            p = chunk_probs
            raw_values_1[start:end] = (
                p[:, :, action_i["YIELD"]]
                + p[:, :, action_i["REPLAN"]]
                - p[:, :, action_i["SPEAK"]]
            )
            raw_values_2[start:end] = (
                p[:, :, action_i["LISTEN"]]
                + p[:, :, action_i["YIELD"]]
                + p[:, :, action_i["REPLAN"]]
                - p[:, :, action_i["SPEAK"]]
            )
            current_global = all_valid[start:end]
            selected_local = [
                (offset, runtime_lookup[int(row)])
                for offset, row in enumerate(current_global)
                if int(row) in runtime_lookup
            ]
            if selected_local:
                offsets = np.asarray([item[0] for item in selected_local], dtype=np.int64)
                out_rows = np.asarray([item[1] for item in selected_local], dtype=np.int64)
                selected_probs = chunk_probs[offsets]
                runtime_probs[out_rows] = selected_probs
                runtime_features = features[runtime_rows[out_rows]]
                runtime_meta = [metadata[int(row)] for row in runtime_rows[out_rows]]
                delta, weight, activity = _runtime_curves(
                    selected_probs.reshape(-1, len(ACTIONS)),
                    features=runtime_features,
                    row_indices=np.arange(len(out_rows)),
                    metadata=runtime_meta,
                    phase_coordinates=grid,
                    phase_index=phase_index,
                    paired_ratio=paired,
                )
                runtime_values[out_rows] = delta
                runtime_weights[out_rows] = weight
                runtime_activities[out_rows] = activity
        raw_summaries[coordinate_name] = {
            "grid_points": int(len(grid)),
            "grid_min": float(grid.min()),
            "grid_max": float(grid.max()),
            "phase3_threshold_epsilon_points": (
                [float(x) for x in grid if abs(x - RATIO_ACTIVITY_THRESHOLD) <= 1e-6 + 1e-12]
                if phase_index == 3 else None
            ),
            "yield_replan_minus_speak": _curve_statistics(
                raw_values_1, row_indices=all_valid, coordinates=grid, metadata=metadata
            ),
            "listen_yield_replan_minus_speak": _curve_statistics(
                raw_values_2, row_indices=all_valid, coordinates=grid, metadata=metadata
            ),
        }
        exclude_validity_pair = None
        if phase_index == 0 and not paired:
            exact_eps = int(np.flatnonzero(np.isclose(grid, -6.0, rtol=0.0, atol=1e-12))[0])
            if exact_eps < len(grid) - 1:
                exclude_validity_pair = exact_eps
        runtime_summaries[coordinate_name] = {
            "speaker_active_peer_inactive_rows": int(len(runtime_rows)),
            "all_coordinate_steps": _curve_statistics(
                runtime_values,
                row_indices=runtime_rows,
                coordinates=grid,
                metadata=metadata,
                weights=runtime_weights,
                activities=runtime_activities,
                probabilities=runtime_probs,
            ),
            "interior_excluding_user_rms_epsilon_validity_pair": (
                _curve_statistics(
                    runtime_values,
                    row_indices=runtime_rows,
                    coordinates=grid,
                    metadata=metadata,
                    exclude_pair=exclude_validity_pair,
                    weights=runtime_weights,
                    activities=runtime_activities,
                    probabilities=runtime_probs,
                ) if exclude_validity_pair is not None else None
            ),
        }

    return {
        "protocol": "reproducible_native1540_coordinate_sweeps_v1",
        "purpose": "diagnostic_forward_evaluator_not_a_historical_or_certification_gate",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_npz": str(validation_npz),
        "validation_npz_sha256": _sha256(validation_npz),
        "validation_json_sha256": _sha256(validation_json),
        "checkpoint_phase_profile": load_report.get("native_control6_phase_profile"),
        "feature_shape": list(features.shape),
        "label_count": int(labels.shape[0]),
        "context_valid_rows": int(np.count_nonzero(arrays["context_valid"])),
        "runtime_coordinate_eligibility": "speaker_active_at_frame_start and peer_active_during_frame=false; phase3 additionally requires exact assistant t-1 reference, user RMS > epsilon, and assistant RMS >= speech floor",
        "phase0_grid": {
            "base_points_minus6_to_minus3_inclusive": 301,
            "boundary_log10_rms_probes": [-6.000001, -6.0, -5.999999, -3.000001, -3.0, -2.999999],
            "extended_points_when_observed_above_minus3": 20,
        },
        "phase3_grid": {
            "source_clip": [-6.0, 6.0],
            "base_points": 101,
            "ratio_activity_threshold": RATIO_ACTIVITY_THRESHOLD,
            "threshold_probes": [RATIO_ACTIVITY_THRESHOLD - 1e-6, RATIO_ACTIVITY_THRESHOLD, RATIO_ACTIVITY_THRESHOLD + 1e-6],
        },
        "raw_sweeps": raw_summaries,
        "runtime_sweeps": runtime_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-npz", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=64)
    args = parser.parse_args()
    result = evaluate(
        validation_npz=args.validation_npz,
        validation_json=args.validation_json,
        checkpoint=args.checkpoint,
        batch_rows=args.batch_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "context_valid_rows": result["context_valid_rows"],
        "runtime_phase0": result["runtime_sweeps"]["phase0_user_rms_log"],
        "runtime_phase3": result["runtime_sweeps"]["phase3_user_assistant_log_ratio"],
        "runtime_paired": result["runtime_sweeps"]["causal_paired_phase0_and_ratio"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
