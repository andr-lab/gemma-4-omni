#!/usr/bin/env python3
"""Sequential compatibility capture using the installed E2E session helper.

The 3090 copy of eval_duplex_qlora_e2e.py may still pair scenarios 05 and 06
with asyncio.gather. This entry point calls its existing _run_session helper
one scenario at a time and writes the same report shape used by offline ASR.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_duplex_qlora_e2e as e2e


def _empty_result(scenario: Any, label: str, exc: Exception) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": label,
        "session_id": scenario.scenario_id,
        "errors": [f"{type(exc).__name__}: {exc}"],
        "inputs": [],
        "input_references": [],
        "responses": [],
        "text_by_response": {},
        "output_parts": [],
        "output_parts_by_response": {},
    }


def _pcm(parts: Any) -> np.ndarray:
    if isinstance(parts, np.ndarray):
        return np.asarray(parts, dtype=np.float32).reshape(-1)
    arrays = [np.asarray(part, dtype=np.float32).reshape(-1) for part in (parts or [])]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)


async def _capture_session(
    scenario: Any,
    assets: dict[str, Path],
    *,
    label: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    output_dir: Path,
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return await e2e._run_session(
            scenario=scenario,
            assets=assets,
            label=label,
            ws_url=args.server,
            policy=args.turn_policy,
            manifest=manifest,
            output_dir=output_dir,
            trace_rows=trace_rows,
            pace=getattr(args, "frame_pacing", True),
            special_barge=scenario.scenario_id == "05_barge_in_interruption",
            input_audio_transcription=getattr(args, "input_audio_transcription", False),
        )
    except Exception as exc:
        result = _empty_result(scenario, label, exc)
        requested = bool(getattr(args, "input_audio_transcription", False))
        result["input_audio_transcription"] = {
            "status": "not_enabled" if requested else "not_requested",
            "requested": requested,
            "enabled": False,
            "partial_events": 0,
            "completed_events": 0,
            "utterances": 0,
            "revision_monotonic": None,
            "first_partial_from_utterance_onset_ms": [],
            "first_partial_delivery_lag_ms": [],
        }
        return result


def _save_canonical_result(
    result: dict[str, Any],
    scenario: Any,
    output_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.get("label", scenario.scenario_id)))
    session_id = str(result.get("session_id") or scenario.scenario_id)
    input_pcm = _pcm(result.get("inputs"))
    output_pcm = _pcm(result.get("output_pcm") if result.get("output_pcm") is not None else result.get("output_parts"))
    output_rate = int(result.get("output_rate") or e2e.OUTPUT_RATE)
    input_path = output_dir / f"{label}_input.wav"
    output_path = output_dir / f"{label}_output.wav"
    if input_pcm.size:
        e2e._write_wav(input_path, input_pcm, e2e.INPUT_RATE)
    if output_pcm.size:
        e2e._write_wav(output_path, output_pcm, output_rate)

    playout = e2e._save_playout_simulation_artifacts(result, output_dir, label)
    report["artifacts"]["session_audio"][session_id] = {
        "input_wav": str(input_path) if input_pcm.size else None,
        "output_wav": str(output_path) if output_pcm.size else None,
    }
    report["artifacts"]["session_playout"][session_id] = playout
    response_ids = result.get("responses", [])
    text_by_response = result.get("text_by_response", {})
    response_text = " ".join(text_by_response.get(item, "") for item in response_ids).strip()
    turn: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "session_id": session_id,
        "input_audio": e2e.audio_quality(input_pcm, e2e.INPUT_RATE),
        "output_audio": e2e.audio_quality(
            output_pcm, output_rate, result.get("output_packet_boundaries", [])
        ),
        "playout_simulation": playout,
        "reference_text": " ".join(result.get("input_references", [])),
        "response_text": response_text,
        "asr_metrics": e2e.compare_asr_metrics(
            " ".join(result.get("input_references", [])), "", response_text, "",
            configured=False,
        ),
        "session_errors": result.get("errors", []),
    }
    assistant_steps = [step for step in scenario.steps if step.speaker == "assistant"]
    if assistant_steps:
        pairs = (
            [(assistant_steps[-1], response_ids[-1:])]
            if scenario.scenario_id == "05_barge_in_interruption"
            else [(step, response_ids[index:index + 1]) for index, step in enumerate(assistant_steps)]
        )
        checks = []
        for step, ids in pairs:
            step_text = " ".join(text_by_response.get(item, "") for item in ids).strip()
            checks.append({
                "step_id": step.step_id,
                "response_ids": ids,
                "status": "observed" if step_text else "unavailable",
                "checks": [bool(check(step_text)) for check in step.deterministic_checks],
            })
        turn["text_behavior_checks"] = {
            "status": "observed" if checks and all(row["status"] == "observed" for row in checks) else "incomplete",
            "steps": checks,
        }
    return turn


def _save_qualitative_result(
    result: dict[str, Any],
    scenario: Any,
    output_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.get("label", scenario.scenario_id)))
    response_ids = result.get("responses", [])
    text_by_response = result.get("text_by_response", {})
    output_rate = int(result.get("output_rate") or e2e.OUTPUT_RATE)
    turns = []
    user_steps = [step for step in scenario.steps if step.speaker == "user"]
    for index, step in enumerate(user_steps):
        input_pcm = (
            np.asarray(result["inputs"][index], dtype=np.float32).reshape(-1)
            if index < len(result.get("inputs", [])) else np.empty(0, dtype=np.float32)
        )
        response_id = response_ids[index] if index < len(response_ids) else None
        response_text = text_by_response.get(response_id, "") if response_id else ""
        output_pcm = _pcm(result.get("output_parts_by_response", {}).get(response_id, [])) if response_id else np.empty(0, dtype=np.float32)
        input_path = output_dir / f"{label}__turn_{index + 1:02d}_input.wav"
        output_path = output_dir / f"{label}__turn_{index + 1:02d}_output.wav"
        if input_pcm.size:
            e2e._write_wav(input_path, input_pcm, e2e.INPUT_RATE)
        if output_pcm.size:
            e2e._write_wav(output_path, output_pcm, output_rate)
        turns.append({
            "user_step_id": step.step_id,
            "input_reference_text": step.expected_text,
            "input_asr_text": None,
            "response_id": response_id,
            "response_text": response_text,
            "response_asr_text": None,
            "asr_metrics": e2e.compare_asr_metrics(
                step.expected_text, "", response_text, "", configured=False,
            ),
            "input_wav": str(input_path) if input_pcm.size else None,
            "response_wav": str(output_path) if output_pcm.size else None,
        })
    playout = e2e._save_playout_simulation_artifacts(result, output_dir, label)
    report["artifacts"]["session_playout"][str(result.get("session_id") or scenario.scenario_id)] = playout
    expected = len(user_steps)
    captured = bool(
        expected
        and not result.get("errors")
        and len(response_ids) >= expected
        and all(turn["response_text"].strip() and turn["response_wav"] for turn in turns)
    )
    return {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "session_id": result.get("session_id"),
        "assessment": "manual_review_only",
        "capture_status": "captured" if captured else "incomplete",
        "turns": turns,
        "session_errors": result.get("errors", []),
    }


async def _capture(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = e2e.load_checkpoint_manifest(checkpoint_root)
    if manifest.get("_response_bundle"):
        expected_bundle_id = getattr(args, "expected_bundle_id", None)
        if expected_bundle_id and manifest["bundle_id"] != expected_bundle_id:
            raise ValueError(
                f"requested bundle_id {expected_bundle_id!r} does not match manifest {manifest['bundle_id']!r}"
            )
    elif args.expected_run_id and manifest["run_id"] != args.expected_run_id:
        raise ValueError(
            f"requested run_id {args.expected_run_id!r} does not match manifest {manifest['run_id']!r}"
        )

    canonical, qualitative = e2e._evaluation_scenarios(
        bool(getattr(args, "include_qualitative_prompts", False))
    )
    canonical = e2e._select_canonical_scenarios(
        canonical, getattr(args, "scenario_ids", None),
    )
    all_scenarios = canonical + qualitative
    audio_dir = Path(args.scenario_audio_dir).expanduser()
    available, skipped = e2e._scenario_assets_for(audio_dir, all_scenarios)
    assets_by_id = {scenario.scenario_id: (scenario, assets) for scenario, assets in available}
    trace_rows: list[dict[str, Any]] = []
    qualitative_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested": {
            "checkpoint_root": str(checkpoint_root),
            "server": args.server,
            "turn_policy": args.turn_policy,
            "expected_run_id": args.expected_run_id,
            "expected_bundle_id": getattr(args, "expected_bundle_id", None),
            "scenario_ids": list(getattr(args, "scenario_ids", None) or []),
        },
        "identity": {"status": "unavailable", "requested_manifest": None, "observed_model_loaded": None},
        "system": {
            "status": "not_run",
            "transport": None,
            "playback_simulation": {
                "status": "unavailable", "endpoint": "simulated_headless_pcm_playout",
                "sample_rate_hz": e2e.OUTPUT_RATE, "frame_ms": e2e.FRAME_MS,
                "physical_device_playback": False, "device_latency_measured": False, "responses": [],
            },
        },
        "behavior": {
            "status": "unavailable", "action_coverage": None, "raw_action_coverage": None,
            "session_isolation": {
                "status": "not_tested",
                "reason": "the compatibility capture opens one scenario session at a time",
            },
            "barge_in": {"status": "unavailable", "evidence": None},
            "turns": [], "asr": {"status": "unavailable", "reason": "run offline ASR after stopping the server"},
        },
        "scenario_coverage": {
            "expected": [scenario.scenario_id for scenario in canonical],
            "run": [], "missing": [scenario.scenario_id for scenario in canonical],
            "skipped": [entry for entry in skipped if entry["scenario_id"] in {s.scenario_id for s in canonical}],
            "status": "incomplete", "execution_mode": "sequential_single_session_compat_wrapper",
            "concurrent_sessions": [],
        },
        "qualitative_probes": {
            "status": "requested" if qualitative else "not_requested",
            "assessment": "manual_review_only",
            "run": [], "missing": [],
            "skipped": [entry for entry in skipped if entry["scenario_id"] in {s.scenario_id for s in qualitative}],
            "probes": [],
        },
        "artifacts": {
            "trace_jsonl": str(output_dir / "trace.jsonl"),
            "session_audio": {}, "session_playout": {},
        },
    }
    try:
        if manifest.get("_response_bundle"):
            report["identity"]["requested_manifest"] = {
                "path": str(manifest["_manifest_path"]), "bundle_id": manifest["bundle_id"],
                "base_model_id": manifest["base_model_id"], "feature_source": manifest["feature_source"],
                "runtime_abi": manifest["runtime_abi"],
                "adapter_path": str(manifest["adapter_path"]),
                "control_head_path": str(manifest["control_head_path_resolved"]),
                "feature_contract": {
                    key: manifest["control_head"][key]
                    for key in ("input_dim", "pooling", "tick_ms", "actions")
                },
            }
        else:
            report["identity"]["requested_manifest"] = {
                "path": str(manifest["_manifest_path"]), "run_id": manifest["run_id"],
                "base_model_id": manifest["base_model_id"], "base_model_path": manifest.get("base_model_path"),
                "adapter_path": str(manifest["adapter_path"]),
                "control_head_path": str(manifest["control_head_path_resolved"]),
                "feature_contract": {
                    key: manifest["control_head"][key]
                    for key in ("input_dim", "pooling", "tick_ms", "actions")
                },
            }
        canonical_results = []
        for scenario in canonical:
            if scenario.scenario_id not in assets_by_id:
                continue
            scenario, assets = assets_by_id[scenario.scenario_id]
            label = "barge_primary" if scenario.scenario_id == "05_barge_in_interruption" else scenario.scenario_id
            result = await _capture_session(
                scenario, assets, label=label, args=args, manifest=manifest,
                output_dir=output_dir, trace_rows=trace_rows,
            )
            canonical_results.append(result)
            report["behavior"]["turns"].append(_save_canonical_result(result, scenario, output_dir, report))

        qualitative_results = []
        for scenario in qualitative:
            if scenario.scenario_id not in assets_by_id:
                continue
            scenario, assets = assets_by_id[scenario.scenario_id]
            label = scenario.scenario_id
            result = await _capture_session(
                scenario, assets, label=label, args=args, manifest=manifest,
                output_dir=output_dir, trace_rows=qualitative_rows,
            )
            qualitative_results.append(result)
            report["qualitative_probes"]["probes"].append(
                _save_qualitative_result(result, scenario, output_dir, report)
            )

        report["scenario_coverage"]["run"] = [result["scenario_id"] for result in canonical_results]
        report["scenario_coverage"]["missing"] = [
            scenario_id for scenario_id in report["scenario_coverage"]["expected"]
            if scenario_id not in report["scenario_coverage"]["run"]
        ]
        report["scenario_coverage"]["status"] = (
            "complete" if not report["scenario_coverage"]["missing"] else "incomplete"
        )
        if qualitative:
            report["artifacts"]["qualitative_trace_jsonl"] = str(output_dir / "qualitative_trace.jsonl")
            report["qualitative_probes"]["run"] = [result["scenario_id"] for result in qualitative_results]
            report["qualitative_probes"]["missing"] = [
                scenario_id for scenario_id in (s.scenario_id for s in qualitative)
                if scenario_id not in report["qualitative_probes"]["run"]
            ]
            report["qualitative_probes"]["status"] = (
                "captured" if qualitative_results and all(
                    probe["capture_status"] == "captured"
                    for probe in report["qualitative_probes"]["probes"]
                ) and not report["qualitative_probes"]["missing"] else "incomplete"
            )

        trace_path = output_dir / "trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n" for row in trace_rows),
            encoding="utf-8",
        )
        if qualitative_rows:
            (output_dir / "qualitative_trace.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n" for row in qualitative_rows),
                encoding="utf-8",
            )
        all_rows = list(trace_rows)
        report["identity"].update(e2e.validate_loaded_identity(manifest, all_rows, args.turn_policy))
        report["identity"]["observed_model_loaded"] = report["identity"].get("loaded_event")
        report["system"]["transport"] = e2e.compute_event_metrics(all_rows)
        report["system"]["status"] = (
            "failed" if report["system"]["transport"].get("server_errors") else
            "observed" if report["system"]["transport"].get("responses_created") else "incomplete"
        )
        coverage_raw = e2e.action_coverage(all_rows, e2e.ACTION_ORDER, args.turn_policy)
        accepted_rows = [row for row in all_rows if e2e._event(row).get("accepted") is True]
        coverage_accepted = e2e.action_coverage(accepted_rows, e2e.ACTION_ORDER, args.turn_policy)
        coverage_accepted["accepted_only"] = True
        report["behavior"]["raw_action_coverage"] = coverage_raw
        report["behavior"]["action_coverage"] = coverage_accepted
        report["behavior"]["live_policy_semantics"] = e2e.live_policy_semantics(all_rows)
        runtime_events = [
            {"session_id": row.get("session_id", row.get("session")),
             "event_type": row.get("event", {}).get("type"),
             "response_id": row.get("event", {}).get("response_id"),
             "metrics": row.get("event", {}).get("runtime_metrics")}
            for row in all_rows
            if row.get("direction") == "server"
            and row.get("event", {}).get("type") in {"response.done", "response.cancelled"}
            and isinstance(row.get("event", {}).get("runtime_metrics"), dict)
        ]
        session_events = [
            {"session_id": row.get("session_id", row.get("session")),
             "event_type": "session.created", "metrics": row.get("event", {}).get("runtime_metrics")}
            for row in all_rows
            if row.get("direction") == "server"
            and row.get("event", {}).get("type") == "session.created"
            and isinstance(row.get("event", {}).get("runtime_metrics"), dict)
        ]
        report["system"]["runtime_metrics"] = {
            "status": "available" if runtime_events or session_events else "unavailable",
            "response_events": runtime_events,
            "session_events": session_events,
            "final_scheduler_summary": "server log line [realtime_session_metrics] emitted on socket close",
        }
        playout_rows = []
        for session_id, details in report["artifacts"]["session_playout"].items():
            playout_rows.extend(
                {"session_id": session_id, "response_id": response_id, **response["metrics"]}
                for response_id, response in details.get("responses", {}).items()
            )
        if playout_rows:
            report["system"]["playback_simulation"] = {
                "status": "available", "endpoint": "simulated_headless_pcm_playout",
                "sample_rate_hz": e2e.OUTPUT_RATE, "frame_ms": e2e.FRAME_MS,
                "pcm_source": "actual response.audio.delta PCM received by the harness",
                "physical_device_playback": False, "device_latency_measured": False,
                "responses": playout_rows,
            }

        barge = next((result for result in canonical_results
                      if result.get("scenario_id") == "05_barge_in_interruption"), None)
        if barge:
            start_ns = barge.get("barge_in_start_ns")
            end_ns = barge.get("barge_response_end_ns")
            observed = bool(
                start_ns and barge.get("overlap_at_barge_start")
                and barge.get("barge_cancel_observed") and barge.get("recovery_response_created")
            )
            report["behavior"]["barge_in"] = {
                "status": "observed" if observed else "incomplete",
                "overlap_at_barge_start": bool(barge.get("overlap_at_barge_start")),
                "cancel_observed": bool(barge.get("barge_cancel_observed")),
                "interrupt_to_cancel_latency_ms": (
                    round((end_ns - start_ns) / 1_000_000.0, 3)
                    if start_ns and end_ns and barge.get("barge_cancel_observed") else None
                ),
                "assistant_last_audio_after_interrupt_ms": (
                    round((barge["barge_last_audio_ns"] - start_ns) / 1_000_000.0, 3)
                    if start_ns and barge.get("barge_last_audio_ns") else None
                ),
                "recovery_response_created": bool(barge.get("recovery_response_created")),
            }
        text_checks_ok = all(
            turn.get("text_behavior_checks", {}).get("status") == "observed"
            and all(check for step in turn["text_behavior_checks"]["steps"] for check in step.get("checks", []))
            for turn in report["behavior"]["turns"]
        )
        all_audio = all(
            turn["input_audio"]["status"] == "available" and turn["output_audio"]["status"] == "available"
            for turn in report["behavior"]["turns"]
        )
        all_session_errors = any(turn.get("session_errors") for turn in report["behavior"]["turns"])
        if report["identity"].get("status") != "matched":
            report["behavior"]["status"] = "unavailable"
        elif (
            coverage_accepted["missing"]
            or report["scenario_coverage"]["status"] != "complete"
            or report["behavior"]["barge_in"]["status"] != "observed"
            or not all_audio or all_session_errors or not text_checks_ok
        ):
            report["behavior"]["status"] = "incomplete"
        else:
            report["behavior"]["status"] = "observed"
        return report
    finally:
        (output_dir / "trace.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n" for row in trace_rows),
            encoding="utf-8",
        )
        if qualitative_rows:
            (output_dir / "qualitative_trace.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n" for row in qualitative_rows),
                encoding="utf-8",
            )


def main(argv: list[str] | None = None) -> int:
    args = e2e._parser().parse_args(argv)
    if args.offline_asr_report:
        return e2e.main(argv)
    if not args.server or args.output_dir is None or not (args.expected_run_id or args.expected_bundle_id):
        e2e._parser().error(
            "sequential capture requires --server, --output-dir and either --expected-run-id or --expected-bundle-id"
        )
    report = asyncio.run(_capture(args))
    output_dir = Path(args.output_dir).expanduser().resolve()
    e2e._write_reports(report, output_dir)
    print(json.dumps({
        "evaluation_status": report.get("evaluation_status"),
        "system": report.get("system", {}).get("status"),
        "identity": report.get("identity", {}).get("status"),
        "behavior": report.get("behavior", {}).get("status"),
        "scenario_coverage": report.get("scenario_coverage", {}).get("status"),
        "report": str(output_dir / "report.json"),
    }, ensure_ascii=False))
    return 0 if report.get("evaluation_status") == "evidence_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
