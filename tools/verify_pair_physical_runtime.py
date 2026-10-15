#!/usr/bin/env python3
"""Fail-closed physical duplex gates over one captured runner report.

This is intentionally additive to the broader certification analyzer. It uses
the raw report/event stream plus the runner manifest and writes a separate
physical_runtime_gate.json; it does not edit or reinterpret captured audio.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


RMS_FLOOR = 0.008
INPUT_FRAME_MS = 80.0
LATENCY_P95_LIMIT_MS = 80.0
MIN_AUDIBLE_RESPONSE_TARGET = 4


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_python_executable(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    executable = Path(value).name.lower()
    return re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", executable) is not None


def _is_workspace_script_entry(value: Any, script: str) -> bool:
    return value in (script, f"./{script}", f"/workspace/{script}")


def _python_argv_runs_script(argv: Sequence[Any], script: str) -> bool:
    return len(argv) >= 2 and _is_python_executable(argv[0]) and _is_workspace_script_entry(argv[1], script)


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _event_rows(events: Sequence[Mapping[str, Any]], kind: str) -> list[Mapping[str, Any]]:
    return [row for row in events if row.get("type") == kind]


def _source_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    rows = manifest.get("source_files")
    if not isinstance(rows, list):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str):
            result[row["path"]] = row["sha256"].lower()
    return result


def _recorded_source_inventory(manifest: Mapping[str, Any], required: Sequence[str]) -> dict[str, str] | None:
    inventory = _source_map(manifest)
    if any(not isinstance(inventory.get(name), str) or len(inventory[name]) != 64 for name in required):
        return None
    if any(any(char not in "0123456789abcdef" for char in inventory[name]) for name in required):
        return None
    return {name: inventory[name] for name in required}


def _manifest_source_matches(root: Path, manifest: Mapping[str, Any], relative: str) -> bool:
    recorded = _source_map(manifest).get(relative)
    path = root / relative
    if not recorded or not path.is_file():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == recorded


def _runner_uses_same_container_exec(root: Path) -> bool:
    """Check the runner source path rather than inferring clocks from timestamps."""
    path = root / "tools" / "run_pair_duplex_proof_3090.py"
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError):
        return False
    run_client = next(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_client"),
        None,
    )
    if run_client is None or not any(arg.arg == "container_id" for arg in run_client.args.args):
        return False
    run_client_source = ast.get_source_segment(source, run_client) or ""
    if '"docker"' not in run_client_source or '"exec"' not in run_client_source or "container_id" not in run_client_source:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_client"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "own_container_id"
        for node in ast.walk(tree)
    )


def attest_shared_monotonic_clock(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Attest cross-process CLOCK_MONOTONIC only from a same-container source path."""
    container = manifest.get("container")
    argv = manifest.get("argv")
    container_id = container.get("id") if isinstance(container, Mapping) else None
    server_argv = argv.get("server") if isinstance(argv, Mapping) else None
    client_argv = argv.get("client") if isinstance(argv, Mapping) else None
    docker_argv = argv.get("docker_run") if isinstance(argv, Mapping) else None
    if not isinstance(container_id, str) or not container_id:
        return {"attested": False, "reason": "container_id_missing"}
    if not all(isinstance(value, list) for value in (server_argv, client_argv, docker_argv)):
        return {"attested": False, "reason": "runner_argv_missing"}
    server_python = bool(server_argv) and _is_python_executable(server_argv[0])
    server_module_entry = (
        server_python
        and len(server_argv) >= 3
        and server_argv[1] == "-m"
        and server_argv[2] == "duplex_kernel.launch"
    )
    server_script_entry = _python_argv_runs_script(server_argv, "duplex_kernel/launch.py")
    client_script_entry = _python_argv_runs_script(client_argv, "client/duplex_stream_e2e.py")
    if not (server_module_entry or server_script_entry) or not client_script_entry:
        return {"attested": False, "reason": "server_or_client_entrypoint_mismatch"}
    if docker_argv[:2] != ["docker", "run"]:
        return {"attested": False, "reason": "server_container_launch_unattested"}
    required_source_paths = (
        "tools/run_pair_duplex_proof_3090.py",
        "client/duplex_stream_e2e.py",
        "duplex_kernel/server.py",
    )
    recorded_hashes = _recorded_source_inventory(manifest, required_source_paths)
    if recorded_hashes is None:
        return {"attested": False, "reason": "clock_source_hash_missing_or_invalid"}
    if not isinstance(manifest.get("git"), Mapping) or not isinstance(manifest["git"].get("status_sha256"), str):
        return {"attested": False, "reason": "dirty_source_snapshot_hash_missing"}
    if not _manifest_source_matches(root, manifest, "tools/run_pair_duplex_proof_3090.py") or not _runner_uses_same_container_exec(root):
        return {"attested": False, "reason": "runner_source_hash_or_same_container_exec_check_failed"}
    # The manifest fingerprints the exact capture-time source tree. The source
    # checkout used later to reanalyze a capture may have advanced, so current
    # local hashes are not substituted for that immutable inventory.
    return {
        "attested": True,
        "clock": "CLOCK_MONOTONIC",
        "server_container_id": container_id,
        "client_container_id": container_id,
        "source_sha256": recorded_hashes,
        "basis": "capture-time source inventory hashes, git status hash, container ID, server/client argv and docker run argv; runner client path uses docker exec into this container",
    }


def _response_index(report: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    raw = report.get("responses")
    if not isinstance(raw, list):
        return {}, ["report.responses missing or not a list"]
    indexed: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for row in raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("response_id"), str) or not row["response_id"]:
            errors.append("response row missing response_id")
            continue
        response_id = row["response_id"]
        if response_id in indexed:
            errors.append(f"duplicate response_id {response_id}")
        indexed[response_id] = row
    return indexed, errors


def _natural_eos_audible_response(row: Mapping[str, Any]) -> bool:
    if row.get("terminal_event") != "response.done" or (_int(row.get("audio_samples")) or 0) <= 0:
        return False
    runtime = row.get("runtime_metrics")
    if not isinstance(runtime, Mapping):
        return False
    generation = runtime.get("generation_trace")
    talker = runtime.get("talker")
    return (
        isinstance(generation, Mapping)
        and generation.get("terminal_reason") == "eos"
        and isinstance(talker, Mapping)
        and talker.get("terminated_by") == "eos"
    )


def _drain_gate(report: Mapping[str, Any], events: Sequence[Mapping[str, Any]], participants: Any) -> dict[str, Any]:
    failures: list[str] = []
    missing: list[str] = []
    drain_rows = _event_rows(events, "probe.response_drain_completed")
    if len(drain_rows) != 1:
        missing.append(f"expected one response_drain_completed event, found {len(drain_rows)}")
    drain = drain_rows[0] if len(drain_rows) == 1 else {}
    snapshot = drain.get("pipeline_snapshot") if isinstance(drain, Mapping) else None
    if not isinstance(snapshot, Mapping):
        missing.append("drain pipeline_snapshot missing")
        snapshot = {}
    if report.get("pair_completion_status") != "complete" or report.get("pair_drain_completed") is not True:
        failures.append("top-level report does not claim a complete drained pair")

    created = _int(report.get("created_responses"))
    audible = _int(report.get("audible_completed_responses"))
    target = _int(report.get("target_audible_responses"))
    for key, expected in (("created_responses", created), ("audible_completed_responses", audible), ("target_audible_responses", target)):
        observed = _int(drain.get(key)) if isinstance(drain, Mapping) else None
        if observed is None or expected is None:
            missing.append(f"drain/report {key} accounting missing")
        elif observed != expected:
            failures.append(f"drain/report {key} mismatch: {observed} != {expected}")
    if audible is None or target is None:
        missing.append("report audible/target response counts missing")
    else:
        if target < MIN_AUDIBLE_RESPONSE_TARGET:
            failures.append(f"reported audible response target {target} is below the minimum target {MIN_AUDIBLE_RESPONSE_TARGET}")
        if audible < target:
            failures.append(f"audible response accounting {audible} is below target {target}")
    responses, response_errors = _response_index(report)
    missing.extend(response_errors)
    natural_audible_rows = [row for row in responses.values() if _natural_eos_audible_response(row)]
    if audible is not None and len(natural_audible_rows) != audible:
        failures.append(f"natural-EOS completed audible response rows {len(natural_audible_rows)} != report count {audible}")
    if len(natural_audible_rows) < MIN_AUDIBLE_RESPONSE_TARGET:
        failures.append(f"natural-EOS completed audible response count {len(natural_audible_rows)} is below the minimum {MIN_AUDIBLE_RESPONSE_TARGET}")
    if created is not None and len(responses) != created:
        failures.append(f"response rows {len(responses)} != created response count {created}")

    for speaker in ("A", "B"):
        peer = snapshot.get(speaker)
        if not isinstance(peer, Mapping):
            missing.append(f"drain snapshot missing peer {speaker}")
            continue
        for key in ("peer_audio_queue_frames", "route_pending_items", "unsent_pcm_frames", "pending_ack_frames"):
            value = _int(peer.get(key))
            if value is None:
                missing.append(f"drain snapshot {speaker}.{key} missing")
            elif value != 0:
                failures.append(f"drain snapshot {speaker}.{key}={value}")
        active = peer.get("active_response_ids")
        if not isinstance(active, list):
            missing.append(f"drain snapshot {speaker}.active_response_ids missing")
        elif active:
            failures.append(f"drain snapshot {speaker} still has active responses")
        enqueued = _int(peer.get("enqueued_pcm_frames"))
        sent = _int(peer.get("sent_pcm_frames"))
        if enqueued is None or sent is None:
            missing.append(f"drain snapshot {speaker} PCM frame accounting missing")
        elif enqueued != sent:
            failures.append(f"drain snapshot {speaker} enqueued/sent PCM differ: {enqueued}/{sent}")

    if not isinstance(participants, list):
        missing.append("report participants missing")
    else:
        participant_map = {row.get("speaker"): row for row in participants if isinstance(row, Mapping)}
        for speaker in ("A", "B"):
            row = participant_map.get(speaker)
            if not isinstance(row, Mapping):
                missing.append(f"participant summary missing for {speaker}")
                continue
            sent = _int(row.get("input_frames_sent"))
            acked = _int(row.get("input_frames_acked"))
            pending = _int(row.get("pending_acks_at_close"))
            missing_ack = _int(row.get("missing_ack_count"))
            if None in (sent, acked, pending, missing_ack):
                missing.append(f"participant {speaker} sent/ACK/pending fields missing")
            elif sent != acked or pending != 0 or missing_ack != 0:
                failures.append(f"participant {speaker} input drain is not lossless: sent={sent} ACK={acked} pending={pending} missing={missing_ack}")
    status = "FAIL" if failures else "INCOMPLETE" if missing else "PASS"
    return {
        "status": status,
        "minimum_audible_response_target": MIN_AUDIBLE_RESPONSE_TARGET,
        "natural_eos_audible_response_count": len(natural_audible_rows),
        "failures": failures,
        "missing": missing,
        "drain_event_count": len(drain_rows),
    }


def _native_decision_gate(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, int], Mapping[str, Any]]]:
    all_rows = _event_rows(events, "duplex_control6.decision")
    rows = [row for row in all_rows if row.get("native_control6_head_scored") is True]
    scheduler_rows = [row for row in all_rows if row.get("native_control6_head_scored") is False]
    failures: list[str] = []
    missing: list[str] = []
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    age_count = 0
    accepted_count = 0
    expected_disabled_rejections = 0
    rejection_reasons: Counter[str] = Counter()
    target_rows = _event_rows(events, "probe.audible_response_target_reached")
    target_at_ms = _number(target_rows[0].get("at_ms")) if len(target_rows) == 1 else None
    if len(target_rows) != 1 or target_at_ms is None:
        missing.append("expected one timestamped audible_response_target_reached event for expected post-target rejection classification")
    for row in rows:
        speaker = row.get("speaker")
        frame = _int(row.get("frame_id"))
        if speaker not in ("A", "B") or frame is None:
            missing.append("native decision missing speaker/frame id")
            continue
        key = (speaker, frame)
        if key in indexed:
            failures.append(f"duplicate native decision for {speaker} frame {frame}")
        indexed[key] = row
        if row.get("native_control_hidden_supplied") is not True:
            missing.append(f"head-scored {speaker} frame {frame} lacks supplied listener hidden state")
        hidden_frame = _int(row.get("native_control_hidden_frame_id"))
        if hidden_frame is None:
            missing.append(f"head-scored {speaker} frame {frame} lacks hidden-state frame ID")
        elif hidden_frame != frame:
            failures.append(f"head-scored {speaker} frame {frame} used hidden frame {hidden_frame}")
        accepted = row.get("accepted")
        reason = row.get("rejection_reason")
        if accepted is True:
            accepted_count += 1
            if reason not in (None, ""):
                failures.append(f"accepted native decision has rejection reason for {speaker} frame {frame}")
        elif accepted is False:
            if not isinstance(reason, str) or not reason:
                missing.append(f"rejected native decision lacks reason for {speaker} frame {frame}")
            else:
                rejection_reasons[reason] += 1
                if reason == "auto_response_disabled" and target_at_ms is not None:
                    decision_at_ms = _number(row.get("at_ms"))
                    if decision_at_ms is not None and decision_at_ms >= target_at_ms:
                        expected_disabled_rejections += 1
                    else:
                        failures.append(f"auto_response_disabled rejection precedes audible target for {speaker} frame {frame}")
                else:
                    failures.append(f"unexpected native decision rejection for {speaker} frame {frame}: {reason}")
        else:
            missing.append(f"native decision acceptance missing for {speaker} frame {frame}")

        hidden_age = _int(row.get("listener_hidden_age_frames"))
        hidden_age_ms = _number(row.get("listener_hidden_age_ms"))
        if hidden_age is None:
            missing.append(f"{speaker} frame {frame} listener_hidden_age_frames missing")
        elif not 0 <= hidden_age <= 1:
            failures.append(f"{speaker} frame {frame} listener_hidden_age_frames={hidden_age} outside 0..1")
        else:
            age_count += 1
        if hidden_age_ms is None:
            missing.append(f"{speaker} frame {frame} listener_hidden_age_ms missing")
        elif hidden_age_ms < 0:
            failures.append(f"{speaker} frame {frame} listener_hidden_age_ms is negative")

        reused = row.get("listener_query_reused")
        if reused is True:
            query_age = _int(row.get("listener_query_age_frames"))
            if query_age is None:
                missing.append(f"{speaker} frame {frame} reused listener query age missing")
            elif not 0 <= query_age <= 1:
                failures.append(f"{speaker} frame {frame} reused listener query age={query_age} outside 0..1")
            else:
                age_count += 1
        elif reused is not False:
            missing.append(f"{speaker} frame {frame} listener_query_reused flag missing")
    if all_rows and not rows:
        failures.append("no accepted native decisions")
    if not rows:
        missing.append("no head-scored native decision events")
    fallback_sources = Counter(str(row.get("source") or "unknown") for row in scheduler_rows)
    if len(scheduler_rows) + len(rows) != len(all_rows):
        missing.append("some decision events lack an explicit native head-scored flag")
    status = "FAIL" if failures else "INCOMPLETE" if missing else "PASS"
    return ({"status": status, "decision_event_count": len(all_rows), "head_scored_count": len(rows), "accepted_head_scored_count": accepted_count, "rejected_head_scored_count": sum(row.get("accepted") is False for row in rows), "expected_post_target_auto_response_disabled_rejections": expected_disabled_rejections, "rejection_reason_counts": dict(sorted(rejection_reasons.items())), "scheduler_fallback_count": len(scheduler_rows), "scheduler_fallback_sources": dict(sorted(fallback_sources.items())), "age_values_checked": age_count, "failures": failures, "missing": missing}, indexed)


def _percentile_gate(values_by_speaker: Mapping[str, list[float]], *, label: str) -> dict[str, Any]:
    missing = [f"{label} has no values for {speaker}" for speaker in ("A", "B") if not values_by_speaker.get(speaker)]
    p95_by_speaker = {speaker: _p95(values_by_speaker.get(speaker, [])) for speaker in ("A", "B")}
    failures = [f"{label} {speaker} p95 {p95:.3f} ms exceeds {LATENCY_P95_LIMIT_MS:.1f} ms" for speaker, p95 in p95_by_speaker.items() if p95 is not None and p95 > LATENCY_P95_LIMIT_MS]
    all_values = [value for speaker in ("A", "B") for value in values_by_speaker.get(speaker, [])]
    overall = _p95(all_values)
    if overall is not None and overall > LATENCY_P95_LIMIT_MS:
        failures.append(f"{label} pooled p95 {overall:.3f} ms exceeds {LATENCY_P95_LIMIT_MS:.1f} ms")
    status = "FAIL" if failures else "INCOMPLETE" if missing else "PASS"
    return {"status": status, "limit_ms": LATENCY_P95_LIMIT_MS, "percentile_method": "linear", "p95_ms": overall, "p95_by_speaker_ms": p95_by_speaker, "sample_count": len(all_values), "failures": failures, "missing": missing}


def _aggregate_gate_status(statuses: Sequence[Any]) -> str:
    normalized = {
        status if status in {"PASS", "FAIL", "INCOMPLETE"} else "INCOMPLETE"
        for status in statuses
    }
    if "FAIL" in normalized:
        return "FAIL"
    if "INCOMPLETE" in normalized:
        return "INCOMPLETE"
    return "PASS"


def _latency_gates(
    events: Sequence[Mapping[str, Any]],
    clock_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    sent_rows = _event_rows(events, "probe.input_frame_sent")
    ack_rows = _event_rows(events, "probe.input_frame_ack")
    all_decision_rows = _event_rows(events, "duplex_control6.decision")
    sends: dict[tuple[str, int], Mapping[str, Any]] = {}
    acks: dict[tuple[str, int], Mapping[str, Any]] = {}
    all_decisions: dict[tuple[str, int], Mapping[str, Any]] = {}
    structural_missing: list[str] = []
    structural_failures: list[str] = []
    for row in sent_rows:
        speaker, frame = row.get("speaker"), _int(row.get("frame_id"))
        if speaker not in ("A", "B") or frame is None:
            structural_missing.append("input send missing speaker/frame id")
            continue
        key = (speaker, frame)
        if key in sends:
            structural_failures.append(f"duplicate input send {speaker} frame {frame}")
        sends[key] = row
    for row in ack_rows:
        speaker, frame = row.get("speaker"), _int(row.get("client_frame_id"))
        ack_frame = _int(row.get("ack_frame_id"))
        if speaker not in ("A", "B") or frame is None or ack_frame is None:
            structural_missing.append("input ACK missing speaker/frame identity")
            continue
        if frame != ack_frame:
            structural_failures.append(f"input ACK identity mismatch {speaker} client={frame} ack={ack_frame}")
        key = (speaker, frame)
        if key in acks:
            structural_failures.append(f"duplicate input ACK {speaker} frame {frame}")
        acks[key] = row
    for row in all_decision_rows:
        speaker, frame = row.get("speaker"), _int(row.get("frame_id"))
        if speaker in ("A", "B") and frame is not None:
            key = (speaker, frame)
            if key in all_decisions:
                structural_failures.append(f"duplicate decision event {speaker} frame {frame}")
            all_decisions[key] = row

    ack_values: dict[str, list[float]] = {"A": [], "B": []}
    scored_event_values: dict[str, list[float]] = {"A": [], "B": []}
    fallback_event_values: dict[str, list[float]] = {"A": [], "B": []}
    all_event_values: dict[str, list[float]] = {"A": [], "B": []}
    server_decision_values: dict[str, list[float]] = {"A": [], "B": []}
    accepted_server_decision_values: dict[str, list[float]] = {"A": [], "B": []}
    fallback_server_decision_values: dict[str, list[float]] = {"A": [], "B": []}
    all_server_decision_values: dict[str, list[float]] = {"A": [], "B": []}
    missing_ack_pairs: list[str] = []
    missing_decision_pairs: list[str] = []
    for key, sent in sends.items():
        speaker, frame = key
        start_ns = _int(sent.get("send_started_at_ns"))
        sent_ns = _int(sent.get("sent_at_ns"))
        if start_ns is None or sent_ns is None or sent_ns < start_ns:
            structural_missing.append(f"{speaker} frame {frame} missing/invalid append timestamps")
        ack = acks.get(key)
        if ack is None:
            missing_ack_pairs.append(f"{speaker}:{frame}")
        elif start_ns is not None:
            ack_ns = _int(ack.get("ack_received_at_ns"))
            if ack_ns is None:
                structural_missing.append(f"{speaker} frame {frame} missing ACK timestamp")
            elif ack_ns < start_ns:
                structural_failures.append(f"{speaker} frame {frame} ACK precedes append start")
            else:
                ack_values[speaker].append((ack_ns - start_ns) / 1_000_000.0)
        decision_event = all_decisions.get(key)
        if decision_event is None:
            missing_decision_pairs.append(f"{speaker}:{frame}")
            continue
        sent_ms = _number(sent.get("at_ms"))
        decision_ms = _number(decision_event.get("at_ms"))
        if start_ns is not None and sent_ns is not None and decision_ms is not None and sent_ms is not None:
            append_start_elapsed_ms = sent_ms - (sent_ns - start_ns) / 1_000_000.0
            client_elapsed = decision_ms - append_start_elapsed_ms
            if client_elapsed < 0:
                structural_failures.append(f"{speaker} frame {frame} decision event precedes append start on client clock")
            else:
                all_event_values[speaker].append(client_elapsed)
                if decision_event.get("native_control6_head_scored") is True:
                    scored_event_values[speaker].append(client_elapsed)
                elif decision_event.get("native_control6_head_scored") is False:
                    fallback_event_values[speaker].append(client_elapsed)
        elif decision_ms is None or sent_ms is None:
            structural_missing.append(f"{speaker} frame {frame} missing client event-clock timestamps")
        policy_ns = _int(decision_event.get("policy_decision_at_ns"))
        if start_ns is not None and policy_ns is not None:
            delta = (policy_ns - start_ns) / 1_000_000.0
            if delta < 0:
                structural_failures.append(f"{speaker} frame {frame} native policy timestamp precedes append start")
            else:
                all_server_decision_values[speaker].append(delta)
                if decision_event.get("native_control6_head_scored") is True:
                    server_decision_values[speaker].append(delta)
                elif decision_event.get("native_control6_head_scored") is False:
                    fallback_server_decision_values[speaker].append(delta)
                if decision_event.get("native_control6_head_scored") is True and decision_event.get("accepted") is True:
                    accepted_server_decision_values[speaker].append(delta)
        elif policy_ns is None:
            structural_missing.append(f"{speaker} frame {frame} lacks policy_decision_at_ns")
    unmatched_acks = sorted(f"{speaker}:{frame}" for speaker, frame in acks if (speaker, frame) not in sends)
    if unmatched_acks:
        structural_failures.append(f"ACKs without matching sent frames: {unmatched_acks[:8]}")

    ack_gate = _percentile_gate(ack_values, label="append_to_ack")
    ack_gate["measurement_semantics"] = (
        "client receipt of input_audio_buffer.consumed after the listener consumes the frame; "
        "this ingress ACK precedes the separate policy callback and does not attest policy completion"
    )
    ack_gate["policy_decision_metrics"] = [
        "append_to_native_policy_decision",
        "append_to_accepted_native_policy_decision",
    ]
    scored_observed_gate = _percentile_gate(scored_event_values, label="append_to_head_scored_decision_event_received")
    fallback_observed_gate = _percentile_gate(fallback_event_values, label="append_to_scheduler_fallback_event_received")
    all_observed_gate = _percentile_gate(all_event_values, label="append_to_all_decision_events_received")
    if structural_failures:
        ack_gate["status"] = "FAIL"
        for gate in (scored_observed_gate, fallback_observed_gate, all_observed_gate):
            gate["status"] = "FAIL"
    elif structural_missing or missing_ack_pairs or missing_decision_pairs:
        if ack_gate["status"] != "FAIL":
            ack_gate["status"] = "INCOMPLETE"
        for gate in (scored_observed_gate, fallback_observed_gate, all_observed_gate):
            if gate["status"] != "FAIL":
                gate["status"] = "INCOMPLETE"
    ack_gate["missing_frame_pairs"] = missing_ack_pairs
    ack_gate["structural_failures"] = structural_failures
    ack_gate["structural_missing"] = structural_missing
    for gate in (scored_observed_gate, fallback_observed_gate, all_observed_gate):
        gate["missing_frame_pairs"] = missing_decision_pairs
        gate["structural_failures"] = structural_failures
        gate["structural_missing"] = structural_missing

    server_gate = _percentile_gate(server_decision_values, label="append_to_native_policy_decision")
    accepted_server_gate = _percentile_gate(accepted_server_decision_values, label="append_to_accepted_native_policy_decision")
    fallback_server_gate = _percentile_gate(fallback_server_decision_values, label="append_to_scheduler_fallback_policy_decision")
    all_server_gate = _percentile_gate(all_server_decision_values, label="append_to_all_policy_decision")
    if not clock_attestation.get("attested"):
        for gate in (server_gate, accepted_server_gate, fallback_server_gate, all_server_gate):
            gate["status"] = "INCOMPLETE"
            gate["not_used_for_pass"] = True
            gate["clock_attestation_reason"] = clock_attestation.get("reason", "unattested_clock_domain")
    elif structural_failures:
        for gate in (server_gate, accepted_server_gate, fallback_server_gate, all_server_gate):
            gate["status"] = "FAIL"
    elif structural_missing or missing_decision_pairs:
        for gate in (server_gate, accepted_server_gate, fallback_server_gate, all_server_gate):
            gate["status"] = "INCOMPLETE"
    gates = {
        "append_to_ack": ack_gate,
        "append_to_head_scored_decision_event_received": scored_observed_gate,
        "append_to_scheduler_fallback_event_received": fallback_observed_gate,
        "append_to_all_decision_events_received": all_observed_gate,
        "append_to_native_policy_decision": server_gate,
        "append_to_accepted_native_policy_decision": accepted_server_gate,
        "append_to_scheduler_fallback_policy_decision": fallback_server_gate,
        "append_to_all_policy_decision": all_server_gate,
        "clock_attestation": dict(clock_attestation),
    }
    gates["status"] = _aggregate_gate_status(
        value.get("status") for value in gates.values() if isinstance(value, Mapping)
    )
    return gates


def _chunk_intervals(events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    intervals: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in _event_rows(events, "probe.response_audio_chunk"):
        response_id = row.get("response_id")
        speaker = row.get("speaker")
        server_start = _int(row.get("server_emit_started_at_ns"))
        client_start = _int(row.get("received_at_ns"))
        samples = _int(row.get("sample_count"))
        rate = _int(row.get("sample_rate_hz"))
        chunk_id = _int(row.get("output_chunk_id"))
        quality = row.get("raw_source_audio_quality")
        rms = _number(quality.get("rms")) if isinstance(quality, Mapping) else None
        if not isinstance(response_id, str) or speaker not in ("A", "B") or samples is None or rate is None or samples <= 0 or rate <= 0 or chunk_id is None:
            missing.append("response chunk lacks response/speaker/sample bounds/chunk id")
            continue
        if client_start is None:
            missing.append(f"response {response_id} chunk {chunk_id} lacks client receive timestamp")
        if rms is None:
            missing.append(f"response {response_id} chunk {chunk_id} source RMS missing")
            continue
        duration_ns = round(samples * 1_000_000_000 / rate)
        intervals.append({
            "speaker": speaker,
            "response_id": response_id,
            "chunk_id": chunk_id,
            "server_start_ns": server_start,
            "server_end_ns": server_start + duration_ns if server_start is not None else None,
            "client_start_ns": client_start,
            "client_end_ns": client_start + duration_ns if client_start is not None else None,
            "rms": rms,
        })
    return intervals, missing


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted((start, end) for start, end in intervals if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _intersect_interval_unions(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def _overlap_gate(
    report: Mapping[str, Any], events: Sequence[Mapping[str, Any]], clock_attestation: Mapping[str, Any]
) -> dict[str, Any]:
    responses, response_errors = _response_index(report)
    chunks, chunk_missing = _chunk_intervals(events)
    failures = list(response_errors)
    missing = list(chunk_missing)
    chunks_by_response: dict[str, list[dict[str, Any]]] = {}
    chunk_ids: dict[str, set[int]] = {}
    for chunk in chunks:
        chunks_by_response.setdefault(chunk["response_id"], []).append(chunk)
        chunk_ids.setdefault(chunk["response_id"], set()).add(chunk["chunk_id"])

    # Same-server-clock output/output overlap is kept explicitly virtual.
    virtual_windows: list[dict[str, Any]] = []
    a_chunks = [c for c in chunks if c["speaker"] == "A" and c["rms"] > RMS_FLOOR and c["server_start_ns"] is not None]
    b_chunks = [c for c in chunks if c["speaker"] == "B" and c["rms"] > RMS_FLOOR and c["server_start_ns"] is not None]
    for a in a_chunks:
        for b in b_chunks:
            start, end = max(a["server_start_ns"], b["server_start_ns"]), min(a["server_end_ns"], b["server_end_ns"])
            if end > start:
                virtual_windows.append({"start_ns": start, "end_ns": end, "duration_ms": (end - start) / 1_000_000.0, "response_A": a["response_id"], "response_B": b["response_id"]})

    candidate_frames: set[tuple[str, int, str]] = set()
    input_intervals: dict[str, list[tuple[int, int]]] = {"A": [], "B": []}
    local_generation_frames: set[tuple[str, int, str]] = set()
    routed_missing: list[str] = []
    for row in _event_rows(events, "probe.input_frame_sent"):
        speaker = row.get("speaker")
        frame = _int(row.get("frame_id"))
        rms = _number(row.get("rms"))
        source_id = row.get("source_response_id")
        local_ids = row.get("local_audio_response_ids")
        source_chunk_ids = row.get("source_response_audio_chunk_ids")
        if speaker not in ("A", "B") or frame is None or rms is None or not isinstance(source_id, str) or not isinstance(local_ids, list) or not isinstance(source_chunk_ids, list):
            routed_missing.append("input send event missing routed PCM attribution fields")
            continue
        if rms <= RMS_FLOOR or source_id in {"seed", "live_silence", "endpoint_silence"}:
            continue
        source_response = responses.get(source_id)
        if not isinstance(source_response, Mapping):
            routed_missing.append(f"{speaker} frame {frame} source response {source_id} absent from report")
            continue
        if source_response.get("speaker") == speaker:
            failures.append(f"{speaker} frame {frame} source response is not from the peer")
            continue
        if source_response.get("terminal_event") != "response.done" or (_int(source_response.get("audio_samples")) or 0) <= 0:
            routed_missing.append(f"{speaker} frame {frame} source response {source_id} is not a completed audible response")
            continue
        if not source_chunk_ids or any(_int(value) is None or _int(value) not in chunk_ids.get(source_id, set()) for value in source_chunk_ids):
            routed_missing.append(f"{speaker} frame {frame} source chunk IDs are missing/unmatched")
            continue
        start = _int(row.get("send_started_at_ns"))
        if start is None:
            routed_missing.append(f"{speaker} frame {frame} missing send_started_at_ns")
            continue
        frame_end = start + int(INPUT_FRAME_MS * 1_000_000)
        candidate_frames.add((speaker, frame, source_id))
        input_intervals[speaker].append((start, frame_end))
        for local_id in local_ids:
            local_response = responses.get(local_id) if isinstance(local_id, str) else None
            if not isinstance(local_response, Mapping):
                continue
            if local_response.get("speaker") != speaker:
                continue
            local_chunks = chunks_by_response.get(local_id, [])
            timed_chunks = [chunk for chunk in local_chunks if chunk["client_start_ns"] is not None and chunk["client_end_ns"] is not None]
            if timed_chunks:
                local_generation_frames.add((speaker, frame, local_id))

    actual_windows: list[dict[str, Any]] = []
    input_unions = {speaker: _merge_intervals(input_intervals[speaker]) for speaker in ("A", "B")}
    for start, end in _intersect_interval_unions(input_unions["A"], input_unions["B"]):
        actual_windows.append({"start_ns": start, "end_ns": end, "overlap_ms": (end - start) / 1_000_000.0})
    if not candidate_frames:
        failures.append("no high-RMS routed PCM frame from a completed peer response")
    if candidate_frames and not actual_windows:
        failures.append("no simultaneous interval intersection between A<-B and B<-A routed PCM")
    if candidate_frames and not clock_attestation.get("attested"):
        missing.append("two-peer routed overlap requires source-attested shared CLOCK_MONOTONIC")
    status = "FAIL" if failures else "INCOMPLETE" if missing else "PASS"
    directional_ms = {speaker: sum(end - start for start, end in input_unions[speaker]) / 1_000_000.0 for speaker in ("A", "B")}
    return {
        "status": status,
        "rms_floor": RMS_FLOOR,
        "input_frame_duration_ms": INPUT_FRAME_MS,
        "routed_frame_candidates": len(candidate_frames),
        "routed_frame_candidates_by_receiver": {speaker: sum(key[0] == speaker for key in candidate_frames) for speaker in ("A", "B")},
        "local_generation_active_frame_mappings": len(local_generation_frames),
        "routed_peer_pcm_union_duration_by_receiver_ms": directional_ms,
        "routed_pcm_overlap_windows": len(actual_windows),
        "routed_pcm_overlap_duration_ms": sum(row["overlap_ms"] for row in actual_windows),
        "routed_windows": actual_windows,
        "virtual_server_output_overlap_windows": len(virtual_windows),
        "virtual_server_output_overlap_duration_ms": sum(row["duration_ms"] for row in virtual_windows),
        "virtual_windows": virtual_windows,
        "virtual_overlap_is_not_routed_pcm_evidence": True,
        "failures": failures,
        "missing": missing + routed_missing,
    }


def _native_talker_gate(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    responses, response_errors = _response_index(report)
    failures = list(response_errors)
    missing: list[str] = []
    checked: list[dict[str, Any]] = []
    audible = [row for row in responses.values() if (_int(row.get("audio_samples")) or 0) > 0]
    if not audible:
        missing.append("no audible responses to attest")
    docker_argv = manifest.get("argv", {}).get("docker_run") if isinstance(manifest.get("argv"), Mapping) else None
    graph_setting: str | None = None
    if isinstance(docker_argv, list):
        for index, arg in enumerate(docker_argv):
            env_value = None
            if arg in ("--env", "-e") and index + 1 < len(docker_argv):
                env_value = docker_argv[index + 1]
            elif isinstance(arg, str) and arg.startswith("--env="):
                env_value = arg[len("--env="):]
            if isinstance(env_value, str) and env_value.startswith("DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH="):
                graph_setting = env_value.split("=", 1)[1]
                break
    expected_graph: bool | None = {"1": True, "0": False, "true": True, "false": False}.get(graph_setting.lower()) if isinstance(graph_setting, str) else None
    if expected_graph is None:
        missing.append("captured manifest does not attest DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH as 0/1")
    for row in audible:
        runtime = row.get("runtime_metrics")
        talker = runtime.get("talker") if isinstance(runtime, Mapping) else None
        backend = talker.get("talker_q1_backend") if isinstance(talker, Mapping) else None
        calls = _int(talker.get("native_talker_q1_calls")) if isinstance(talker, Mapping) else None
        graph_requested = talker.get("native_talker_q1_graph_requested") if isinstance(talker, Mapping) else None
        graph_calls = _int(talker.get("native_talker_q1_graph_calls")) if isinstance(talker, Mapping) else None
        item = {"response_id": row.get("response_id"), "backend": backend, "native_calls": calls, "graph_requested": graph_requested, "graph_calls": graph_calls}
        checked.append(item)
        if backend != "native_audio_q1" or calls is None:
            missing.append(f"audible response {row.get('response_id')} lacks native Q1 backend/call metric")
        elif calls <= 0:
            failures.append(f"audible response {row.get('response_id')} has no native Q1 calls")
        if not isinstance(graph_requested, bool):
            missing.append(f"audible response {row.get('response_id')} lacks boolean graph-requested metric")
        elif expected_graph is not None and graph_requested != expected_graph:
            failures.append(f"audible response {row.get('response_id')} graph-requested flag does not match manifest setting {int(expected_graph)}")
        elif expected_graph is True and (graph_calls is None or graph_calls <= 0):
            failures.append(f"audible response {row.get('response_id')} requested graph but has no graph Q1 calls")
    attestation = manifest.get("native_talker_postrun")
    if isinstance(attestation, Mapping):
        if attestation.get("status") != "PASS":
            failures.append("runner native_talker_postrun attestation did not PASS")
        if _int(attestation.get("audible_response_count")) != len(audible):
            failures.append("runner native_talker_postrun response count does not match raw report")
    elif not checked or any(item.get("native_calls", 0) <= 0 for item in checked if _int(item.get("native_calls")) is not None):
        missing.append("runner composite native Talker attestation absent")
    status = "FAIL" if failures else "INCOMPLETE" if missing else "PASS"
    return {"status": status, "audible_response_count": len(audible), "responses": checked, "failures": failures, "missing": missing}


def verify_physical_gates(report: Mapping[str, Any], events: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    clock = attest_shared_monotonic_clock(root, manifest)
    native_decisions, indexed_decisions = _native_decision_gate(events)
    latency = _latency_gates(events, clock)
    overlap = _overlap_gate(report, events, clock)
    drain = _drain_gate(report, events, report.get("participants"))
    native_talker = _native_talker_gate(report, manifest)
    gates = {
        "accepted_native_decisions_and_fresh_ages": native_decisions,
        "append_latency": latency,
        "native_talker_calls": native_talker,
        "routed_pcm_overlap": overlap,
        "response_drain_and_accounting": drain,
    }
    def nested_statuses(value: Any, path: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        if isinstance(value, Mapping):
            if "status" in value:
                raw_status = value.get("status")
                normalized = raw_status if raw_status in ("PASS", "FAIL", "INCOMPLETE") else "INCOMPLETE"
                found.append((path, normalized))
            for key, child in value.items():
                if key != "status":
                    found.extend(nested_statuses(child, f"{path}.{key}"))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found.extend(nested_statuses(child, f"{path}[{index}]"))
        return found

    status_sources = [entry for key, value in gates.items() for entry in nested_statuses(value, key)]
    statuses = [status for _, status in status_sources]
    overall = _aggregate_gate_status(statuses)
    return {"schema_version": 1, "status": overall, "gates": gates, "nested_status_sources": dict(status_sources), "clock_attestation": clock}


def verify_capture(capture_dir: Path) -> dict[str, Any]:
    def load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} must contain a JSON object")
        return value

    report = load_json(capture_dir / "report.json")
    manifest = load_json(capture_dir / "server_run_manifest.json")
    events_path = capture_dir / "events.jsonl"
    events: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"events.jsonl:{line_no} must be an object")
        events.append(row)
    result = verify_physical_gates(report, events, manifest, capture_dir.parent.parent)
    result["capture_dir"] = str(capture_dir)
    result["event_count"] = len(events)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="capture directory with report.json, events.jsonl, and server_run_manifest.json")
    parser.add_argument("--output", type=Path, default=None, help="separate output JSON (defaults to physical_runtime_gate.json in the capture)")
    args = parser.parse_args()
    result = verify_capture(args.input_dir.resolve())
    output = args.output or (args.input_dir / "physical_runtime_gate.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{result['status']}: {output}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
