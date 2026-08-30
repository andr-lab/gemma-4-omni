#!/usr/bin/env python3
"""Independent verifier for P0 reports.

The verifier treats the JSONL trace as the source of truth: request counts,
health counters, timings, RTFs, frame/sample totals and CUDA peaks are all
recomputed here rather than trusted from the runner summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def verify_source_attestation(report: dict[str, Any], report_path: Path, errors: list[str]) -> None:
    """Recompute the attestation against its content-addressed Git commit.

    Evidence from an earlier clean workspace remains verifiable after later
    source commits; the attestation itself binds the executed bytes to the
    recorded Git object.  The current checkout is still required to be clean.
    """
    relative = report.get("source_attestation")
    if not relative:
        fail(errors, "source_attestation mancante")
        return
    path = Path(relative)
    if not path.is_absolute():
        path = report_path.parent / path
    if not path.is_file():
        fail(errors, f"source attestation mancante: {path}")
        return
    if sha256(path) != report.get("source_attestation_sha256"):
        fail(errors, "hash source attestation non coincide")
        return
    try:
        attestation = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"source attestation JSON invalido: {exc}")
        return
    if attestation.get("schema_version") != 1 or attestation.get("attestation_type") != "git-clean-worktree-content-addressed":
        fail(errors, "tipo/schema source attestation non valido")
    if attestation.get("git_dirty") is not False:
        fail(errors, "source attestation dichiara un checkout dirty")
    if attestation.get("git_sha") != report.get("git_sha"):
        fail(errors, "git_sha source attestation diverso dal report")
    repo = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
        dirty = subprocess.check_output(("git", "status", "--porcelain"), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
        # A report may be verified from a later documentation/evidence commit;
        # the attested commit must exist, while every attested worktree byte
        # below must still match it.
        subprocess.check_output(("git", "cat-file", "-e", f"{attestation['git_sha']}^{{commit}}"),
                                 cwd=repo, stderr=subprocess.STDOUT)
        if dirty:
            fail(errors, "verifier checkout Git dirty")
    except Exception as exc:
        fail(errors, f"impossibile verificare checkout Git: {exc}")
        return
    entries = attestation.get("files")
    if not isinstance(entries, dict) or not entries:
        fail(errors, "source attestation senza file")
        return
    for relative_name, expected in entries.items():
        try:
            # Validate the exact historical bytes that were executed, rather
            # than assuming the current HEAD still contains that workspace.
            data = subprocess.check_output(("git", "show", f"{attestation['git_sha']}:{relative_name}"), cwd=repo)
        except Exception as exc:
            fail(errors, f"impossibile leggere blob Git {relative_name}: {exc}")
            continue
        if expected.get("bytes") != len(data):
            fail(errors, f"size attestata diversa: {relative_name}")
        if expected.get("sha256") != hashlib.sha256(data).hexdigest():
            fail(errors, f"sha256 Git diversa: {relative_name}")
        blob = hashlib.sha1(f"blob {len(data)}".encode("ascii") + b"\x00" + data).hexdigest()
        if expected.get("git_blob_sha1") != blob:
            fail(errors, f"Git blob diversa: {relative_name}")
        if expected.get("git_show_sha256") != hashlib.sha256(data).hexdigest():
            fail(errors, f"git show non coincide: {relative_name}")


def verify_workspace_binding(report: dict[str, Any], events: list[dict[str, Any]], report_path: Path,
                             errors: list[str]) -> None:
    binding_events = [e for e in events if e.get("event") == "source_workspace_binding"]
    file_events = [e for e in events if e.get("event") == "source_workspace_file"]
    binding = report.get("source_workspace_binding")
    if len(binding_events) != 1:
        fail(errors, f"source_workspace_binding count={len(binding_events)}")
        return
    event = binding_events[0]
    if binding != {k: event.get(k) for k in binding if k in event}:
        fail(errors, "source workspace binding report/raw non coincide")
    if event.get("passed") is not True or not event.get("workspace"):
        fail(errors, "source workspace binding non attestato come PASS")
    attestation_path = Path(report.get("source_attestation", ""))
    if not attestation_path.is_absolute():
        attestation_path = report_path.parent / attestation_path
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        expected = attestation.get("files", {})
    except Exception as exc:
        fail(errors, f"impossibile leggere attestation per workspace binding: {exc}")
        return
    actual = {}
    for item in file_events:
        relative = item.get("path")
        if relative in actual:
            fail(errors, f"source workspace file duplicato: {relative}")
        actual[relative] = {k: item.get(k) for k in ("path", "bytes", "sha256", "git_blob_sha1")}
    expected_events = {
        relative: {"path": relative, "bytes": value.get("bytes"), "sha256": value.get("sha256"),
                   "git_blob_sha1": value.get("git_blob_sha1")}
        for relative, value in expected.items()
    }
    if actual != expected_events:
        fail(errors, "source workspace file digests non coincidono con attestation")
    if event.get("files_verified") != len(expected) or len(file_events) != len(expected):
        fail(errors, "conteggio file verificati workspace non coincide")
    if event.get("attestation_git_sha") != attestation.get("git_sha"):
        fail(errors, "workspace binding git_sha diverso dall'attestation")


def pstats(events: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(e[key]) for e in events if e.get(key) is not None and math.isfinite(float(e[key]))]
    return {
        "p50": round(float(np.percentile(values, 50)), 3) if values else None,
        "p95": round(float(np.percentile(values, 95)), 3) if values else None,
    }


def same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return actual is not None and expected is not None and abs(float(actual) - float(expected)) <= 1e-9
    return actual == expected


def verify_metrics(report: dict[str, Any], request_events: list[dict[str, Any]], phase_metadata: dict[str, Any], errors: list[str]) -> None:
    metrics = report.get("metrics", {})
    successful = [e for e in request_events if e.get("success")]
    recomputed: dict[str, Any] = {
        "requests": len(request_events),
        "successful_requests": len(successful),
        "success_rate": len(successful) / len(request_events) if request_events else 0.0,
        "nan_count": sum(int(e.get("nan_count", 0)) for e in request_events),
        "inf_count": sum(int(e.get("inf_count", 0)) for e in request_events),
        "empty_audio_responses": sum(bool(e.get("empty_audio")) for e in request_events),
        "uncaught_exceptions": sum(len(e.get("exceptions", [])) for e in request_events),
        "generated_cb0_frames": sum(int(e.get("generated_cb0_frames", 0)) for e in request_events),
        "generated_pcm_samples": sum(int(e.get("generated_pcm_samples", 0)) for e in request_events),
        "cuda_peak_allocated_bytes": max((int(e.get("cuda_peak_allocated_bytes", 0)) for e in request_events), default=0),
        "cuda_peak_reserved_bytes": max((int(e.get("cuda_peak_reserved_bytes", 0)) for e in request_events), default=0),
        "think_time_ms": pstats(successful, "think_time_ms"),
        "v9_time_ms": pstats(successful, "v9_time_ms"),
        "talker_time_ms": pstats(successful, "talker_time_ms"),
        "ttfa_ms": pstats(successful, "ttfa_ms"),
        "e2e_rtf": pstats(successful, "e2e_rtf"),
        "speaker_rtf": pstats(successful, "speaker_rtf"),
    }
    for key, expected in recomputed.items():
        if key not in metrics:
            fail(errors, f"metrica obbligatoria mancante: {key}")
        elif not same(metrics[key], expected):
            fail(errors, f"metrica {key} non coincide col raw trace: {metrics[key]!r} != {expected!r}")
    if metrics.get("cpu_contract") != phase_metadata.get("cpu_contract"):
        fail(errors, "cpu_contract non coincide col record phase_metadata del raw trace")


P1_FSM_ACCEPTED = {
    "LISTENING": {"LISTEN", "SPEAK", "BACKCHANNEL", "HARD_STOP"},
    "SPEAKING": {"CONTINUE", "LISTEN", "YIELD", "REPLAN", "HARD_STOP"},
    "BACKCHANNELING": {"CONTINUE", "LISTEN", "YIELD", "REPLAN", "HARD_STOP"},
    "STOPPING": {"LISTEN"},
}
P1_FSM_NEXT = {
    "LISTENING": {"LISTEN": "LISTENING", "SPEAK": "SPEAKING", "BACKCHANNEL": "BACKCHANNELING", "HARD_STOP": "LISTENING"},
    "SPEAKING": {"CONTINUE": "SPEAKING", "LISTEN": "STOPPING", "YIELD": "STOPPING", "REPLAN": "STOPPING", "HARD_STOP": "STOPPING"},
    "BACKCHANNELING": {"CONTINUE": "BACKCHANNELING", "LISTEN": "LISTENING", "YIELD": "STOPPING", "REPLAN": "STOPPING", "HARD_STOP": "STOPPING"},
    "STOPPING": {"LISTEN": "LISTENING"},
}


def verify_p1(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Verify P1 from elemental events only; runner summaries are untrusted."""
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("phase") != "P1":
        fail(errors, "report non valido per P1")
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    verify_source_attestation(report, report_path, errors)
    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = report_path.parent / raw
    events: list[dict[str, Any]] = []
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        for line_no, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object")
                events.append(event)
            except Exception as exc:
                fail(errors, f"raw trace line {line_no} invalida: {exc}")
    verify_workspace_binding(report, events, report_path, errors)

    metadata_events = [e for e in events if e.get("event") == "phase_metadata"]
    if len(metadata_events) != 1:
        fail(errors, f"phase_metadata count={len(metadata_events)}")
    metadata_event = metadata_events[0] if metadata_events else {}
    if metadata_event.get("seed") != report.get("seed") or metadata_event.get("rng_seed_applied") is not True:
        fail(errors, "RNG seed P1 non attestato")

    cpu_events = [e for e in events if e.get("event") == "cpu_contract_result"]
    if len(cpu_events) != 1:
        fail(errors, f"cpu_contract_result count={len(cpu_events)}")
    cpu_event = cpu_events[0] if cpu_events else {}
    recomputed_cpu = {k: cpu_event.get(k) for k in ("command", "exit_code", "elapsed_s", "stdout_tail", "stderr_tail")}

    dependency = next((e for e in events if e.get("event") == "dependency_scan_result"), {})
    fsm = [e for e in events if e.get("event") == "fsm_transition_event"]
    fsm_sequences = [e for e in events if e.get("event") == "fsm_sequence_result"]
    unexpected_accepted = 0
    unexpected_rejected = 0
    for event in fsm:
        state = event.get("previous_state")
        action = event.get("action")
        expected = action in P1_FSM_ACCEPTED.get(state, set())
        expected_next = P1_FSM_NEXT.get(state, {}).get(action, state)
        actual = bool(event.get("accepted"))
        if actual != expected:
            if actual:
                unexpected_accepted += 1
            else:
                unexpected_rejected += 1
        if event.get("next_state") != expected_next:
            unexpected_accepted += 1
        if event.get("expected_accepted") != expected or event.get("expected_next_state") != expected_next:
            fail(errors, f"FSM event {event.get('sequence_id')} contiene aspettativa non indipendente")

    race_results = [e for e in events if e.get("event") == "race_result"]
    race_starts = {e.get("race_id") for e in events if e.get("event") == "race_start"}
    race_invalidations = {e.get("race_id") for e in events if e.get("event") == "race_invalidate"}
    race_b_starts = {e.get("race_id") for e in events if e.get("event") == "race_B_start"}
    stale_attempts = [e for e in events if e.get("event") == "stale_write_attempt" and e.get("actor") == "A"]
    stale_blocked = [e for e in events if e.get("event") == "stale_write_blocked" and e.get("actor") == "A"]
    stale_accepted = sum(bool(e.get("accepted")) for e in stale_attempts)
    stale_attempt_keys = {(e.get("race_id"), e.get("kind")) for e in stale_attempts}
    blocked_keys = {(e.get("race_id"), e.get("kind")) for e in stale_blocked}
    if stale_attempt_keys != blocked_keys:
        fail(errors, "non tutti i tentativi stale risultano bloccati nel trace")
    history_events = [e for e in events if e.get("event") == "history_commit"]
    history_emits = {(e.get("race_id"), e.get("actor")) for e in events if e.get("event") == "history_emit"}
    history_non_emitted_accepted = sum(
        bool(e.get("accepted")) and (e.get("race_id"), e.get("actor")) not in history_emits
        for e in history_events
    )
    policy_events = [e for e in events if e.get("event") == "policy_swap_check"]
    history_runtime_results = [e for e in events if e.get("event") == "history_runtime_result"]
    history_runtime_emits = [e for e in events if e.get("event") == "history_emit" and e.get("actor") == "partial_runtime"]
    history_runtime_commits = [e for e in events if e.get("event") == "history_commit" and e.get("actor") == "partial_runtime"]
    if any(e.get("event") == "history_scenario_result" for e in events):
        fail(errors, "history_scenario_result sintetico ancora presente")
    if len(history_runtime_results) != 1 or not history_runtime_results[0].get("passed"):
        fail(errors, "partial interruption history runtime result mancante/fallito")
    if len(history_runtime_emits) != 1 or len(history_runtime_commits) != 2:
        fail(errors, "trace history runtime parziale incompleto")
    else:
        runtime_emit = history_runtime_emits[0]
        first_commit = next((e for e in history_runtime_commits if not e.get("late")), {})
        late_commit = next((e for e in history_runtime_commits if e.get("late")), {})
        if not first_commit.get("accepted") or late_commit.get("accepted"):
            fail(errors, "commit emitted/late invalidate sequence non valida")
        if first_commit.get("committed_text") != runtime_emit.get("text"):
            fail(errors, "history runtime non coincide col testo emesso")
        if runtime_emit.get("text") != history_runtime_results[0].get("history_text"):
            fail(errors, "history runtime result non coincide con history effettiva")
        if history_runtime_results[0].get("planned_text") in (history_runtime_results[0].get("history_text") or ""):
            fail(errors, "planned non emesso presente nella history runtime")
    uncaught = len([e for e in events if e.get("event") in {"fsm_exception", "history_runtime_exception"}])
    recomputed = {
        "seed": metadata_event.get("seed"),
        "fsm_sequences": len({e.get("sequence_id") for e in fsm_sequences}),
        "unexpected_accepted_transitions": unexpected_accepted,
        "unexpected_rejected_transitions": unexpected_rejected,
        "illegal_state_transitions": unexpected_accepted + unexpected_rejected,
        "uncaught_exceptions": uncaught,
        "cancellation_races": len(race_results),
        "stale_write_attempts_blocked": len(stale_blocked),
        "stale_audio_events_emitted": stale_accepted,
        "stale_text_events_emitted": stale_accepted,
        "stale_history_writes": stale_accepted,
        "stale_kv_writes": stale_accepted,
        "stale_session_mutations": stale_accepted,
        "history_non_emitted_accepted": history_non_emitted_accepted,
        "history_runtime_partial_interruptions": len(history_runtime_results),
        "history_runtime_passed": bool(history_runtime_results and history_runtime_results[0].get("passed")),
        "direct_turnhead_imports": dependency.get("direct_turnhead_imports"),
        "direct_barge_in_server_branches": dependency.get("direct_barge_in_server_branches"),
        "policy_swap_checks": sum(bool(e.get("passed")) for e in policy_events),
        "cpu_contract": recomputed_cpu,
    }
    if not (race_starts == race_invalidations == race_b_starts == {e.get("race_id") for e in race_results}):
        fail(errors, "sequenza race A→invalidate→B incompleta")
    if any(not e.get("passed") for e in race_results):
        fail(errors, "race integration con risultato fallito")
    if len(stale_attempts) != len(stale_blocked) or stale_accepted:
        fail(errors, f"stale writes non completamente bloccati: attempts={len(stale_attempts)} blocked={len(stale_blocked)} accepted={stale_accepted}")
    if recomputed["fsm_sequences"] != 10_000 or recomputed["cancellation_races"] != 1_000:
        fail(errors, "conteggio FSM/race non conforme al gate P1")
    if len(fsm) < 10_000 or len(events) < 30_000:
        fail(errors, "raw trace P1 non granulare o incompleto")

    metrics = report.get("metrics", {})
    for key, expected in recomputed.items():
        if key == "cpu_contract":
            if metrics.get(key) != expected:
                fail(errors, "cpu_contract report non coincide col raw trace")
        elif metrics.get(key) != expected:
            fail(errors, f"metrica P1 {key} non coincide col raw trace: {metrics.get(key)!r} != {expected!r}")
    thresholds = report.get("thresholds", {})
    threshold_values = {"cpu_contract_exit_code": recomputed_cpu.get("exit_code"), **{
        key: recomputed.get(key) for key in thresholds if key != "cpu_contract_exit_code"
    }}
    for key, expected in thresholds.items():
        actual = threshold_values.get(key)
        if actual != expected:
            fail(errors, f"threshold P1 fallita: {key}={actual!r} != {expected!r}")
    independently_passed = not errors
    if report.get("passed") is not independently_passed:
        fail(errors, f"passed non coincide col risultato indipendente: {report.get('passed')} != {independently_passed}")
    return report, errors


def verify_p2(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Verify P2 from elemental scheduler events, never from runner summaries."""
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("phase") != "P2":
        fail(errors, "report non valido per P2")
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    verify_source_attestation(report, report_path, errors)
    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = report_path.parent / raw
    events: list[dict[str, Any]] = []
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        for line_no, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object")
                events.append(event)
            except Exception as exc:
                fail(errors, f"raw trace line {line_no} invalida: {exc}")
    verify_workspace_binding(report, events, report_path, errors)

    metadata_events = [e for e in events if e.get("event") == "phase_metadata"]
    if len(metadata_events) != 1:
        fail(errors, f"phase_metadata count={len(metadata_events)}")
    metadata_event = metadata_events[0] if metadata_events else {}
    if metadata_event.get("seed") != report.get("seed") or metadata_event.get("rng_seed_applied") is not True:
        fail(errors, "RNG seed P2 non attestato")
    if metadata_event.get("frame_duration_ms") != 80 or metadata_event.get("sample_rate") != 16_000:
        fail(errors, "contratto frame P2 non attestato")

    manifest = Path(report.get("input_manifest", ""))
    if not manifest.is_absolute():
        manifest = report_path.parent / manifest
    if not manifest.is_file():
        fail(errors, f"manifest mancante: {manifest}")
    elif sha256(manifest) != report.get("input_manifest_sha256"):
        fail(errors, "hash manifest non coincide")

    def sequence(clock: str, count: int) -> list[dict[str, Any]]:
        selected = [e for e in events if e.get("event") == "p2_tick" and e.get("clock") == clock]
        if len(selected) != count:
            fail(errors, f"tick {clock}: count={len(selected)} atteso={count}")
        ids = [e.get("frame_id") for e in selected]
        if ids != list(range(count)):
            fail(errors, f"tick {clock}: frame_id non sequenziale")
        for index, event in enumerate(selected):
            expected_start = index * 80_000_000 if clock.startswith("virtual") else event.get("capture_start_ns")
            if clock.startswith("virtual"):
                if event.get("capture_start_ns") != expected_start:
                    fail(errors, f"tick {clock} {index}: capture_start_ns non causale")
                if event.get("capture_end_ns") != expected_start + 80_000_000:
                    fail(errors, f"tick {clock} {index}: capture_end_ns non valido")
            if event.get("enqueue_ns") is None:
                fail(errors, f"tick {clock} {index}: frame ingress mancante")
        return selected

    virtual_10 = sequence("virtual_10s", 125)
    virtual_60 = sequence("virtual_60s", 750)
    real_run = [e for e in events if e.get("event") == "p2_real_run"]
    if len(real_run) != 1:
        fail(errors, f"p2_real_run count={len(real_run)}")
        real_config: dict[str, Any] = {}
    else:
        real_config = real_run[0]
    requested_duration = float(real_config.get("requested_duration_s", 0.0))
    expected_real = int(real_config.get("expected_ticks", 0))
    real_ticks = sequence("real", expected_real) if expected_real else []
    real_jitter = []
    for index, event in enumerate(real_ticks):
        deadline = event.get("deadline_ns")
        observed = event.get("observed_ns")
        if not isinstance(deadline, int) or not isinstance(observed, int):
            fail(errors, f"tick real {index}: deadline/observed mancanti")
            continue
        real_jitter.append(max(0, observed - deadline) / 1_000_000.0)
    jitter_p99 = round(float(np.percentile(real_jitter, 99)), 3) if real_jitter else None
    miss_rate = sum(value > 20.0 for value in real_jitter) / len(real_jitter) if real_jitter else None

    def transport(stream: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        arrivals = [e for e in events if e.get("event") == "p2_input_arrival" and e.get("stream") == stream]
        enqueued = [e for e in events if e.get("event") == "p2_input_enqueued" and e.get("stream") == stream]
        consumed = [e for e in events if e.get("event") == "p2_input_consumed" and e.get("stream") == stream]
        dropped = [e for e in events if e.get("event") == "p2_input_drop" and e.get("stream") == stream]
        if len(arrivals) != len(enqueued) + len(dropped):
            fail(errors, f"transport {stream}: arrivals != enqueued + dropped")
        return arrivals, enqueued, consumed, dropped

    dependency_events = [e for e in events if e.get("event") == "p2_dependency_scan"]
    if len(dependency_events) != 1:
        fail(errors, f"p2_dependency_scan count={len(dependency_events)}")
    dependency_event = dependency_events[0] if dependency_events else {}
    for key in (
        "direct_server_audio_ingress_consumers",
        "direct_ws_policy_calls",
    ):
        if not isinstance(dependency_event.get(key), int) or dependency_event.get(key) < 0:
            fail(errors, f"p2_dependency_scan campo non valido: {key}")
    for key in ("scheduler_policy_binding", "scheduler_start_binding"):
        if not isinstance(dependency_event.get(key), bool):
            fail(errors, f"p2_dependency_scan campo non valido: {key}")

    virtual_arrivals, virtual_enqueued, virtual_consumed, virtual_dropped = transport("virtual_60s")
    concurrent_arrivals, concurrent_enqueued, concurrent_consumed, concurrent_dropped = transport("concurrent")
    concurrent_speaker = [e for e in events if e.get("event") == "p2_concurrent_speaker_frame"]
    speaking = [e for e in events if e.get("event") == "p2_speaking_frame"]
    speaking_ids = [e.get("frame_id") for e in speaking]
    if len(speaking_ids) != len(set(speaking_ids)) or speaking_ids != list(range(750)):
        fail(errors, "frame input durante speaking non completo o non ordinato")
    if any(e.get("tick_frame_id") != e.get("frame_id") for e in speaking):
        fail(errors, "speaking frame non associato al tick corretto")
    if [e.get("frame_id") for e in virtual_arrivals] != list(range(750)):
        fail(errors, "arrivi virtuali P2 mancanti/duplicati/non ordinati")
    if [e.get("frame_id") for e in virtual_enqueued] != list(range(750)):
        fail(errors, "enqueue virtuali P2 mancanti/duplicati/non ordinati")
    if [e.get("frame_id") for e in virtual_consumed] != list(range(750)):
        fail(errors, "consume virtuali P2 mancanti/duplicati/non ordinati")
    if [e.get("frame_id") for e in concurrent_speaker] != [e.get("frame_id") for e in concurrent_consumed]:
        fail(errors, "consumer concurrent non sincronizzato con lo speaker")

    cpu_contract = metadata_event.get("cpu_contract", {})
    recomputed = {
        "cpu_contract": cpu_contract,
        "direct_server_audio_ingress_consumers": dependency_event.get("direct_server_audio_ingress_consumers"),
        "direct_ws_policy_calls": dependency_event.get("direct_ws_policy_calls"),
        "scheduler_policy_binding": dependency_event.get("scheduler_policy_binding"),
        "scheduler_start_binding": dependency_event.get("scheduler_start_binding"),
        "virtual_10_ticks": len(virtual_10),
        "virtual_10_missing_ticks": 0 if len(virtual_10) == 125 else 1,
        "virtual_10_duplicate_ticks": 0 if [e.get("frame_id") for e in virtual_10] == list(range(125)) else 1,
        "virtual_10_out_of_order_ticks": 0 if [e.get("frame_id") for e in virtual_10] == list(range(125)) else 1,
        "virtual_60_ticks": len(virtual_60),
        "virtual_60_missing_ticks": 0 if len(virtual_60) == 750 else 1,
        "virtual_60_duplicate_ticks": 0 if len(speaking_ids) == len(set(speaking_ids)) else 1,
        "virtual_60_out_of_order_ticks": 0 if [e.get("frame_id") for e in virtual_60] == list(range(750)) else 1,
        "input_frames_received": len(virtual_arrivals),
        "input_frames_enqueued": len(virtual_enqueued),
        "input_frames_consumed": len(virtual_consumed),
        "input_samples_received": sum(int(e.get("samples", 0)) for e in virtual_arrivals),
        "dropped_input_frames": len(virtual_dropped),
        "duplicate_input_frames": len(virtual_consumed) - len({e.get("frame_id") for e in virtual_consumed}),
        "out_of_order_input_frames": int([e.get("frame_id") for e in virtual_consumed] != sorted(e.get("frame_id") for e in virtual_consumed)),
        "input_frames_during_speaking": len(speaking_ids),
        "concurrent_input_frames_received": len(concurrent_arrivals),
        "concurrent_input_frames_enqueued": len(concurrent_enqueued),
        "concurrent_input_frames_consumed": len(concurrent_consumed),
        "concurrent_input_frames_during_speaking": sum(bool(e.get("speaker_active")) for e in concurrent_arrivals),
        "concurrent_dropped_input_frames": len(concurrent_dropped),
        "concurrent_duplicate_input_frames": len(concurrent_consumed) - len({e.get("frame_id") for e in concurrent_consumed}),
        "concurrent_out_of_order_input_frames": int([e.get("frame_id") for e in concurrent_consumed] != sorted(e.get("frame_id") for e in concurrent_consumed)),
        "concurrent_speaker_frames": len(concurrent_speaker),
        "real_ticks": len(real_ticks),
        "real_duration_s": round(float(real_config.get("elapsed_s", 0.0)), 3),
        "real_deadline_miss_rate": miss_rate,
        "real_tick_jitter_p99_ms": jitter_p99,
    }
    metrics = report.get("metrics", {})
    for key, expected in recomputed.items():
        if metrics.get(key) != expected:
            fail(errors, f"metrica P2 {key} non coincide col raw trace: {metrics.get(key)!r} != {expected!r}")
    if metrics.get("cpu_contract") != metadata_event.get("cpu_contract"):
        fail(errors, "cpu_contract P2 non coincide col raw trace")

    thresholds = report.get("thresholds", {})
    exact_thresholds = {
        "cpu_contract_exit_code": cpu_contract.get("exit_code"),
        "direct_server_audio_ingress_consumers": recomputed["direct_server_audio_ingress_consumers"],
        "direct_ws_policy_calls": recomputed["direct_ws_policy_calls"],
        "scheduler_policy_binding": recomputed["scheduler_policy_binding"],
        "scheduler_start_binding": recomputed["scheduler_start_binding"],
        "virtual_10_ticks": recomputed["virtual_10_ticks"],
        "virtual_10_missing_ticks": recomputed["virtual_10_missing_ticks"],
        "virtual_10_duplicate_ticks": recomputed["virtual_10_duplicate_ticks"],
        "virtual_10_out_of_order_ticks": recomputed["virtual_10_out_of_order_ticks"],
        "virtual_60_ticks": recomputed["virtual_60_ticks"],
        "virtual_60_missing_ticks": recomputed["virtual_60_missing_ticks"],
        "virtual_60_duplicate_ticks": recomputed["virtual_60_duplicate_ticks"],
        "virtual_60_out_of_order_ticks": recomputed["virtual_60_out_of_order_ticks"],
        "input_frames_received": recomputed["input_frames_received"],
        "input_frames_enqueued": recomputed["input_frames_enqueued"],
        "input_frames_consumed": recomputed["input_frames_consumed"],
        "input_samples_received": recomputed["input_samples_received"],
        "dropped_input_frames": recomputed["dropped_input_frames"],
        "duplicate_input_frames": recomputed["duplicate_input_frames"],
        "out_of_order_input_frames": recomputed["out_of_order_input_frames"],
        "concurrent_input_frames_received": recomputed["concurrent_input_frames_received"],
        "concurrent_input_frames_enqueued": recomputed["concurrent_input_frames_enqueued"],
        "concurrent_input_frames_consumed": recomputed["concurrent_input_frames_consumed"],
        "concurrent_dropped_input_frames": recomputed["concurrent_dropped_input_frames"],
        "concurrent_duplicate_input_frames": recomputed["concurrent_duplicate_input_frames"],
        "concurrent_out_of_order_input_frames": recomputed["concurrent_out_of_order_input_frames"],
        "concurrent_speaker_frames": recomputed["concurrent_speaker_frames"],
        "real_ticks": recomputed["real_ticks"],
    }
    for key, actual in exact_thresholds.items():
        if key not in thresholds:
            fail(errors, f"threshold P2 mancante: {key}")
        elif actual != thresholds[key]:
            fail(errors, f"threshold P2 fallita: {key}={actual!r} != {thresholds[key]!r}")
    if float(recomputed["input_frames_during_speaking"]) < float(thresholds.get("input_frames_during_speaking_min", 1)):
        fail(errors, "threshold P2 fallita: nessun input durante speaking")
    if float(recomputed["concurrent_input_frames_during_speaking"]) < float(thresholds.get("concurrent_input_frames_during_speaking_min", 1)):
        fail(errors, "threshold P2 fallita: producer concurrent non attivo durante speaking")
    if requested_duration < float(thresholds.get("real_duration_min_s", 60.0)):
        fail(errors, "threshold P2 fallita: real clock sotto 60 secondi")
    if float(recomputed["real_duration_s"]) < requested_duration:
        fail(errors, "real clock elapsed sotto la durata richiesta")
    if miss_rate is None or miss_rate > float(thresholds.get("real_deadline_miss_rate_max", 0.005)):
        fail(errors, "threshold P2 fallita: deadline_miss_rate")
    if jitter_p99 is None or jitter_p99 > float(thresholds.get("real_tick_jitter_p99_ms_max", 20.0)):
        fail(errors, "threshold P2 fallita: tick_jitter_p99")

    independently_passed = not errors
    if report.get("passed") is not independently_passed:
        fail(errors, f"passed non coincide col risultato indipendente: {report.get('passed')} != {independently_passed}")
    return report, errors


def _p3_source_scan(report: dict[str, Any], report_path: Path, errors: list[str]) -> dict[str, Any]:
    """Scan the attested historical source, not the verifier's current files."""
    try:
        attestation_path = Path(report["source_attestation"])
        if not attestation_path.is_absolute():
            attestation_path = report_path.parent / attestation_path
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        sha = attestation["git_sha"]
        repo = Path(__file__).resolve().parents[2]
        server = subprocess.check_output(("git", "show", f"{sha}:duplex_kernel/server.py"), cwd=repo, text=True)
        listener = subprocess.check_output(("git", "show", f"{sha}:duplex_kernel/listener.py"), cwd=repo, text=True)
    except Exception as exc:
        fail(errors, f"impossibile scansionare source P3 attestato: {exc}")
        return {}
    return {
        "event": "p3_dependency_scan",
        "direct_realtime_asr_calls": len(re.findall(r"engine\.transcribe_audio\s*\(", server)),
        "scheduler_listener_binding": "on_listener = scheduler_listener" in server,
        "reference_buffer_binding": "assistant_reference_buffer.visible_for" in server,
        "listener_step_binding": "sess.listener.step" in server,
        "native_audio_tower_backend": "get_audio_features" in listener and "transcribe_audio" not in listener,
        "assistant_reference_default_delay_frames": 1,
    }


def verify_p3(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Verify P3 from individual listener-frame events only."""
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("phase") != "P3":
        fail(errors, "report non valido per P3")
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    verify_source_attestation(report, report_path, errors)
    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = report_path.parent / raw
    events: list[dict[str, Any]] = []
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        for line_no, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object")
                events.append(event)
            except Exception as exc:
                fail(errors, f"raw trace line {line_no} invalida: {exc}")
    verify_workspace_binding(report, events, report_path, errors)

    metadata_events = [e for e in events if e.get("event") == "phase_metadata"]
    if len(metadata_events) != 1:
        fail(errors, f"phase_metadata count={len(metadata_events)}")
    metadata = metadata_events[0] if metadata_events else {}
    if metadata.get("seed") != report.get("seed") or metadata.get("rng_seed_applied") is not True:
        fail(errors, "RNG seed P3 non attestato")
    if metadata.get("sample_rate") != 16_000 or metadata.get("frame_duration_ms") != 80:
        fail(errors, "contratto frame P3 non attestato")
    if metadata.get("reference_delay_frames") != 1 or metadata.get("real_model_required") is not True:
        fail(errors, "contratto self-reference/model reale P3 non attestato")

    cpu_events = [e for e in events if e.get("event") == "cpu_contract_result"]
    if len(cpu_events) != 1:
        fail(errors, f"cpu_contract_result count={len(cpu_events)}")
    cpu = cpu_events[0] if cpu_events else {}
    cpu_contract = {key: cpu.get(key) for key in ("command", "exit_code", "elapsed_s", "stdout_tail", "stderr_tail")}

    model_events = [e for e in events if e.get("event") == "p3_model_metadata"]
    if len(model_events) != 1:
        fail(errors, f"p3_model_metadata count={len(model_events)}")
    model = model_events[0] if model_events else {}
    if model.get("real_model") is not True or model.get("backend") != "GemmaAudioTowerBackend":
        fail(errors, "P3 non attestata su GemmaAudioTowerBackend reale")

    dependency_events = [e for e in events if e.get("event") == "p3_dependency_scan"]
    if len(dependency_events) != 1:
        fail(errors, f"p3_dependency_scan count={len(dependency_events)}")
    dependency = dependency_events[0] if dependency_events else {}
    expected_dependency = _p3_source_scan(report, report_path, errors)
    for key, expected in expected_dependency.items():
        if dependency.get(key) != expected:
            fail(errors, f"dependency P3 {key} non indipendente: {dependency.get(key)!r} != {expected!r}")

    frame_events = [e for e in events if e.get("event") == "p3_listener_frame"]
    error_events = [e for e in events if e.get("event") == "p3_listener_error"]
    attempts = sorted(frame_events + error_events, key=lambda event: int(event.get("frame_id", -1)))
    attempt_ids = [e.get("frame_id") for e in attempts]
    if attempt_ids != list(range(len(attempts))):
        fail(errors, "P3 listener frame_id mancanti/duplicati/non ordinati")
    if any(e.get("samples") != 1280 for e in attempts):
        fail(errors, "P3 evento listener con sample count diverso da 1280")
    if len(attempts) < 10_000:
        fail(errors, f"P3 listener updates insufficienti: {len(attempts)}")

    successful = frame_events
    latencies = [float(e["listener_step_gpu_ms"]) for e in successful
                 if e.get("listener_step_gpu_ms") is not None and math.isfinite(float(e["listener_step_gpu_ms"]))]
    listener_pstats = {
        "p50": round(float(np.percentile(latencies, 50)), 3) if latencies else None,
        "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
        "max": round(max(latencies), 3) if latencies else None,
    }
    speaking_expected = sum(bool(e.get("speaker_active")) for e in attempts)
    speaking_updates = sum(bool(e.get("speaker_active")) for e in successful)
    future_leaks = 0
    wrong_references = 0
    for event in successful:
        frame_id = int(event["frame_id"])
        expected_reference = frame_id - 1 if frame_id > 0 else None
        actual_reference = event.get("reference_frame_id")
        if isinstance(actual_reference, int) and actual_reference > frame_id - 1:
            future_leaks += 1
        if actual_reference != expected_reference:
            wrong_references += 1
        if event.get("expected_reference_frame_id") != expected_reference:
            fail(errors, f"P3 aspettativa reference non causale al frame {frame_id}")
    replay_frames = min(len(attempts), 3_750)
    recomputed = {
        "listener_input_frames": len(successful),
        "samples_received": sum(int(e.get("samples", 0)) for e in successful),
        "samples_accounted_for": len(successful) * 1280,
        "dropped_listener_frames": len(error_events),
        "replay_frames": replay_frames,
        "replay_seconds": replay_frames * 80 / 1000.0,
        "input_frames_while_speaking": speaking_expected,
        "listener_updates_while_speaking": speaking_updates,
        "listener_coverage_while_speaking": speaking_updates / speaking_expected if speaking_expected else 0.0,
        "nonfinite_hidden_count": sum(not bool(e.get("finite")) for e in successful),
        "empty_hidden_count": sum(int(e.get("hidden_numel", 0)) <= 0 for e in successful),
        "future_reference_leaks": future_leaks,
        "wrong_reference_frames": wrong_references,
        "assistant_reference_delay_frames": dependency.get("assistant_reference_default_delay_frames"),
        "listener_step_gpu_ms": listener_pstats,
        "listener_errors": len(error_events),
        "p3_dependency_scan": expected_dependency,
        "cpu_contract": cpu_contract,
    }
    metrics = report.get("metrics", {})
    for key, expected in recomputed.items():
        if metrics.get(key) != expected:
            fail(errors, f"metrica P3 {key} non coincide col raw trace: {metrics.get(key)!r} != {expected!r}")

    thresholds = report.get("thresholds", {})
    exact = {
        "cpu_contract_exit_code": cpu.get("exit_code"),
        "dropped_listener_frames": recomputed["dropped_listener_frames"],
        "nonfinite_hidden_count": recomputed["nonfinite_hidden_count"],
        "empty_hidden_count": recomputed["empty_hidden_count"],
        "future_reference_leaks": recomputed["future_reference_leaks"],
        "wrong_reference_frames": recomputed["wrong_reference_frames"],
        "listener_errors": recomputed["listener_errors"],
        "assistant_reference_delay_frames": recomputed["assistant_reference_delay_frames"],
    }
    for key, actual in exact.items():
        if key not in thresholds:
            fail(errors, f"threshold P3 mancante: {key}")
        elif actual != thresholds[key]:
            fail(errors, f"threshold P3 fallita: {key}={actual!r} != {thresholds[key]!r}")
    for key, actual in {
        "listener_input_frames_min": recomputed["listener_input_frames"],
        "replay_frames_min": recomputed["replay_frames"],
        "input_frames_while_speaking_min": recomputed["input_frames_while_speaking"],
    }.items():
        if actual < thresholds.get(key, float("inf")):
            fail(errors, f"threshold P3 fallita: {key}={actual}")
    if "samples_received_equals_accounted" not in thresholds or thresholds.get("samples_received_equals_accounted") is not True:
        fail(errors, "threshold P3 mancante/non valida: samples_received_equals_accounted")
    if recomputed["samples_received"] != recomputed["samples_accounted_for"]:
        fail(errors, "threshold P3 fallita: samples non contabilizzati")
    if recomputed["replay_seconds"] < thresholds.get("replay_seconds_min", float("inf")):
        fail(errors, "threshold P3 fallita: replay sotto cinque minuti")
    if recomputed["listener_coverage_while_speaking"] < thresholds.get("listener_coverage_while_speaking", float("inf")):
        fail(errors, "threshold P3 fallita: coverage listener durante speaking")
    p95 = recomputed["listener_step_gpu_ms"].get("p95")
    if p95 is None or p95 > thresholds.get("listener_step_gpu_ms_p95_max", float("inf")):
        fail(errors, "threshold P3 fallita: listener step p95")
    if metrics.get("samples_received") != metrics.get("samples_accounted_for"):
        fail(errors, "report P3 dichiara samples non contabilizzati")

    independently_passed = not errors
    if report.get("passed") is not independently_passed:
        fail(errors, f"passed non coincide col risultato indipendente: {report.get('passed')} != {independently_passed}")
    return report, errors


def _close(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def verify_p4(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Verify P4 metrics from granular V9/Qwen frame events."""
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("phase") != "P4":
        fail(errors, "report non valido per P4")
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    verify_source_attestation(report, report_path, errors)
    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = report_path.parent / raw
    events: list[dict[str, Any]] = []
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        for line_no, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object")
                events.append(event)
            except Exception as exc:
                fail(errors, f"raw trace line {line_no} invalida: {exc}")
    verify_workspace_binding(report, events, report_path, errors)
    metadata_events = [event for event in events if event.get("event") == "phase_metadata"]
    if len(metadata_events) != 1:
        fail(errors, f"phase_metadata count={len(metadata_events)}")
    metadata = metadata_events[0] if metadata_events else {}
    if metadata.get("seed") != report.get("seed") or metadata.get("rng_seed_applied") is not True:
        fail(errors, "RNG seed P4 non attestato")
    if metadata.get("real_model_required") is not True or metadata.get("max_seq_len") != 16_384:
        fail(errors, "contratto P4 modello reale/max_seq_len non attestato")
    model_events = [event for event in events if event.get("event") == "p4_model_metadata"]
    if len(model_events) != 1 or model_events[0].get("real_model") is not True:
        fail(errors, "modello V9 reale P4 mancante")
    elif model_events[0].get("future_prefix_dtype") != "float32":
        fail(errors, "P4-T2 non attestato in FP32")
    qwen_events = [event for event in events if event.get("event") == "p4_qwen_model_metadata"]
    if len(qwen_events) != 1 or qwen_events[0].get("real_model") is not True:
        fail(errors, "modello Qwen reale P4 mancante")
    if any(event.get("event") == "p4_qwen_load_error" for event in events):
        fail(errors, "caricamento Qwen P4 fallito")
    cpu_events = [event for event in events if event.get("event") == "cpu_contract_result"]
    if len(cpu_events) != 1:
        fail(errors, f"cpu_contract_result count={len(cpu_events)}")
    cpu = cpu_events[0] if cpu_events else {}
    cpu_contract = {key: cpu.get(key) for key in ("command", "exit_code", "elapsed_s", "stdout_tail", "stderr_tail")}

    equivalence_frames = [event for event in events if event.get("event") == "p4_equivalence_frame"]
    equivalence_sequences = [event for event in events if event.get("event") == "p4_equivalence_sequence"]
    sequence_ids = sorted({event.get("sequence_id") for event in equivalence_frames})
    if sequence_ids != list(range(len(sequence_ids))):
        fail(errors, "P4 equivalence sequence_id non contigui")
    for sequence_id in sequence_ids:
        frames = sorted(event.get("frame_id") for event in equivalence_frames if event.get("sequence_id") == sequence_id)
        if frames != list(range(len(frames))):
            fail(errors, f"P4 equivalence frame_id non contigui nella sequenza {sequence_id}")
    if len(equivalence_sequences) != len(sequence_ids):
        fail(errors, "P4 equivalence sequence/frame count incoerente")
    eq_by_sequence = sorted(equivalence_sequences, key=lambda event: int(event["sequence_id"]))
    eq_mae = [float(event["mae"]) for event in eq_by_sequence]
    eq_cosine = [float(event["cosine_min"]) for event in eq_by_sequence]
    eq_recomputed = {
        "sequences": len(eq_by_sequence),
        "frames": len(equivalence_frames),
        "mae": sum(eq_mae) / len(eq_mae) if eq_mae else None,
        "mae_max_sequence": max(eq_mae) if eq_mae else None,
        "cosine_min": min(eq_cosine) if eq_cosine else None,
        "finite": all(bool(event.get("finite")) for event in equivalence_frames + eq_by_sequence),
    }

    future_events = sorted(
        [event for event in events if event.get("event") == "p4_future_prefix_sequence"],
        key=lambda event: int(event["sequence_id"]),
    )
    for event in future_events:
        expected_max = max(float(event["step_max_difference"]), float(event["batch_max_difference"]))
        if not _close(event.get("max_future_prefix_difference"), expected_max):
            fail(errors, f"P4 future-prefix max non coincide nella sequenza {event.get('sequence_id')}")
    future_recomputed = {
        "sequences": len(future_events),
        "prefix_frames": int(future_events[0]["prefix_frames"]) if future_events else None,
        "max_difference": max((float(event["max_future_prefix_difference"]) for event in future_events), default=None),
        "mean_difference": (
            sum(float(event["max_future_prefix_difference"]) for event in future_events) / len(future_events)
            if future_events else None
        ),
        "finite": all(bool(event.get("finite")) for event in future_events),
    }

    timing_events = sorted(
        [event for event in events if event.get("event") == "p4_v9_step_timing"],
        key=lambda event: int(event["step_id"]),
    )
    timing_ids = [event.get("step_id") for event in timing_events]
    if timing_ids != list(range(len(timing_events))):
        fail(errors, "P4 timing step_id mancanti/duplicati")
    latencies = np.asarray([float(event["latency_ms"]) for event in timing_events], dtype=np.float64)
    performance = {
        "steps": len(timing_events),
        "warmup_steps": int(metadata.get("warmup_steps", report.get("metrics", {}).get("performance", {}).get("warmup_steps", 100))),
        "p50_ms": float(np.quantile(latencies, 0.50)) if len(latencies) else None,
        "p95_ms": float(np.quantile(latencies, 0.95)) if len(latencies) else None,
        "max_ms": float(latencies.max()) if len(latencies) else None,
        "finite": bool(np.isfinite(latencies).all()) if len(latencies) else False,
    }

    ablation_frames = [event for event in events if event.get("event") == "p4_qwen_ablation_frame"]
    ablation_sequence_ids = sorted({event.get("sequence_id") for event in ablation_frames})
    if ablation_sequence_ids != list(range(len(ablation_sequence_ids))):
        fail(errors, "P4 Qwen sequence_id non contigui")
    for sequence_id in ablation_sequence_ids:
        frame_ids = sorted(event.get("frame_id") for event in ablation_frames if event.get("sequence_id") == sequence_id)
        if frame_ids != list(range(len(frame_ids))):
            fail(errors, f"P4 Qwen frame_id non contigui nella sequenza {sequence_id}")
    full_ce = [float(event["full_ce"]) for event in ablation_frames]
    causal_ce = [float(event["causal_ce"]) for event in ablation_frames]
    full_top1 = [bool(event["full_top1"]) for event in ablation_frames]
    causal_top1 = [bool(event["causal_top1"]) for event in ablation_frames]
    full_top5 = [bool(event["full_top5"]) for event in ablation_frames]
    causal_top5 = [bool(event["causal_top5"]) for event in ablation_frames]
    kl = [float(event["kl_full_to_causal"]) for event in ablation_frames]
    deltas = [float(event["max_logit_delta"]) for event in ablation_frames]
    ablation_recomputed: dict[str, Any] = {
        "sequences": len({event.get("sequence_id") for event in ablation_frames}),
        "frames": len(ablation_frames),
        "full_ce": sum(full_ce) / len(full_ce) if full_ce else None,
        "causal_ce": sum(causal_ce) / len(causal_ce) if causal_ce else None,
        "delta_ce": (sum(causal_ce) - sum(full_ce)) / len(full_ce) if full_ce else None,
        "full_top1": sum(full_top1) / len(full_top1) if full_top1 else None,
        "causal_top1": sum(causal_top1) / len(causal_top1) if causal_top1 else None,
        "full_top5": sum(full_top5) / len(full_top5) if full_top5 else None,
        "causal_top5": sum(causal_top5) / len(causal_top5) if causal_top5 else None,
        "kl_full_to_causal": sum(kl) / len(kl) if kl else None,
        "max_logit_delta": max(deltas) if deltas else None,
        "quartile_delta_ce": {},
        "finite": all(math.isfinite(value) for values in (full_ce, causal_ce, kl, deltas) for value in values),
        "mask_future_prefix": True,
    }
    if ablation_frames:
        frame_count = max(int(event["frame_id"]) for event in ablation_frames) + 1
        for bucket in range(4):
            values = [
                float(event["causal_ce"]) - float(event["full_ce"])
                for event in ablation_frames
                if min(3, int(int(event["frame_id"]) * 4 / frame_count)) == bucket
            ]
            if values:
                ablation_recomputed["quartile_delta_ce"][f"q{bucket}"] = sum(values) / len(values)
    ablation_recomputed["passed"] = bool(
        ablation_recomputed["frames"] >= 5_000
        and ablation_recomputed["finite"]
        and ablation_recomputed["mask_future_prefix"]
        and (ablation_recomputed["max_logit_delta"] or 0.0) > 1e-6
    )

    metrics = report.get("metrics", {})
    expected_sections = {
        "equivalence": eq_recomputed,
        "future_prefix": future_recomputed,
        "performance": performance,
        "qwen_ablation": ablation_recomputed,
    }
    for section, expected in expected_sections.items():
        actual = metrics.get(section, {})
        for key, value in expected.items():
            if key == "passed":
                continue
            if key == "quartile_delta_ce":
                actual_quartiles = actual.get(key, {})
                for quartile, quartile_value in value.items():
                    if not _close(actual_quartiles.get(quartile), quartile_value):
                        fail(errors, f"metrica P4 {section}.{key}.{quartile} non coincide col raw trace")
            elif not _close(actual.get(key), value):
                fail(errors, f"metrica P4 {section}.{key} non coincide col raw trace: {actual.get(key)!r} != {value!r}")
    if metrics.get("cpu_contract") != cpu_contract:
        fail(errors, "cpu_contract P4 non coincide col raw trace")
    thresholds = report.get("thresholds", {})
    checks = {
        "cpu_contract_exit_code": cpu.get("exit_code"),
        "equivalence_sequences_min": eq_recomputed["sequences"],
        "equivalence_frames_min": eq_recomputed["frames"],
        "equivalence_mae_max": eq_recomputed["mae"],
        "equivalence_cosine_min": eq_recomputed["cosine_min"],
        "future_prefix_sequences_min": future_recomputed["sequences"],
        "future_prefix_max_difference": future_recomputed["max_difference"],
        "performance_steps_min": performance["steps"],
        "v9_step_p95_ms_max": performance["p95_ms"],
        "ablation_sequences_min": ablation_recomputed["sequences"],
        "ablation_frames_min": ablation_recomputed["frames"],
    }
    if thresholds.get("qwen_ablation_required") is not True:
        fail(errors, "threshold P4 qwen_ablation_required mancante")
    if not ablation_recomputed["passed"]:
        fail(errors, "P4 Qwen ablation fallita: criteri di delta o finitezza non soddisfatti")
    minimums = {
        "equivalence_sequences_min": (lambda value, limit: value >= limit),
        "equivalence_frames_min": (lambda value, limit: value >= limit),
        "equivalence_mae_max": (lambda value, limit: value <= limit),
        "equivalence_cosine_min": (lambda value, limit: value >= limit),
        "future_prefix_sequences_min": (lambda value, limit: value >= limit),
        "future_prefix_max_difference": (lambda value, limit: value <= limit),
        "performance_steps_min": (lambda value, limit: value >= limit),
        "v9_step_p95_ms_max": (lambda value, limit: value <= limit),
        "ablation_sequences_min": (lambda value, limit: value >= limit),
        "ablation_frames_min": (lambda value, limit: value >= limit),
    }
    for key, actual in checks.items():
        if key == "cpu_contract_exit_code":
            continue
        limit = thresholds.get(key)
        if actual is None or limit is None or not minimums[key](actual, limit):
            fail(errors, f"threshold P4 fallita: {key}={actual!r}, limite={limit!r}")
    if thresholds.get("cpu_contract_exit_code") != checks["cpu_contract_exit_code"]:
        fail(errors, "threshold P4 cpu_contract_exit_code fallita")
    independently_passed = not errors
    if report.get("passed") is not independently_passed:
        fail(errors, f"passed non coincide col risultato indipendente: {report.get('passed')} != {independently_passed}")
    return report, errors


def verify_one(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        fail(errors, "schema_version non supportata")
    if report.get("phase") == "P2":
        return verify_p2(report_path)
    if report.get("phase") == "P3":
        return verify_p3(report_path)
    if report.get("phase") == "P4":
        return verify_p4(report_path)
    verify_source_attestation(report, report_path, errors)
    if report.get("phase") == "P1":
        return verify_p1(report_path)
    if report.get("phase") != "P0":
        fail(errors, "verify_report supporta P0, P1, P2 e P3")
    if report.get("git_dirty") is not False:
        fail(errors, "git_dirty deve essere false")
    raw = Path(report.get("raw_trace", ""))
    if not raw.is_absolute():
        raw = report_path.parent / raw
    events: list[dict[str, Any]] = []
    if not raw.is_file():
        fail(errors, f"raw trace mancante: {raw}")
    else:
        if sha256(raw) != report.get("raw_trace_sha256"):
            fail(errors, "hash raw trace non coincide")
        for line_no, line in enumerate(raw.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object")
                events.append(event)
            except Exception as exc:
                fail(errors, f"raw trace line {line_no} invalida: {exc}")
    verify_workspace_binding(report, events, report_path, errors)
    manifest = Path(report.get("input_manifest", ""))
    if not manifest.is_absolute():
        manifest = report_path.parent / manifest
    if not manifest.is_file():
        fail(errors, f"manifest mancante: {manifest}")
    elif sha256(manifest) != report.get("input_manifest_sha256"):
        fail(errors, "hash manifest non coincide")

    request_events = [e for e in events if e.get("event") == "p0_request"]
    phase_metadata = next((e for e in events if e.get("event") == "phase_metadata"), {})
    if phase_metadata.get("seed") != report.get("seed") or phase_metadata.get("rng_seed_applied") is not True:
        fail(errors, "RNG seed non attestato nel raw trace")
    expected_seeds = [int(report.get("seed", 0)) + i for i in range(len(request_events))]
    actual_seeds = [e.get("rng_seed") for e in request_events]
    if actual_seeds != expected_seeds:
        fail(errors, "sequenza RNG per richiesta non coincide col seed dichiarato")
    if len(request_events) != len(json.loads(manifest.read_text(encoding="utf-8")).get("inputs", [])) if manifest.is_file() else False:
        fail(errors, "numero richieste nel raw trace diverso dal manifest")
    if events:
        verify_metrics(report, request_events, phase_metadata, errors)

    metrics = report.get("metrics", {})
    thresholds = report.get("thresholds", {})
    checks = {
        "cpu_contract_exit_code": metrics.get("cpu_contract", {}).get("exit_code"),
        "success_rate": metrics.get("success_rate"),
        "nan_count": metrics.get("nan_count"),
        "inf_count": metrics.get("inf_count"),
        "empty_audio_responses": metrics.get("empty_audio_responses"),
        "uncaught_exceptions": metrics.get("uncaught_exceptions"),
    }
    for key, actual in checks.items():
        expected = thresholds.get(key)
        if actual is None or expected is None:
            fail(errors, f"threshold non verificabile: {key}")
        elif key == "success_rate":
            if float(actual) < float(expected):
                fail(errors, f"threshold fallita: {key}={actual} < {expected}")
        elif actual != expected:
            fail(errors, f"threshold fallita: {key}={actual} != {expected}")

    independently_passed = not errors
    if report.get("passed") is not independently_passed:
        fail(errors, f"passed non coincide col risultato indipendente: {report.get('passed')} != {independently_passed}")
    return report, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--compare", type=Path, default=None, help="secondo report per il gate di riproducibilità")
    args = ap.parse_args()
    report, errors = verify_one(args.report)
    if args.compare is not None:
        other, other_errors = verify_one(args.compare)
        errors.extend(f"compare: {e}" for e in other_errors)
        for key in ("git_sha", "git_dirty", "input_manifest_sha256", "gpu_uuid",
                    "torch_version", "cuda_version", "transformers_version"):
            if report.get(key) != other.get(key):
                fail(errors, f"riproducibilità fallita: {key} differente")
        if report.get("phase") != other.get("phase") or report.get("phase") != "P0":
            fail(errors, "i report di confronto non sono entrambi P0")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
