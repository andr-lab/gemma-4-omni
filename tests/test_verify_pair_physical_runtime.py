from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.verify_pair_physical_runtime as physical
from tools.verify_pair_physical_runtime import (
    _drain_gate,
    _intersect_interval_unions,
    _merge_intervals,
    _native_talker_gate,
    _native_decision_gate,
    verify_physical_gates,
    verify_capture,
)


def _scored_decision(frame_id: int, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "type": "duplex_control6.decision",
        "speaker": "A",
        "frame_id": frame_id,
        "accepted": True,
        "rejection_reason": None,
        "at_ms": 120.0,
        "native_control6_head_scored": True,
        "native_control_hidden_supplied": True,
        "native_control_hidden_frame_id": frame_id,
        "listener_hidden_age_frames": 0,
        "listener_hidden_age_ms": 0.5,
        "listener_query_reused": False,
        "listener_query_age_frames": None,
    }
    row.update(updates)
    return row


def _completed_response(response_id: str, *, audio_samples: int = 100, terminal_event: str = "response.done", talker_terminal: str = "eos", thinker_terminal: str = "eos") -> dict[str, object]:
    return {
        "response_id": response_id,
        "terminal_event": terminal_event,
        "audio_samples": audio_samples,
        "runtime_metrics": {
            "generation_trace": {"terminal_reason": thinker_terminal},
            "talker": {"terminated_by": talker_terminal},
        },
    }


def _clock_attestation_manifest(
    root: Path, server_argv: list[str], client_argv: list[str] | None = None,
) -> dict[str, object]:
    runner_path = root / "tools" / "run_pair_duplex_proof_3090.py"
    runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    return {
        "container": {"id": "same-task-container"},
        "argv": {
            "server": server_argv,
            "client": client_argv or ["/workspace/.venv_e2e_runtime/bin/python", "client/duplex_stream_e2e.py"],
            "docker_run": ["docker", "run", "--name", "pair"],
        },
        "source_files": [
            {"path": "tools/run_pair_duplex_proof_3090.py", "sha256": runner_sha},
            {"path": "client/duplex_stream_e2e.py", "sha256": "a" * 64},
            {"path": "duplex_kernel/server.py", "sha256": "b" * 64},
        ],
        "git": {"status_sha256": "c" * 64},
    }


def _drain_fixture(target: int, responses: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    audible = sum(
        1 for row in responses
        if row.get("terminal_event") == "response.done"
        and int(row.get("audio_samples", 0)) > 0
        and row.get("runtime_metrics", {}).get("generation_trace", {}).get("terminal_reason") == "eos"
        and row.get("runtime_metrics", {}).get("talker", {}).get("terminated_by") == "eos"
    )
    report: dict[str, object] = {
        "pair_completion_status": "complete",
        "pair_drain_completed": True,
        "created_responses": len(responses),
        "audible_completed_responses": audible,
        "target_audible_responses": target,
        "responses": responses,
        "participants": [
            {"speaker": speaker, "input_frames_sent": 20, "input_frames_acked": 20, "pending_acks_at_close": 0, "missing_ack_count": 0}
            for speaker in ("A", "B")
        ],
    }
    snapshot = {
        speaker: {
            "active_response_ids": [],
            "peer_audio_queue_frames": 0,
            "route_pending_items": 0,
            "unsent_pcm_frames": 0,
            "pending_ack_frames": 0,
            "enqueued_pcm_frames": count,
            "sent_pcm_frames": count,
        }
        for speaker, count in (("A", 206), ("B", 189))
    }
    drain: dict[str, object] = {
        "type": "probe.response_drain_completed",
        "created_responses": len(responses),
        "audible_completed_responses": audible,
        "target_audible_responses": target,
        "pipeline_snapshot": snapshot,
    }
    return report, drain


class PhysicalRuntimeGateTests(unittest.TestCase):
    def test_query_age_is_required_only_when_cached_query_was_reused(self) -> None:
        events = [
            {"type": "probe.audible_response_target_reached", "at_ms": 100.0},
            _scored_decision(1),
            _scored_decision(
                2,
                accepted=False,
                rejection_reason="auto_response_disabled",
                at_ms=101.0,
            ),
            {
                "type": "duplex_control6.decision",
                "speaker": "B",
                "frame_id": 3,
                "accepted": True,
                "native_control6_head_scored": False,
                "source": "scheduler_verified_exact_zero_user_and_self_pcm",
            },
        ]
        gate, indexed = _native_decision_gate(events)
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["accepted_head_scored_count"], 1)
        self.assertEqual(gate["expected_post_target_auto_response_disabled_rejections"], 1)
        self.assertEqual(gate["scheduler_fallback_count"], 1)
        self.assertEqual(set(indexed), {("A", 1), ("A", 2)})

    def test_reused_query_and_stale_hidden_state_fail_closed(self) -> None:
        base = {"type": "probe.audible_response_target_reached", "at_ms": 100.0}
        for update, expected_status in (
            ({"listener_query_reused": True, "listener_query_age_frames": None}, "INCOMPLETE"),
            ({"listener_query_reused": True, "listener_query_age_frames": 2}, "FAIL"),
            ({"native_control_hidden_frame_id": 7}, "FAIL"),
            ({"listener_hidden_age_frames": -1}, "FAIL"),
        ):
            with self.subTest(update=update):
                gate, _ = _native_decision_gate([base, _scored_decision(1, **update)])
                self.assertEqual(gate["status"], expected_status)

    def test_snapshot_rejection_before_target_is_not_hidden_as_drain_rejection(self) -> None:
        events = [
            {"type": "probe.audible_response_target_reached", "at_ms": 100.0},
            _scored_decision(
                1,
                accepted=False,
                rejection_reason="direct_audio_snapshot_invalid:pcm=0",
                at_ms=99.0,
            ),
        ]
        gate, _ = _native_decision_gate(events)
        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(any("unexpected native decision rejection" in item for item in gate["failures"]))

    def test_interval_union_prevents_duplicate_frame_pair_counting(self) -> None:
        left = _merge_intervals([(0, 80), (40, 120), (200, 240)])
        right = _merge_intervals([(20, 100), (50, 140), (230, 260)])
        overlap = _intersect_interval_unions(left, right)
        self.assertEqual(overlap, [(20, 120), (230, 240)])
        self.assertEqual(sum(end - start for start, end in overlap), 110)

    def test_drain_target_is_minimum_and_snapshot_counts_must_match(self) -> None:
        report = {
            "pair_completion_status": "complete",
            "pair_drain_completed": True,
            "created_responses": 5,
            "audible_completed_responses": 5,
            "target_audible_responses": 4,
            "responses": [_completed_response(f"r{i}") for i in range(5)],
            "participants": [
                {"speaker": speaker, "input_frames_sent": 20, "input_frames_acked": 20, "pending_acks_at_close": 0, "missing_ack_count": 0}
                for speaker in ("A", "B")
            ],
        }
        snapshot = {
            speaker: {
                "active_response_ids": [],
                "peer_audio_queue_frames": 0,
                "route_pending_items": 0,
                "unsent_pcm_frames": 0,
                "pending_ack_frames": 0,
                "enqueued_pcm_frames": count,
                "sent_pcm_frames": count,
            }
            for speaker, count in (("A", 206), ("B", 189))
        }
        drain = {
            "type": "probe.response_drain_completed",
            "created_responses": 5,
            "audible_completed_responses": 5,
            "target_audible_responses": 4,
            "pipeline_snapshot": snapshot,
        }
        self.assertEqual(_drain_gate(report, [drain], report["participants"])["status"], "PASS")

        snapshot["A"]["pending_ack_frames"] = 1
        failed = _drain_gate(report, [drain], report["participants"])
        self.assertEqual(failed["status"], "FAIL")
        self.assertTrue(any("pending_ack_frames=1" in item for item in failed["failures"]))
        self.assertEqual(_drain_gate(report, [], report["participants"])["status"], "INCOMPLETE")

    def test_drain_requires_reported_target_of_at_least_four(self) -> None:
        report, drain = _drain_fixture(3, [_completed_response(f"r{i}") for i in range(3)])
        gate = _drain_gate(report, [drain], report["participants"])
        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(any("minimum target" in item for item in gate["failures"]))

    def test_drain_counts_only_done_audible_natural_eos_and_ignores_partial_audio(self) -> None:
        responses = [_completed_response(f"done{i}") for i in range(4)]
        responses.append(_completed_response("cancelled", terminal_event="response.cancelled", talker_terminal="budget"))
        report, drain = _drain_fixture(4, responses)
        self.assertEqual(_drain_gate(report, [drain], report["participants"])["status"], "PASS")

        responses[0] = _completed_response("budget-ended", talker_terminal="budget")
        report, drain = _drain_fixture(4, responses)
        gate = _drain_gate(report, [drain], report["participants"])
        self.assertEqual(gate["status"], "FAIL")
        self.assertEqual(gate["natural_eos_audible_response_count"], 3)

    def test_nested_latency_failure_propagates_to_overall_status(self) -> None:
        with (
            patch.object(physical, "attest_shared_monotonic_clock", return_value={"attested": True}),
            patch.object(physical, "_native_decision_gate", return_value=({"status": "PASS"}, {})),
            patch.object(physical, "_latency_gates", return_value={"append_to_ack": {"status": "FAIL"}}),
            patch.object(physical, "_overlap_gate", return_value={"status": "PASS"}),
            patch.object(physical, "_drain_gate", return_value={"status": "PASS"}),
            patch.object(physical, "_native_talker_gate", return_value={"status": "PASS"}),
        ):
            result = verify_physical_gates({}, [], {}, Path("."))
        self.assertEqual(result["status"], "FAIL")

    def test_latency_parent_status_preserves_failure_when_other_evidence_is_missing(self) -> None:
        events = []
        for speaker in ("A", "B"):
            events.append({
                "type": "probe.input_frame_sent",
                "speaker": speaker,
                "frame_id": 1,
                "send_started_at_ns": 0,
                "sent_at_ns": 1,
                "at_ms": 0.0,
            })
            events.append({
                "type": "probe.input_frame_ack",
                "speaker": speaker,
                "client_frame_id": 1,
                "ack_frame_id": 1,
                "ack_received_at_ns": 100_000_001,
                "at_ms": 100.0,
            })

        gate = physical._latency_gates(events, {"attested": False, "reason": "test_unattested"})

        self.assertEqual(gate["append_to_ack"]["status"], "FAIL")
        self.assertIn("listener consumes the frame", gate["append_to_ack"]["measurement_semantics"])
        self.assertEqual(
            gate["append_to_ack"]["policy_decision_metrics"],
            ["append_to_native_policy_decision", "append_to_accepted_native_policy_decision"],
        )
        self.assertTrue(gate["append_to_all_decision_events_received"]["missing"])
        self.assertEqual(gate["status"], "FAIL")

    def test_clock_attestation_accepts_module_style_server_entrypoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = _clock_attestation_manifest(
            root, ["/workspace/.venv_e2e_runtime/bin/python", "-m", "duplex_kernel.launch"],
        )
        result = physical.attest_shared_monotonic_clock(root, manifest)

        self.assertTrue(result["attested"], result)
        self.assertEqual(result["clock"], "CLOCK_MONOTONIC")
        self.assertEqual(result["server_container_id"], "same-task-container")

    def test_clock_attestation_still_accepts_script_style_server_entrypoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for script in ("duplex_kernel/launch.py", "/workspace/duplex_kernel/launch.py"):
            with self.subTest(script=script):
                manifest = _clock_attestation_manifest(root, ["python", script])
                self.assertTrue(physical.attest_shared_monotonic_clock(root, manifest)["attested"])

    def test_clock_attestation_rejects_launch_name_outside_invoked_module(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for server_argv in (
            ["python", "-m", "other.module", "duplex_kernel/launch.py"],
            ["python", "-m", "other.module", "-m", "duplex_kernel.launch"],
            ["python", "-m", "other.module"],
            ["other-command", "-m", "duplex_kernel.launch"],
            ["python", "not-duplex/launch.py"],
        ):
            with self.subTest(server_argv=server_argv):
                result = physical.attest_shared_monotonic_clock(
                    root, _clock_attestation_manifest(root, server_argv),
                )
                self.assertFalse(result["attested"], result)

    def test_clock_attestation_requires_client_python_entrypoint_at_argv_start(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for client_argv in (
            ["python", "-m", "other.module", "client/duplex_stream_e2e.py"],
            ["other-command", "client/duplex_stream_e2e.py"],
        ):
            with self.subTest(client_argv=client_argv):
                result = physical.attest_shared_monotonic_clock(
                    root,
                    _clock_attestation_manifest(
                        root,
                        ["/workspace/.venv_e2e_runtime/bin/python", "-m", "duplex_kernel.launch"],
                        client_argv,
                    ),
                )
                self.assertFalse(result["attested"], result)

    def test_native_talker_graph_flag_must_match_captured_manifest(self) -> None:
        response = _completed_response("r1")
        response["runtime_metrics"]["talker"].update({
            "talker_q1_backend": "native_audio_q1",
            "native_talker_q1_calls": 10,
            "native_talker_q1_graph_requested": False,
            "native_talker_q1_graph_calls": 0,
        })
        manifest = {"argv": {"docker_run": ["docker", "run", "--env", "DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH=1"]}}
        gate = _native_talker_gate({"responses": [response]}, manifest)
        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(any("does not match manifest" in item for item in gate["failures"]))

    def test_native_nongraph_path_is_valid_when_manifest_disables_graph(self) -> None:
        response = _completed_response("r1")
        response["runtime_metrics"]["talker"].update({
            "talker_q1_backend": "native_audio_q1",
            "native_talker_q1_calls": 10,
            "native_talker_q1_graph_requested": False,
            "native_talker_q1_graph_calls": 0,
        })
        manifest = {"argv": {"docker_run": ["docker", "run", "--env", "DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH=0"]}}
        self.assertEqual(_native_talker_gate({"responses": [response]}, manifest)["status"], "PASS")

    def test_20261010_capture_preserves_raw_failures_and_actual_duplex_overlap(self) -> None:
        capture = Path(__file__).resolve().parents[1] / "artifacts" / "pair_duplex_e2e_proof_20261010T220755Z"
        if not capture.is_dir():
            self.skipTest("the latest raw physical capture is not present in this checkout")
        result = verify_capture(capture)
        gates = result["gates"]
        self.assertEqual(result["status"], "FAIL")
        # This historical capture predates source edits, so current files cannot
        # attest its exact capture-time clock/source identity.
        self.assertFalse(result["clock_attestation"]["attested"])
        self.assertEqual(
            result["clock_attestation"]["reason"],
            "runner_source_hash_or_same_container_exec_check_failed",
        )
        native = gates["accepted_native_decisions_and_fresh_ages"]
        self.assertEqual((native["head_scored_count"], native["accepted_head_scored_count"]), (647, 607))
        self.assertEqual(native["expected_post_target_auto_response_disabled_rejections"], 35)
        self.assertEqual(sum(native["rejection_reason_counts"].values()), 40)
        self.assertEqual(gates["native_talker_calls"]["status"], "PASS")
        self.assertEqual(gates["response_drain_and_accounting"]["status"], "PASS")
        self.assertEqual(gates["routed_pcm_overlap"]["routed_frame_candidates"], 183)
        self.assertAlmostEqual(gates["routed_pcm_overlap"]["routed_pcm_overlap_duration_ms"], 1035.997739, places=5)
        latency = gates["append_latency"]
        self.assertAlmostEqual(latency["append_to_accepted_native_policy_decision"]["p95_ms"], 225.7914412, places=5)
        self.assertAlmostEqual(latency["append_to_ack"]["p95_ms"], 235.2205445, places=5)

    def test_1809_capture_remains_failed_for_missing_overlap_and_drain(self) -> None:
        capture = Path(__file__).resolve().parents[1] / "artifacts" / "pair_duplex_e2e_proof_20261010T180901Z_ac9e9217"
        if not capture.is_dir():
            self.skipTest("the preserved 18:09 raw capture is not present in this checkout")
        result = verify_capture(capture)
        gates = result["gates"]
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(gates["routed_pcm_overlap"]["status"], "FAIL")
        self.assertEqual(gates["routed_pcm_overlap"]["routed_pcm_overlap_duration_ms"], 0)
        self.assertEqual(gates["response_drain_and_accounting"]["status"], "FAIL")
        self.assertEqual(gates["response_drain_and_accounting"]["drain_event_count"], 0)
        self.assertGreater(gates["append_latency"]["append_to_ack"]["p95_ms"], 80)


if __name__ == "__main__":
    unittest.main()
