from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_pair_duplex_proof_3090 as runner


def _controller_guided_health() -> dict[str, object]:
    return {
        "status": "ok",
        "engine_loaded": True,
        "first_input_scheduler_anchor_enabled": False,
        "ready_frame_scheduler_dispatch_enabled": False,
        "native_listener_cuda_profile_enabled": False,
        "listener_model_lock_host_timeline_enabled": False,
        "listener_pair_lock_handoff_setting": "0",
        "listener_pair_lock_handoff_enabled": False,
        "ready_frame_scheduler_dispatch_sessions": [],
        "response_protocol": "controller_guided_say_v1",
        "direct_audio_candidate": True,
        "audio_history_archive_defer_offline": True,
        "audio_history_archive_cpu_startup_available": False,
        "talker_q1_backend": "native_audio_q1",
        "talker_q1_native_status": "ready",
        "native_action_selection": "controller_authoritative",
        "frame_residual_adapter": {
            "enabled": True,
            "architecture": "response_hidden_q1_query_film_v2",
            "checkpoint": "/workspace/checkpoints/frame_residual_adapter.pt",
        },
    }


def _code2wav_graph_family_health() -> dict[str, object]:
    health = _controller_guided_health()
    health["controller_guided_talk_context_pool"] = {
        "enabled": True,
        "capacity": 2,
        "memory": {
            "code2wav_graph_family_enabled": True,
            "code2wav_graph_odd_tail_enabled": False,
            "code2wav_graph_capture_status_by_slot": [
                {
                    "captured": True,
                    "graph_count": 37,
                    "window_frames": list(range(2, 75, 2)),
                    "capture_error": None,
                }
                for _ in range(2)
            ],
        },
    }
    return health


def _scheduler_anchor_preflight(enabled: bool = False) -> dict[str, object]:
    return {
        "first_input_scheduler_anchor_setting": "1" if enabled else "0",
        "first_input_scheduler_anchor_enabled": enabled,
        "ready_frame_scheduler_dispatch_setting": "0",
        "ready_frame_scheduler_dispatch_enabled": False,
        "native_control6_trace_setting": "0",
        "native_listener_cuda_profile_setting": "0",
        "native_listener_cuda_profile_directory_setting": None,
        "native_listener_cuda_profile_enabled": False,
        "listener_model_lock_host_timeline_enabled": False,
        "listener_pair_lock_handoff_setting": "0",
        "listener_pair_lock_handoff_enabled": False,
    }


def _code2wav_graph_odd_tail_health() -> dict[str, object]:
    health = _code2wav_graph_family_health()
    memory = health["controller_guided_talk_context_pool"]["memory"]
    memory["code2wav_graph_odd_tail_enabled"] = True
    memory["code2wav_graph_capture_status_by_slot"] = [
        {
            "captured": True,
            "graph_count": 74,
            "window_frames": list(range(1, 75)),
            "capture_error": None,
        }
        for _ in range(2)
    ]
    return health


def _native_listener_candidate_evidence() -> dict[str, object]:
    sizes = {
        "config": 224,
        "tensor_meta": 128,
        "layer_weights": 136,
        "weights": 64,
        "imported_kv": 32,
        "imported_cache": 480,
    }
    return {
        "configured_path": runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
        "path": runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
        "sha256": runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
        "abi_version": runner.EXPECTED_NATIVE_LISTENER_ABI,
        "source_manifest_path": runner.NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST,
        "source_manifest_sha256": runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST_SHA256,
        "source_root": runner.NATIVE_LISTENER_CANDIDATE_SOURCE_ROOT,
        "source_file_count": runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_FILE_COUNT,
        "abi_struct_sizes": sizes,
        "python_struct_sizes": dict(sizes),
    }


class PairDuplexRunnerTests(unittest.TestCase):
    def test_nvidia_smi_snapshot_parser_records_free_memory_and_compute_owners(self) -> None:
        snapshot = runner.parse_nvidia_smi_snapshot(
            "0, 24576, 18249, 6327, 0\n",
            "4149289, /workspace/.venv_e2e_runtime/bin/python, 18210\n",
        )

        self.assertEqual(snapshot["gpus"], [{
            "index": 0, "memory_total_mib": 24576, "memory_used_mib": 18249,
            "memory_free_mib": 6327, "utilization_pct": 0,
        }])
        self.assertEqual(snapshot["compute_apps"], [{
            "pid": "4149289", "process_name": "/workspace/.venv_e2e_runtime/bin/python",
            "used_gpu_memory_mib": 18210,
        }])

    def test_gpu_vram_release_gate_distinguishes_managed_and_unrelated_owners(self) -> None:
        released = runner.parse_nvidia_smi_snapshot("0, 24576, 500, 24076, 0\n", "")
        runner.assert_gpu_vram_released({"gpu_process_pids_before_stop": ["101"]}, released)

        managed_survivor = runner.parse_nvidia_smi_snapshot(
            "0, 24576, 18000, 6576, 0\n", "101, python, 17500\n",
        )
        with self.assertRaisesRegex(RuntimeError, "managed GPU process is still resident"):
            runner.assert_gpu_vram_released(
                {"gpu_process_pids_before_stop": ["101"]}, managed_survivor,
            )

        small_unrelated_owner = runner.parse_nvidia_smi_snapshot(
            "0, 24576, 2000, 22576, 0\n", "202, unrelated-job, 32\n",
        )
        with self.assertRaisesRegex(RuntimeError, "unrelated GPU compute process remains"):
            runner.assert_gpu_vram_released(
                {"gpu_process_pids_before_stop": ["101"]}, small_unrelated_owner,
            )

    def test_gpu_release_barrier_waits_boundedly_for_managed_context_teardown(self) -> None:
        stopped = json.dumps([{"Id": "managed-id", "State": {"Running": False}}])
        gpu = "0, 24576, 18000, 6576, 0\n"
        resident = "101, python, 17500\n"
        released = "0, 24576, 500, 24076, 0\n"
        commands = [
            subprocess.CompletedProcess([], 0, stopped, ""),
            subprocess.CompletedProcess([], 0, gpu, ""),
            subprocess.CompletedProcess([], 0, resident, ""),
            subprocess.CompletedProcess([], 0, "101\n", ""),
            subprocess.CompletedProcess([], 0, stopped, ""),
            subprocess.CompletedProcess([], 0, released, ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
        ]
        with (
            patch.object(runner, "_command", side_effect=commands) as command,
            patch.object(runner.time, "monotonic", return_value=0.0),
            patch.object(runner.time, "sleep"),
        ):
            result = runner.wait_for_gpu_vram_release(
                {"id": "managed-id", "was_running": True, "gpu_process_pids_before_stop": ["101"]},
                Path("/repo"), timeout_s=1.0, poll_interval_s=0.1,
            )

        self.assertTrue(result["released"])
        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(command.call_count, 8)

    def test_gpu_release_barrier_clips_sleep_and_command_timeouts_to_deadline(self) -> None:
        now = [0.0]
        sleeps: list[float] = []
        command_timeouts: list[float] = []

        def fake_command(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            timeout = kwargs.get("timeout")
            self.assertIsInstance(timeout, float)
            command_timeouts.append(float(timeout))
            if argv[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(argv, 0, json.dumps([{
                    "Id": "managed-id", "State": {"Running": False},
                }]), "")
            if argv[0] == "nvidia-smi" and "--query-gpu=" in argv[1]:
                return subprocess.CompletedProcess(argv, 0, "0, 24576, 18000, 6576, 0\n", "")
            if argv[0] == "nvidia-smi":
                return subprocess.CompletedProcess(argv, 0, "101, python, 17500\n", "")
            if argv[:2] == ["ps", "-p"]:
                return subprocess.CompletedProcess(argv, 0, "101\n", "")
            raise AssertionError(f"unexpected command: {argv}")

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        with (
            patch.object(runner, "_command", side_effect=fake_command),
            patch.object(runner.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(runner.time, "sleep", side_effect=fake_sleep),
        ):
            with self.assertRaisesRegex(runner.GpuVramBarrierError, r"managed process PID\(s\) still alive"):
                runner.wait_for_gpu_vram_release(
                    {"id": "managed-id", "was_running": True,
                     "gpu_process_pids_before_stop": ["101"]},
                    Path("/repo"), timeout_s=0.1, poll_interval_s=0.5,
                )

        self.assertEqual(sleeps, [0.1])
        self.assertTrue(command_timeouts)
        self.assertLessEqual(max(command_timeouts), 0.1)

    def test_cli_preserves_explicit_managed_container_argument(self) -> None:
        with (
            patch.object(runner, "run_proof") as run_proof,
        ):
            runner.main([
                "--repo-root", "/repo", "--output-dir", "/repo/artifacts/proof",
                "--maintenance-container", runner.DEFAULT_MAINTENANCE_CONTAINER,
            ])
        self.assertEqual(run_proof.call_args.args[0].maintenance_container, runner.DEFAULT_MAINTENANCE_CONTAINER)
        self.assertIn("--maintenance-container", run_proof.call_args.args[0].runner_argv)

    def test_no_maintenance_container_requires_an_unowned_idle_gpu(self) -> None:
        released = runner.parse_nvidia_smi_snapshot("0, 24576, 500, 24076, 0\n", "")
        runner.assert_gpu_ready_without_maintenance(released)

        live_owner = runner.parse_nvidia_smi_snapshot(
            "0, 24576, 2000, 22576, 0\n", "202, unrelated-job, 32\n",
        )
        with self.assertRaisesRegex(RuntimeError, "GPU compute process exists without a maintenance container"):
            runner.assert_gpu_ready_without_maintenance(live_owner)

    def _run_successful_client(
        self,
        args: argparse.Namespace,
        *,
        cleanup_side_effect: object = None,
        analyzer_result: dict[str, object] | None = None,
    ) -> int:
        repository_root = Path(args.repo_root)
        patches = [
            patch.object(runner, "collect_run_metadata", return_value={
                "git": {"sha": "abc"},
                "inputs": {},
                "response_bundle_manifest": {"metadata": {"bundle_tag": "fixture"}},
            }),
            patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
            patch.object(runner, "probe_container_runtime", return_value={"python": "3.11", "packages": {"torch": "2.0", "transformers": "5.0"}, **_scheduler_anchor_preflight()}),
            patch.object(runner, "probe_cli_entrypoints", return_value={"server": {"usage": "usage: launch.py"}, "client": {"usage": "usage: client.py"}}),
            patch.object(runner, "nvidia_smi_memory_snapshot", return_value={
                "gpus": [{"index": 0, "memory_total_mib": 24576, "memory_used_mib": 512,
                          "memory_free_mib": 24064, "utilization_pct": 0}],
                "compute_apps": [],
            }),
            patch.object(runner, "launch_server_container", return_value="task-container-id"),
            patch.object(runner, "start_server_log_capture", return_value=None),
            patch.object(runner, "wait_for_server_ready", return_value=_controller_guided_health()),
            patch.object(runner, "record_runtime_versions", return_value={"python": "3.11"}),
            patch.object(runner, "run_client", return_value=subprocess.CompletedProcess([], 0)),
            patch.object(runner, "attest_native_talker_binary", return_value={
                "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
            }),
            patch.object(runner, "native_talker_postrun_attestation", return_value={"status": "PASS"}),
            patch.object(runner, "archive_defer_postrun_attestation", return_value={"status": "PASS"}),
            patch.object(runner, "run_analyzer", return_value=analyzer_result) if analyzer_result is not None else patch.object(runner, "run_analyzer", wraps=runner.run_analyzer),
            patch.object(runner, "cleanup_owned_container", side_effect=cleanup_side_effect),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14]:
            return runner.run_proof(args)

    def test_provenance_hashes_nested_checkpoint_files_and_runtime_probe_parses(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            nested = root / "checkpoints" / "adapter" / "nested"
            nested.mkdir(parents=True)
            (nested / "adapter.safetensors").write_bytes(b"checkpoint-bytes")
            entries = runner._hash_tree(root / "checkpoints" / "adapter", repository_root=root)

            self.assertEqual(entries[0]["path"], "checkpoints/adapter/nested/adapter.safetensors")
            self.assertEqual(len(entries[0]["sha256"]), 64)
            compile(runner.runtime_version_probe_code(), "runtime-version-probe", "exec")

    def test_analyzer_checkpoint_schema_attests_files_and_qlora_tree(self) -> None:
        file_hash = "a" * 64
        qlora_hash = "b" * 64
        attestations = runner.analyzer_checkpoint_attestations({
            "artifacts": {
                "native_control6": [{
                    "path": runner.ANALYZER_CHECKPOINTS["native_control6"], "sha256": file_hash,
                }],
                "gemma_qlora": [{
                    "path": "checkpoints/gemma4_action_qlora_v1/adapter/config.json", "sha256": qlora_hash,
                }],
                "frame_residual": [{
                    "path": runner.ANALYZER_CHECKPOINTS["frame_residual"], "sha256": file_hash,
                }],
            },
        })

        self.assertEqual(attestations["native_control6"], {
            "path": runner.ANALYZER_CHECKPOINTS["native_control6"], "sha256": file_hash,
        })
        self.assertEqual(attestations["frame_residual"], {
            "path": runner.ANALYZER_CHECKPOINTS["frame_residual"], "sha256": file_hash,
        })
        self.assertEqual(attestations["gemma_qlora"]["path"], runner.ANALYZER_CHECKPOINTS["gemma_qlora"])
        self.assertEqual(len(attestations["gemma_qlora"]["sha256"]), 64)
        self.assertNotEqual(attestations["gemma_qlora"]["sha256"], qlora_hash)

    def test_metadata_accepts_container_only_runtime_python(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/native_control6_gemma_native_rows_rank16_real_mixed_600.pt").write_bytes(b"native")
            (root / "checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt").write_bytes(b"residual")
            (root / "checkpoints/gemma4_action_qlora_v1").mkdir()
            (root / "checkpoints/gemma4_action_qlora_v1/adapter.safetensors").write_bytes(b"qlora")
            bundle = root / "checkpoints/experimental-v5-repair200-fphase-20261003"
            bundle.mkdir()
            (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
            audio = root / "artifacts/scenario_audio_qwen3tts/mode_b_kickoff.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"wav")
            (root / "duplex_kernel").mkdir()
            (root / "duplex_kernel/launch.py").write_text("pass\n", encoding="utf-8")
            (root / "client").mkdir()
            (root / "client/duplex_stream_e2e.py").write_text("pass\n", encoding="utf-8")

            with (
                patch.object(runner, "_git_state", return_value={"sha": "abc", "dirty": False}),
                patch.object(runner, "_source_hashes", return_value={}),
            ):
                metadata = runner.collect_run_metadata(root, bundle)

        server_python = metadata["artifacts"]["server_python"]
        self.assertFalse(server_python["host_exists"])
        self.assertEqual(server_python["container_path"], "/workspace/.venv_e2e_runtime/bin/python")

    def test_runtime_probe_uses_exact_image_path_without_gpu(self) -> None:
        argv = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        self.assertEqual(argv[argv.index("--entrypoint") + 1], "/workspace/.venv_e2e_runtime/bin/python")
        self.assertEqual(argv[argv.index("--user") + 1], "1000:1000")
        self.assertTrue(argv[argv.index("--volume") + 1].endswith("/repo:/workspace"))
        self.assertNotIn("--gpus", argv)
        self.assertNotIn("--privileged", argv)
        self.assertNotIn("--network", argv)

        native_talker = {
            "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "size_bytes": 123,
            "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
            "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
            "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
        }
        versions = {
            "python": "3.11.9",
            "packages": {"torch": "2.7.0", "transformers": "5.0.0"},
            "native_talker": native_talker,
            "talker_native_q1_setting": "1",
            "talker_native_q1_graph_setting": "1",
            **_scheduler_anchor_preflight(),
        }
        with patch.object(runner, "_command", return_value=subprocess.CompletedProcess(argv, 0, json.dumps(versions), "")) as command:
            self.assertEqual(runner.probe_container_runtime("/repo", "gemma4-omni:latest"), versions)
        command.assert_called_once()

    def test_qwen_native_library_path_is_identical_in_cpu_preflight_and_server(self) -> None:
        expected_env = "QWEN_NATIVE_LIBRARY_PATH=/workspace/native/qwen_talker/libqwen_talker.so"
        preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-unique",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        )

        self.assertIn(expected_env, preflight)
        self.assertIn(expected_env, server)
        for setting in (
            "DUPLEX_TALKER_NATIVE_AUDIO_Q1=1",
            "DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH=1",
        ):
            self.assertIn(setting, preflight)
            self.assertIn(setting, server)
        self.assertNotIn("--gpus", preflight)
        self.assertIn(expected_env.split("=", 1)[1], runner.native_talker_binary_probe_code())

    def test_anchor_and_native_control6_trace_flags_are_explicit_in_preflight_and_server(self) -> None:
        expected_anchor = f"{runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV}=1"
        expected_trace = "DUPLEX_NATIVE_CONTROL6_TRACE=0"
        preflight_default = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        server_default = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-default",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        )
        preflight_anchor = runner.build_runtime_probe_argv(
            "/repo", "gemma4-omni:latest", first_input_scheduler_anchor=True,
        )
        server_anchor = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-anchor",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            first_input_scheduler_anchor=True,
        )

        self.assertIn(f"{runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV}=0", preflight_default)
        self.assertIn(f"{runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV}=0", server_default)
        self.assertNotIn(expected_anchor, preflight_default)
        self.assertNotIn(expected_anchor, server_default)
        self.assertIn(expected_anchor, preflight_anchor)
        self.assertIn(expected_anchor, server_anchor)
        for argv in (preflight_default, server_default, preflight_anchor, server_anchor):
            self.assertIn(expected_trace, argv)
        self.assertIn("'native_control6_trace_setting': os.environ.get('DUPLEX_NATIVE_CONTROL6_TRACE')", runner.runtime_version_probe_code())
        self.assertIn("QWEN_NATIVE_LIBRARY_PATH", runner.runtime_version_probe_code())
        self.assertIn("DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH", runner.runtime_version_probe_code())

    def test_native_listener_candidate_override_is_proof_only_and_shared_by_preflight_and_server(self) -> None:
        default_preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        default_server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-default",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        )
        env_prefix = f"{runner.NATIVE_LISTENER_LIBRARY_PATH_ENV}="
        self.assertFalse(any(arg.startswith(env_prefix) for arg in default_preflight + default_server))

        preflight = runner.build_runtime_probe_argv(
            "/repo", "gemma4-omni:latest", native_listener_capacity_candidate=True,
        )
        server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-candidate",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            native_listener_capacity_candidate=True,
            native_listener_attestation_dir="/workspace/artifacts/test-run/listener_attestations",
        )
        expected = f"{env_prefix}{runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH}"
        self.assertEqual(preflight.count(expected), 1)
        self.assertEqual(server.count(expected), 1)
        sidecar_env = (
            f"{runner.NATIVE_LISTENER_ATTESTATION_DIRECTORY_ENV}="
            "/workspace/artifacts/test-run/listener_attestations"
        )
        self.assertEqual(server.count(sidecar_env), 1)
        self.assertNotIn(runner.NATIVE_LISTENER_ATTESTATION_DIRECTORY_ENV, default_server)
        with self.assertRaises(ValueError):
            runner.build_docker_run_argv(
                repository_root="/home/administrator/gemma-4-omni",
                container_name="pair-duplex-proof-candidate",
                image="gemma4-omni:latest",
                server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
                native_listener_capacity_candidate=True,
            )
        with self.assertRaises(ValueError):
            runner.build_docker_run_argv(
                repository_root="/home/administrator/gemma-4-omni",
                container_name="pair-duplex-proof-candidate",
                image="gemma4-omni:latest",
                server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
                native_listener_capacity_candidate=True,
                native_listener_attestation_dir="/tmp/outside/listener_attestations",
            )
        self.assertNotIn("--gpus", preflight)
        self.assertIn(runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256, preflight[-1])
        self.assertIn(runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST_SHA256, preflight[-1])
        self.assertNotIn(runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256, default_preflight[-1])
        self.assertEqual(
            runner.build_parser().parse_args([]).native_listener_capacity_candidate,
            False,
        )
        self.assertTrue(runner.build_parser().parse_args(["--native-listener-capacity-candidate"]).native_listener_capacity_candidate)

    def test_native_listener_candidate_attestation_fails_closed_on_identity_or_abi_mismatch(self) -> None:
        evidence = _native_listener_candidate_evidence()
        self.assertEqual(runner.validate_native_listener_candidate(evidence), evidence)
        for change in (
            {"path": "/workspace/native/gemma4_listener/libgemma4_listener.so"},
            {"sha256": "0" * 64},
            {"abi_version": 2},
            {"source_manifest_sha256": "1" * 64},
            {"source_file_count": 21},
            {"python_struct_sizes": {**evidence["python_struct_sizes"], "config": 1}},
        ):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                runner.validate_native_listener_candidate({**evidence, **change})

    def test_native_listener_candidate_preflight_requires_attested_candidate(self) -> None:
        talker = {
            "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "size_bytes": 123,
            "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
            "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
            "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
        }
        payload = {
            "python": "3.11.9",
            "packages": {"torch": "2.12.1+cu130", "transformers": "5.5.0"},
            "native_talker": talker,
            "native_listener_candidate": _native_listener_candidate_evidence(),
            "talker_native_q1_setting": "1",
            "talker_native_q1_graph_setting": "1",
            "code2wav_graph_family_setting": "1",
            "code2wav_decode_frames_setting": "2",
            "code2wav_graph_odd_tail_setting": "1",
            **_scheduler_anchor_preflight(),
        }
        with patch.object(
            runner, "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), ""),
        ):
            result = runner.probe_container_runtime(
                "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
                code2wav_graph_odd_tail=True, native_listener_capacity_candidate=True,
            )
        self.assertEqual(result["native_listener_candidate"]["sha256"], runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256)

        for invalid in (
            {**payload, "native_listener_candidate": None},
            {**payload, "native_listener_candidate": {**payload["native_listener_candidate"], "sha256": "0" * 64}},
        ):
            with patch.object(
                runner, "_command",
                return_value=subprocess.CompletedProcess([], 0, json.dumps(invalid), ""),
            ):
                with self.assertRaises(RuntimeError):
                    runner.probe_container_runtime(
                        "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
                        code2wav_graph_odd_tail=True, native_listener_capacity_candidate=True,
                    )

    def test_native_listener_postrun_attestation_uses_atomic_session_sidecars(self) -> None:
        participants = [
            {"speaker": "A", "session_id": "sess-a"},
            {"speaker": "B", "session_id": "sess-b"},
        ]
        runtime = {
            "library_path": runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
            "library_sha256": runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".tmp") as directory:
            root = Path(directory)
            report_path = root / "report.json"
            log_path = root / "server.log"
            attestation_dir = root / "listener_attestations"
            attestation_dir.mkdir()
            report_path.write_text(json.dumps({"participants": participants}), encoding="utf-8")
            log_path.write_text(
                "\n".join((
                    "2026-10-12T00:00:00Z stdout F [realtime_session_metrics] {\"broken\":2026-10-12T00:00:01Z}",
                    "2026-10-12T00:00:00Z stdout F [realtime_session_metrics] {\"broken\":2026-10-12T00:00:02Z}",
                )),
                encoding="utf-8",
            )
            for participant in participants:
                payload = {
                    "schema_version": 1,
                    "session_id": participant["session_id"],
                    **runtime,
                }
                (attestation_dir / runner.native_listener_attestation_filename(participant["session_id"])).write_text(
                    json.dumps(payload), encoding="utf-8",
                )
            passed = runner.native_listener_session_postrun_attestation(
                report_path, log_path,
                attestation_dir=attestation_dir,
                expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            self.assertEqual(passed["status"], "PASS")
            self.assertEqual(passed["malformed_metric_rows"], 2)

            sidecar_b = attestation_dir / runner.native_listener_attestation_filename("sess-b")
            for invalid_available in (False, 0, "false", None):
                sidecar_b.write_text(
                    json.dumps({
                        "schema_version": 1,
                        "session_id": "sess-b",
                        "available": invalid_available,
                        **runtime,
                    }),
                    encoding="utf-8",
                )
                invalid_availability = runner.native_listener_session_postrun_attestation(
                    report_path, log_path,
                    attestation_dir=attestation_dir,
                    expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                    expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
                )
                self.assertEqual(invalid_availability["status"], "FAIL", repr(invalid_available))

            sidecar_b.write_text(
                json.dumps({
                    "schema_version": 1,
                    "session_id": "sess-b",
                    "available": True,
                    **runtime,
                }),
                encoding="utf-8",
            )
            available_true = runner.native_listener_session_postrun_attestation(
                report_path, log_path,
                attestation_dir=attestation_dir,
                expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            self.assertEqual(available_true["status"], "PASS")

            wrong_runtime = {**runtime, "library_sha256": "0" * 64}
            sidecar_b.write_text(
                json.dumps({"schema_version": 1, "session_id": "sess-b", **wrong_runtime}),
                encoding="utf-8",
            )
            failed = runner.native_listener_session_postrun_attestation(
                report_path, log_path,
                attestation_dir=attestation_dir,
                expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            self.assertEqual(failed["status"], "FAIL")
            self.assertFalse(failed["sessions"][1]["pass"])

            sidecar_b.unlink()
            missing = runner.native_listener_session_postrun_attestation(
                report_path, log_path,
                attestation_dir=attestation_dir,
                expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            self.assertEqual(missing["status"], "FAIL")

    def test_native_listener_postrun_sidecars_reject_duplicate_sessions_and_escape(self) -> None:
        participants = [
            {"speaker": "A", "session_id": "sess-a"},
            {"speaker": "B", "session_id": "sess-b"},
        ]
        runtime = {
            "library_path": runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
            "library_sha256": runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".tmp") as directory:
            root = Path(directory)
            report_path = root / "report.json"
            log_path = root / "server.log"
            attestation_dir = root / "listener_attestations"
            attestation_dir.mkdir()
            log_path.write_text("", encoding="utf-8")
            for participant in participants:
                payload = {"schema_version": 1, "session_id": participant["session_id"], **runtime}
                (attestation_dir / runner.native_listener_attestation_filename(participant["session_id"])).write_text(
                    json.dumps(payload), encoding="utf-8",
                )
            report_path.write_text(json.dumps({"participants": [participants[0], {**participants[1], "session_id": "sess-a"}]}), encoding="utf-8")
            duplicate = runner.native_listener_session_postrun_attestation(
                report_path, log_path, attestation_dir=attestation_dir,
                expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            self.assertEqual(duplicate["status"], "FAIL")
            report_path.write_text(json.dumps({"participants": participants}), encoding="utf-8")
            escaped = runner.native_listener_session_postrun_attestation(
                report_path, log_path, attestation_dir=root.parent / "outside_attestations",
                expected_library_path=runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=runner.EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            self.assertEqual(escaped["status"], "FAIL")

    def test_code2wav_graph_family_opt_in_is_shared_by_cpu_preflight_and_server_only(self) -> None:
        default_preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        default_server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-default",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        )
        self.assertNotIn("GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1", default_preflight + default_server)
        self.assertNotIn("GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2", default_preflight + default_server)

        preflight = runner.build_runtime_probe_argv(
            "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
        )
        server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-opt-in",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            code2wav_graph_family=True,
        )
        for setting in (
            "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1",
            "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2",
            "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=0",
        ):
            self.assertIn(setting, preflight)
            self.assertIn(setting, server)
        self.assertIn("--controller-guided-talk-context-pool", server)
        self.assertNotIn("--gpus", preflight)

        self.assertFalse(runner.build_parser().parse_args([]).code2wav_graph_family)
        self.assertTrue(runner.build_parser().parse_args(["--code2wav-graph-family"]).code2wav_graph_family)

    def test_code2wav_odd_tail_opt_in_is_explicit_and_shared_by_preflight_and_server(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --code2wav-graph-family"):
            runner.validate_code2wav_graph_options(
                code2wav_graph_family=False,
                code2wav_graph_odd_tail=True,
            )

        default = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        self.assertNotIn("GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1", default)
        preflight = runner.build_runtime_probe_argv(
            "/repo", "gemma4-omni:latest",
            code2wav_graph_family=True,
            code2wav_graph_odd_tail=True,
        )
        server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-odd-tail",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            code2wav_graph_family=True,
            code2wav_graph_odd_tail=True,
        )
        for setting in (
            "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1",
            "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2",
            "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1",
        ):
            self.assertIn(setting, preflight)
            self.assertIn(setting, server)
        self.assertNotIn("--gpus", preflight)

        parser_args = runner.build_parser().parse_args([
            "--code2wav-graph-family", "--code2wav-graph-odd-tail",
        ])
        self.assertTrue(parser_args.code2wav_graph_family)
        self.assertTrue(parser_args.code2wav_graph_odd_tail)
        with self.assertRaises(SystemExit):
            runner.main(["--code2wav-graph-odd-tail"])

    def test_first_input_scheduler_anchor_is_explicit_and_identical_in_preflight_and_server(self) -> None:
        env = runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV
        default_preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        default_server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-default-anchor",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        )
        enabled_preflight = runner.build_runtime_probe_argv(
            "/repo", "gemma4-omni:latest", first_input_scheduler_anchor=True,
        )
        enabled_server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-enabled-anchor",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            first_input_scheduler_anchor=True,
        )

        self.assertIn(f"{env}=0", default_preflight)
        self.assertIn(f"{env}=0", default_server)
        self.assertIn(f"{env}=1", enabled_preflight)
        self.assertIn(f"{env}=1", enabled_server)
        self.assertNotIn("--gpus", enabled_preflight)
        self.assertIn("first_input_scheduler_anchor_setting", runner.runtime_version_probe_code())
        self.assertIn("first_input_scheduler_anchor_enabled", runner.runtime_version_probe_code())
        self.assertFalse(runner.build_parser().parse_args([]).first_input_scheduler_anchor)
        self.assertTrue(
            runner.build_parser().parse_args(["--first-input-scheduler-anchor"]).first_input_scheduler_anchor
        )

    def test_first_input_scheduler_anchor_health_must_match_opt_in(self) -> None:
        self.assertEqual(
            runner.first_input_scheduler_anchor_record(
                {"first_input_scheduler_anchor_enabled": True}, requested=True,
            ),
            {"requested": True, "actual": True, "status": "PASS"},
        )
        with self.assertRaisesRegex(RuntimeError, "health.*differs"):
            runner.first_input_scheduler_anchor_record({}, requested=True)
        with self.assertRaisesRegex(RuntimeError, "health.*differs"):
            runner.first_input_scheduler_anchor_record(
                {"first_input_scheduler_anchor_enabled": False}, requested=True,
            )

    def test_ready_frame_dispatch_flag_is_explicit_default_off_and_mutually_exclusive(self) -> None:
        flag = runner.READY_FRAME_SCHEDULER_DISPATCH_ENV
        default_preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
        default_server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-ready-default",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        )
        enabled_preflight = runner.build_runtime_probe_argv(
            "/repo", "gemma4-omni:latest", ready_frame_scheduler_dispatch=True,
        )
        enabled_server = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-ready-enabled",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            ready_frame_scheduler_dispatch=True,
        )
        self.assertIn(f"{flag}=0", default_preflight)
        self.assertIn(f"{flag}=0", default_server)
        self.assertIn(f"{flag}=1", enabled_preflight)
        self.assertIn(f"{flag}=1", enabled_server)
        parsed_default = runner.build_parser().parse_args([])
        parsed_enabled = runner.build_parser().parse_args(["--ready-frame-scheduler-dispatch"])
        self.assertFalse(parsed_default.ready_frame_scheduler_dispatch)
        self.assertTrue(parsed_enabled.ready_frame_scheduler_dispatch)
        self.assertIn("READY_FRAME_SCHEDULER_DISPATCH_ENABLED", runner.runtime_version_probe_code())
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.validate_scheduler_mode(
                first_input_scheduler_anchor=True,
                ready_frame_scheduler_dispatch=True,
            )
        with self.assertRaises(SystemExit):
            runner.main([
                "--first-input-scheduler-anchor",
                "--ready-frame-scheduler-dispatch",
            ])

    def test_ready_frame_dispatch_health_requires_global_and_per_session_setting(self) -> None:
        health = {
            "ready_frame_scheduler_dispatch_enabled": True,
            "ready_frame_scheduler_dispatch_sessions": [
                {"session_id": "a", "enabled": True},
                {"session_id": "b", "enabled": True},
            ],
        }
        self.assertEqual(
            runner.ready_frame_scheduler_dispatch_record(health, requested=True),
            {"requested": True, "actual": True, "status": "PASS"},
        )
        with self.assertRaisesRegex(RuntimeError, "health differs"):
            runner.ready_frame_scheduler_dispatch_record({}, requested=True)
        with self.assertRaisesRegex(RuntimeError, "session mode"):
            runner.ready_frame_scheduler_dispatch_record(
                {
                    "ready_frame_scheduler_dispatch_enabled": True,
                    "ready_frame_scheduler_dispatch_sessions": [
                        {"session_id": "a", "enabled": False},
                    ],
                },
                requested=True,
            )

    def test_ready_frame_dispatch_cpu_preflight_checks_requested_setting(self) -> None:
        native_talker = {
            "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "size_bytes": 123,
            "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
            "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
            "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
        }
        evidence = {
            "python": "3.11.9",
            "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
            "native_talker": native_talker,
            "talker_native_q1_setting": "1",
            "talker_native_q1_graph_setting": "1",
            "native_control6_trace_setting": "0",
            "first_input_scheduler_anchor_setting": "0",
            "first_input_scheduler_anchor_enabled": False,
            "ready_frame_scheduler_dispatch_setting": "1",
            "ready_frame_scheduler_dispatch_enabled": True,
            "native_listener_cuda_profile_setting": "0",
            "native_listener_cuda_profile_directory_setting": None,
            "native_listener_cuda_profile_enabled": False,
            "listener_model_lock_host_timeline_enabled": False,
            "listener_pair_lock_handoff_setting": "0",
            "listener_pair_lock_handoff_enabled": False,
        }
        with patch.object(
            runner,
            "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(evidence), ""),
        ):
            value = runner.probe_container_runtime(
                "/repo",
                "gemma4-omni:latest",
                ready_frame_scheduler_dispatch=True,
            )
        self.assertTrue(value["ready_frame_scheduler_dispatch_enabled"])

    def test_run_proof_passes_ready_dispatch_to_both_preflight_and_live_container(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "ready-dispatch"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root),
                "--output-dir", str(output),
                "--ready-frame-scheduler-dispatch",
            ])
            native_talker = {
                "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
                "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
                "size_bytes": 123,
                "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
            }
            runtime = {
                "python": "3.11",
                "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
                "native_talker": native_talker,
                "native_listener_cuda_profile_setting": "0",
                "native_listener_cuda_profile_directory_setting": None,
                "native_listener_cuda_profile_enabled": False,
                "listener_model_lock_host_timeline_enabled": False,
                "listener_pair_lock_handoff_setting": "0",
                "listener_pair_lock_handoff_enabled": False,
                "first_input_scheduler_anchor_setting": "0",
                "first_input_scheduler_anchor_enabled": False,
                "ready_frame_scheduler_dispatch_setting": "1",
                "ready_frame_scheduler_dispatch_enabled": True,
            }
            health = {
                **_controller_guided_health(),
                "ready_frame_scheduler_dispatch_enabled": True,
                "ready_frame_scheduler_dispatch_sessions": [],
            }
            with (
                patch.object(runner, "collect_run_metadata", return_value={
                    "git": {"sha": "abc"},
                    "inputs": {},
                    "response_bundle_manifest": {"metadata": {"bundle_tag": "fixture"}},
                }),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value=runtime) as preflight,
                patch.object(runner, "probe_cli_entrypoints", return_value={
                    "server": {"usage": "usage: launch.py"},
                    "client": {"usage": "usage: client.py"},
                }),
                patch.object(runner, "nvidia_smi_memory_snapshot", return_value={
                    "gpus": [{"index": 0, "memory_total_mib": 24576, "memory_used_mib": 512,
                              "memory_free_mib": 24064, "utilization_pct": 0}],
                    "compute_apps": [],
                }),
                patch.object(runner, "_command", return_value=subprocess.CompletedProcess([], 0, "task-id\n", "")),
                patch.object(runner, "launch_server_container", wraps=runner.launch_server_container) as launch,
                patch.object(runner, "start_server_log_capture", return_value=None),
                patch.object(runner, "wait_for_server_ready", return_value=health),
                patch.object(runner, "record_runtime_versions", return_value=runtime),
                patch.object(runner, "attest_native_talker_binary", return_value={
                    "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                    "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                    "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
                }),
                patch.object(runner, "run_client", return_value=subprocess.CompletedProcess([], 0)),
                patch.object(runner, "native_talker_postrun_attestation", return_value={"status": "PASS"}),
                patch.object(runner, "archive_defer_postrun_attestation", return_value={"status": "PASS"}),
                patch.object(runner, "run_analyzer", return_value={"status": "PASS", "exit_code": 0}),
                patch.object(runner, "cleanup_owned_container"),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 0)
            self.assertTrue(preflight.call_args.kwargs["ready_frame_scheduler_dispatch"])
            self.assertTrue(launch.call_args.kwargs["ready_frame_scheduler_dispatch"])
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["ready_frame_scheduler_dispatch"],
                {
                    "requested": True,
                    "status": "PASS",
                    "cpu_preflight_setting": "1",
                    "cpu_preflight_enabled": True,
                    "cpu_preflight_status": "PASS",
                    "actual": True,
                },
            )
            self.assertIn(
                f"{runner.READY_FRAME_SCHEDULER_DISPATCH_ENV}=1",
                manifest["argv"]["docker_run"],
            )

    def test_canonical_report_uses_physical_gate_and_preserves_legacy_analyzer_markdown(self) -> None:
        from tools import verify_pair_physical_runtime as physical

        result = {
            "schema_version": 1,
            "status": "FAIL",
            "gates": {
                "append_latency": {
                    "status": "FAIL",
                    "failures": ["p95 above 80 ms"],
                    "missing": [],
                },
                "natural_eos": {"status": "PASS", "failures": [], "missing": []},
            },
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            output = Path(temp_dir)
            for name in ("report.json", "events.jsonl", "server_run_manifest.json"):
                (output / name).write_text("{}\n", encoding="utf-8")
            canonical = output / "pair_duplex_certification_report.md"
            canonical.write_text("# Legacy analyzer status: PASS\n", encoding="utf-8")
            with patch.object(physical, "verify_capture", return_value=result):
                actual = runner.write_physical_gate_artifacts(output)

            self.assertEqual(actual, result)
            self.assertEqual(
                json.loads((output / "physical_runtime_gate.json").read_text(encoding="utf-8")),
                result,
            )
            self.assertIn("**Status: FAIL**", canonical.read_text(encoding="utf-8"))
            self.assertIn("listener-ingress `input_audio_buffer.consumed` ACK", canonical.read_text(encoding="utf-8"))
            self.assertIn("ACK receipt alone cannot satisfy either policy gate", canonical.read_text(encoding="utf-8"))
            self.assertIn("append_latency", canonical.read_text(encoding="utf-8"))
            self.assertEqual(
                (output / "legacy_pair_duplex_certification_report.md").read_text(encoding="utf-8"),
                "# Legacy analyzer status: PASS\n",
            )

    def test_canonical_report_aggregates_nested_failure_when_parent_status_is_absent(self) -> None:
        result = {
            "schema_version": 1,
            "status": "FAIL",
            "gates": {
                "append_latency": {
                    "append_to_ack": {"status": "FAIL", "p95_ms": 123.2},
                    "other_latency_evidence": {"status": "INCOMPLETE"},
                },
                "natural_eos": {"status": "PASS", "failures": [], "missing": []},
            },
            "nested_status_sources": {
                "append_latency.append_to_ack": "FAIL",
                "append_latency.other_latency_evidence": "INCOMPLETE",
                "natural_eos": "PASS",
            },
        }

        report = runner._physical_gate_markdown(result)

        self.assertIn("| append_latency | **FAIL** |", report)
        self.assertNotIn("| append_latency | **INCOMPLETE** |", report)

    def test_code2wav_graph_family_preflight_fails_closed_on_missing_or_wrong_runtime_settings(self) -> None:
        native_talker = {
            "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "size_bytes": 123,
            "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
            "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
            "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
        }
        payload = {
            "python": "3.11.9",
            "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
            "native_talker": native_talker,
            "talker_native_q1_setting": "1",
            "talker_native_q1_graph_setting": "1",
            "code2wav_graph_family_setting": "1",
            "code2wav_decode_frames_setting": "2",
            "code2wav_graph_odd_tail_setting": "1",
            **_scheduler_anchor_preflight(),
        }
        with patch.object(
            runner, "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), ""),
        ):
            self.assertEqual(
                runner.probe_container_runtime(
                    "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
                    code2wav_graph_odd_tail=True,
                ),
                payload,
            )

        for changes in (
            {"code2wav_graph_family_setting": None},
            {"code2wav_graph_family_setting": "0"},
            {"code2wav_decode_frames_setting": "3"},
            {"code2wav_graph_odd_tail_setting": "0"},
        ):
            bad_payload = {**payload, **changes}
            with patch.object(
                runner, "_command",
                return_value=subprocess.CompletedProcess([], 0, json.dumps(bad_payload), ""),
            ):
                with self.assertRaisesRegex(RuntimeError, "Code2Wav graph family"):
                    runner.probe_container_runtime(
                        "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
                        code2wav_graph_odd_tail=True,
                    )

        even_payload = {**payload, "code2wav_graph_odd_tail_setting": "0"}
        with patch.object(
            runner, "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(even_payload), ""),
        ):
            runner.probe_container_runtime(
                "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
            )
        stale_odd_payload = {**even_payload, "code2wav_graph_odd_tail_setting": "1"}
        with patch.object(
            runner, "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(stale_odd_payload), ""),
        ):
            with self.assertRaisesRegex(RuntimeError, "Code2Wav graph family"):
                runner.probe_container_runtime(
                    "/repo", "gemma4-omni:latest", code2wav_graph_family=True,
                )

    def test_code2wav_graph_family_health_requires_two_complete_captured_families(self) -> None:
        record = runner.code2wav_graph_family_record(_code2wav_graph_family_health(), requested=True)
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["slots"], 2)
        self.assertEqual(record["graphs_per_slot"], 37)

        for health in (
            _controller_guided_health(),
            {
                **_code2wav_graph_family_health(),
                "controller_guided_talk_context_pool": {
                    **_code2wav_graph_family_health()["controller_guided_talk_context_pool"],
                    "capacity": 1,
                },
            },
            {
                **_code2wav_graph_family_health(),
                "controller_guided_talk_context_pool": {
                    **_code2wav_graph_family_health()["controller_guided_talk_context_pool"],
                    "memory": {
                        "code2wav_graph_family_enabled": False,
                        "code2wav_graph_odd_tail_enabled": False,
                        "code2wav_graph_capture_status_by_slot": [],
                    },
                },
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "Code2Wav graph family"):
                runner.code2wav_graph_family_record(health, requested=True)

        self.assertEqual(
            runner.code2wav_graph_family_record(_controller_guided_health(), requested=False)["status"],
            "not_requested",
        )

    def test_code2wav_graph_health_attests_exact_odd_tail_family_or_fails_closed(self) -> None:
        record = runner.code2wav_graph_family_record(
            _code2wav_graph_odd_tail_health(), requested=True, odd_tail_requested=True,
        )
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["graphs_per_slot"], 74)
        self.assertEqual(record["window_frames"], list(range(1, 75)))
        for bad_health in (
            _code2wav_graph_family_health(),
            {
                **_code2wav_graph_odd_tail_health(),
                "controller_guided_talk_context_pool": {
                    **_code2wav_graph_odd_tail_health()["controller_guided_talk_context_pool"],
                    "memory": {
                        **_code2wav_graph_odd_tail_health()["controller_guided_talk_context_pool"]["memory"],
                        "code2wav_graph_capture_status_by_slot": [
                            {
                                "captured": True,
                                "graph_count": 73,
                                "window_frames": list(range(1, 74)),
                                "capture_error": None,
                            }
                            for _ in range(2)
                        ],
                    },
                },
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "Code2Wav graph family"):
                runner.code2wav_graph_family_record(
                    bad_health, requested=True, odd_tail_requested=True,
                )

    def test_code2wav_graph_health_requires_explicit_null_capture_error(self) -> None:
        health = _code2wav_graph_odd_tail_health()
        slots = health["controller_guided_talk_context_pool"]["memory"][
            "code2wav_graph_capture_status_by_slot"
        ]
        del slots[0]["capture_error"]
        with self.assertRaisesRegex(RuntimeError, "Code2Wav graph family"):
            runner.code2wav_graph_family_record(
                health, requested=True, odd_tail_requested=True,
            )

    def test_runtime_preflight_rejects_native_q1_graph_disabled(self) -> None:
        native_talker = {
            "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
            "size_bytes": 123,
            "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
            "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
            "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
        }
        payload = {
            "python": "3.11.9",
            "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
            "native_talker": native_talker,
            "talker_native_q1_setting": "1",
            "talker_native_q1_graph_setting": "0",
        }
        with patch.object(
            runner,
            "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), ""),
        ):
            with self.assertRaisesRegex(RuntimeError, "native Q1 graph.*required"):
                runner.probe_container_runtime("/repo", "gemma4-omni:latest")

    def test_runtime_preflight_rejects_old_abi2_shadow_library(self) -> None:
        old_shadow = {
            "path": "/workspace/.tmp/native_qwen_audio_q1/libqwen_talker_audio_q1.so",
            "size_bytes": 123,
            "sha256": "6cc2f75e22b4d5efc73eb7d22071b61addcb79c08d71eded35a2e7845a81789e",
            "native_abi_version": 2,
            "trace_abi_version": 3,
        }
        payload = {
            "python": "3.11.9",
            "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
            "native_talker": old_shadow,
        }
        with patch.object(
            runner, "_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), ""),
        ):
            with self.assertRaisesRegex(RuntimeError, "native Talker.*library|ABI"):
                runner.probe_container_runtime("/repo", "gemma4-omni:latest")

    def test_cli_smoke_argv_matches_server_and_client_import_environments(self) -> None:
        server = runner.build_cli_smoke_argv(
            "/repo", "gemma4-omni:latest", "duplex_kernel.launch",
            module=True,
        )
        client = runner.build_cli_smoke_argv(
            "/repo", "gemma4-omni:latest", "client/duplex_stream_e2e.py",
            workspace_pythonpath=True,
        )

        self.assertEqual(server[server.index("--entrypoint") + 1], "/workspace/.venv_e2e_runtime/bin/python")
        self.assertNotIn("PYTHONPATH=/workspace", server)
        self.assertEqual(server[-3:], ["-m", "duplex_kernel.launch", "--help"])
        self.assertEqual(client[client.index("--entrypoint") + 1], "/workspace/.venv_e2e_runtime/bin/python")
        self.assertEqual(client[client.index("--env") + 1], "PYTHONPATH=/workspace")
        self.assertEqual(client[-2:], ["client/duplex_stream_e2e.py", "--help"])
        self.assertNotIn("--gpus", server + client)

    def test_cli_smoke_requires_both_real_entrypoints_to_print_usage(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, "usage: launch.py [-h]\n", ""),
            subprocess.CompletedProcess([], 0, "usage: duplex_stream_e2e.py [-h]\n", ""),
        ]
        with patch.object(runner, "_command", side_effect=results) as command:
            observed = runner.probe_cli_entrypoints("/repo", "gemma4-omni:latest")

        self.assertEqual(command.call_count, 2)
        self.assertEqual(observed["server"]["usage"], "usage: launch.py [-h]")
        self.assertEqual(observed["client"]["usage"], "usage: duplex_stream_e2e.py [-h]")

        with patch.object(
            runner, "_command",
            side_effect=[
                subprocess.CompletedProcess([], 0, "usage: launch.py [-h]\n", ""),
                subprocess.CompletedProcess([], 1, "", "ModuleNotFoundError: client"),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "client CLI smoke failed.*ModuleNotFoundError"):
                runner.probe_cli_entrypoints("/repo", "gemma4-omni:latest")

    def test_server_and_client_argv_match_the_pair_contract(self) -> None:
        server = runner.build_server_argv(
            bundle_path="/workspace/artifacts/verified-bundle",
        )
        client = runner.build_client_argv(
            output_dir="artifacts/pair_duplex_e2e_proof_test",
        )

        self.assertIn("--enable-gemma-native-control6", server)
        self.assertEqual(
            server[:3],
            ["/workspace/.venv_e2e_runtime/bin/python", "-m", "duplex_kernel.launch"],
        )
        self.assertIn("--native-control6-adapter", server)
        self.assertIn("--gemma-qlora-adapter-dir", server)
        self.assertEqual(server[server.index("--thinker-prefill-chunk-tokens") + 1], "16")
        self.assertEqual(server[server.index("--thinker-audio-prefill-chunk-tokens") + 1], "0")
        self.assertEqual(server[server.index("--thinker-compile-mode") + 1], "default")
        self.assertIn("--duplex-response-bundle-dir", server)
        self.assertIn("/workspace/artifacts/verified-bundle", server)
        self.assertIn("--ablate-response-lora", server)
        self.assertIn("--controller-guided-talk-context-pool", server)
        self.assertIn("--enforce-v9-training-voice", server)
        self.assertIn("--runtime-backend", server)
        self.assertIn("native", server)

        self.assertEqual(client[:3], ["/workspace/.venv_e2e_runtime/bin/python", "client/duplex_stream_e2e.py", "--mode"])
        self.assertEqual(client[3], "pair")
        self.assertEqual(client[client.index("--ws-url") + 1], "ws://127.0.0.1:8092/v1/realtime?duplex=1")
        self.assertEqual(client[client.index("--audio") + 1], "artifacts/scenario_audio_qwen3tts/mode_b_kickoff.wav")
        self.assertEqual(client[client.index("--turn-policy") + 1], "gemma_native_control6")
        self.assertEqual(client[client.index("--max-turns") + 1], "4")
        self.assertEqual(client[client.index("--max-duration-s") + 1], "60")
        self.assertEqual(client[client.index("--output-dir") + 1], "artifacts/pair_duplex_e2e_proof_test")

    def test_effective_route_attestation_is_separate_and_rejects_wrong_protocol(self) -> None:
        server = runner.build_server_argv(bundle_path="/workspace/artifacts/verified-bundle")
        route = runner.effective_route_record(_controller_guided_health(), server)

        self.assertEqual(route["response_protocol"], "controller_guided_say_v1")
        self.assertTrue(route["direct_audio_candidate"])
        self.assertEqual(route["native_action_selection"], "controller_authoritative")
        self.assertEqual(route["frame_residual_adapter"]["architecture"], "response_hidden_q1_query_film_v2")
        self.assertEqual(route["response_lora"], "ablated")
        self.assertEqual(route["thinker_compile_mode"], "default")
        self.assertEqual(route["talker_q1_backend"], "native_audio_q1")
        self.assertTrue(route["audio_history_archive_defer_offline"])
        with self.assertRaisesRegex(RuntimeError, "unexpected effective response protocol"):
            runner.effective_route_record(
                {**_controller_guided_health(), "response_protocol": "audio_response_v2"}, server,
            )
        with self.assertRaisesRegex(RuntimeError, "native_audio_q1 Talker backend"):
            runner.effective_route_record(
                {**_controller_guided_health(), "talker_q1_backend": "eager"}, server,
            )
        with self.assertRaisesRegex(RuntimeError, "offline archive ASR deferral"):
            runner.effective_route_record(
                {**_controller_guided_health(), "audio_history_archive_defer_offline": False}, server,
            )
        with self.assertRaisesRegex(RuntimeError, "disable automatic CPU archive ASR"):
            runner.effective_route_record(
                {**_controller_guided_health(), "audio_history_archive_cpu_startup_available": True}, server,
            )

    def test_native_talker_postrun_gate_requires_native_calls_per_audible_response(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            report_path.write_text(json.dumps({
                "responses": [{
                    "response_id": "resp-native",
                    "terminal_event": "response.done",
                    "audio_samples": 1280,
                    "runtime_metrics": {"talker": {
                        "talker_q1_backend": "native_audio_q1",
                        "native_talker_q1_calls": 1,
                    }},
                }],
            }), encoding="utf-8")
            passed = runner.native_talker_postrun_attestation(report_path)
            self.assertEqual(passed["status"], "PASS")

            report_path.write_text(json.dumps({
                "responses": [{
                    "response_id": "resp-eager",
                    "terminal_event": "response.done",
                    "audio_samples": 1280,
                    "runtime_metrics": {"talker": {
                        "talker_q1_backend": "eager",
                        "native_talker_q1_calls": 0,
                    }},
                }],
            }), encoding="utf-8")
            failed = runner.native_talker_postrun_attestation(report_path)
            self.assertEqual(failed["status"], "FAIL")
            self.assertFalse(failed["responses"][0]["pass"])

    def test_archive_defer_postrun_gate_requires_both_peer_acks_and_deferred_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            peer_a = {"speaker": "A", "session_id": "session-a", "session_updated": {
                "audio_history_archive_defer_offline": True,
                "audio_history_archive_cpu_enabled": False,
            }}
            peer_b = {"speaker": "B", "session_id": "session-b", "session_updated": {
                "audio_history_archive_defer_offline": True,
                "audio_history_archive_cpu_enabled": False,
            }}
            report_path.write_text(json.dumps({
                "participants": [peer_a, peer_b],
                "responses": [{
                    "response_id": "resp-deferred",
                    "runtime_metrics": {"audio_history_archive": {
                        "source": "post_response_asr",
                        "archive_id": 7,
                        "status": "deferred_offline",
                        "transcript_committed": False,
                    }},
                }],
            }), encoding="utf-8")
            self.assertEqual(
                runner.archive_defer_postrun_attestation(report_path)["status"], "PASS",
            )

            invalid_participants = (
                [peer_a, peer_a],
                [peer_a],
                [peer_a, {**peer_b, "session_id": "session-a"}],
                [peer_a, {"speaker": "B", "session_id": "session-b"}],
            )
            for participants in invalid_participants:
                report_path.write_text(json.dumps({
                    "participants": participants,
                    "responses": [{
                        "response_id": "resp-deferred",
                        "runtime_metrics": {"audio_history_archive": {
                            "source": "post_response_asr",
                            "archive_id": 7,
                            "status": "deferred_offline",
                            "transcript_committed": False,
                        }},
                    }],
                }), encoding="utf-8")
                self.assertEqual(
                    runner.archive_defer_postrun_attestation(report_path)["status"], "FAIL",
                )

            report_path.write_text(json.dumps({
                "participants": [peer_a, peer_b],
                "responses": [{
                    "response_id": "resp-submitted",
                    "runtime_metrics": {"audio_history_archive": {
                        "source": "post_response_asr",
                        "archive_id": 8,
                        "status": "submitted",
                        "transcript_committed": False,
                    }},
                }],
            }), encoding="utf-8")
            self.assertEqual(
                runner.archive_defer_postrun_attestation(report_path)["status"], "FAIL",
            )

    def test_run_client_adds_workspace_import_path_without_changing_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            client_argv = runner.build_client_argv(output_dir="artifacts/pair_duplex_e2e_proof_test")
            log_path = Path(temp_dir) / "client.log"
            with patch.object(
                runner,
                "_command",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as command:
                runner.run_client(Path("/repo"), "task-container-id", client_argv, log_path)

            actual = command.call_args.args[0]
            self.assertEqual(
                actual[:5],
                ["docker", "exec", "--env", "PYTHONPATH=/workspace", "task-container-id"],
            )
            self.assertEqual(actual[5:], client_argv)

    def test_docker_argv_is_scoped_and_uses_explicit_port_mapping(self) -> None:
        argv = runner.build_docker_run_argv(
            repository_root="/home/administrator/gemma-4-omni",
            container_name="pair-duplex-proof-unique",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/artifacts/verified-bundle"),
        )

        self.assertIn("--user", argv)
        self.assertEqual(argv[argv.index("--user") + 1], "1000:1000")
        self.assertEqual(argv[argv.index("--ipc") + 1], "host")
        self.assertEqual(argv[argv.index("--volume") + 1], "/home/administrator/gemma-4-omni:/workspace")
        self.assertEqual(argv[argv.index("--publish") + 1], "127.0.0.1:8092:8092")
        self.assertEqual(
            argv[argv.index("--env", argv.index("--gpus")) + 1],
            "DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU=0",
        )
        self.assertIn("DUPLEX_AUDIO_HISTORY_ARCHIVE_DEFER=1", argv)
        self.assertIn("HF_HUB_OFFLINE=1", argv)
        self.assertIn("TRANSFORMERS_OFFLINE=1", argv)
        self.assertIn("HF_DATASETS_OFFLINE=1", argv)
        self.assertNotIn("--network", argv)
        self.assertNotIn("--privileged", argv)
        self.assertIn("--pull=never", argv)
        self.assertIn("pair-duplex-proof-unique", argv)
        self.assertEqual(argv[-len(runner.build_server_argv(bundle_path="/workspace/artifacts/verified-bundle")):],
                         runner.build_server_argv(bundle_path="/workspace/artifacts/verified-bundle"))

    def test_readiness_requires_load_marker_and_live_engine_health(self) -> None:
        marker = "[launch] load engine: OK in 1.0s"
        self.assertFalse(runner.is_server_ready("loading", {"status": "ok", "engine_loaded": True}))
        self.assertFalse(runner.is_server_ready("[launch] load engine: FAIL", {"status": "ok", "engine_loaded": True}))
        self.assertFalse(runner.is_server_ready(marker, {"status": "ok", "engine_loaded": False}))
        self.assertTrue(runner.is_server_ready(marker, {"status": "ok", "engine_loaded": True}))

    def test_readiness_fails_fast_when_owned_server_container_exits(self) -> None:
        state = {"Status": "exited", "Running": False, "ExitCode": 1}
        inspect_result = subprocess.CompletedProcess(
            ["docker", "inspect"], 0, json.dumps(state), ""
        )
        with (
            patch.object(runner, "_command", return_value=inspect_result) as command,
            patch.object(runner, "_health", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, r"container exited.*ExitCode=1"):
                runner.wait_for_server_ready(
                    Path("/repo"), "task-container-id", Path("/tmp/server.log"),
                    timeout_s=10, poll_interval_s=0.01,
                )
        command.assert_called_once()

    def test_failure_after_pause_cleans_only_owned_container_and_restores_original_id(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root),
                "--output-dir", str(output),
                "--maintenance-container", "gemma4-speech-runtime-query-film-v2-20261009",
            ])
            cleanup_calls: list[tuple[str, str]] = []

            with (
                patch.object(runner, "collect_run_metadata", return_value={"git": {"sha": "abc"}, "inputs": {}}),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value={"python": "3.11", "packages": {"torch": "2.0", "transformers": "5.0"}, **_scheduler_anchor_preflight()}),
                patch.object(runner, "probe_cli_entrypoints", return_value={"server": {"usage": "usage: launch.py"}, "client": {"usage": "usage: client.py"}}),
                patch.object(runner, "snapshot_maintenance_container", return_value={"id": "managed-container-id", "was_running": True, "inspect": {}}),
                patch.object(runner, "pause_maintenance_container", return_value=None),
                patch.object(runner, "wait_for_gpu_vram_release", return_value={
                    "released": True, "required_free_mib": runner.GPU_RELEASE_MIN_FREE_MIB,
                    "observations": [{"container_id": "managed-container-id", "container_running": False,
                                      "memory": {"gpus": [{"index": 0, "memory_free_mib": 24000}], "compute_apps": []}}],
                }),
                patch.object(runner, "launch_server_container", return_value="task-container-id"),
                patch.object(runner, "start_server_log_capture", return_value=None),
                patch.object(runner, "wait_for_server_ready", side_effect=RuntimeError("load engine marker missing")),
                patch.object(runner, "cleanup_owned_container", side_effect=lambda cid, _name, _root: cleanup_calls.append(("remove", cid))),
                patch.object(runner, "restore_maintenance_container", side_effect=lambda snapshot, **_kw: cleanup_calls.append(("restore", snapshot["id"]))),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            self.assertEqual(cleanup_calls, [("remove", "task-container-id"), ("restore", "managed-container-id")])
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("load engine marker missing", manifest["error"])
            self.assertEqual(manifest["maintenance"]["container_id"], "managed-container-id")
            self.assertEqual(manifest["container"]["id"], "task-container-id")

    def test_run_proof_passes_listener_sidecar_dir_to_real_container_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root), "--output-dir", str(output),
                "--native-listener-capacity-candidate",
                "--code2wav-graph-family", "--code2wav-graph-odd-tail",
                "--first-input-scheduler-anchor",
            ])
            runtime = {
                "python": "3.11",
                "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
                "native_listener_candidate": _native_listener_candidate_evidence(),
                "native_listener_cuda_profile_setting": "0",
                "native_listener_cuda_profile_directory_setting": None,
                "native_listener_cuda_profile_enabled": False,
                "listener_model_lock_host_timeline_enabled": False,
                "listener_pair_lock_handoff_setting": "0",
                "listener_pair_lock_handoff_enabled": False,
                "first_input_scheduler_anchor_setting": "1",
                "first_input_scheduler_anchor_enabled": True,
                "ready_frame_scheduler_dispatch_setting": "0",
                "ready_frame_scheduler_dispatch_enabled": False,
            }
            completed = subprocess.CompletedProcess([], 0, "task-container-id\n", "")
            with (
                patch.object(runner, "collect_run_metadata", return_value={
                    "git": {"sha": "abc"}, "inputs": {},
                    "response_bundle_manifest": {"metadata": {"bundle_tag": "fixture"}},
                }),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value=runtime) as preflight,
                patch.object(runner, "probe_cli_entrypoints", return_value={
                    "server": {"usage": "usage: launch.py"},
                    "client": {"usage": "usage: client.py"},
                }),
                patch.object(runner, "nvidia_smi_memory_snapshot", return_value={
                    "gpus": [{"index": 0, "memory_total_mib": 24576, "memory_used_mib": 512,
                              "memory_free_mib": 24064, "utilization_pct": 0}],
                    "compute_apps": [],
                }),
                patch.object(runner, "_command", return_value=completed) as command,
                patch.object(runner, "launch_server_container", wraps=runner.launch_server_container) as launch,
                patch.object(runner, "start_server_log_capture", return_value=None),
                patch.object(runner, "wait_for_server_ready", return_value={
                    **_code2wav_graph_odd_tail_health(),
                    "first_input_scheduler_anchor_enabled": True,
                    "ready_frame_scheduler_dispatch_enabled": False,
                    "ready_frame_scheduler_dispatch_sessions": [],
                }),
                patch.object(runner, "record_runtime_versions", return_value=runtime),
                patch.object(runner, "attest_native_talker_binary", return_value={
                    "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                    "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                    "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
                }),
                patch.object(runner, "run_client", return_value=subprocess.CompletedProcess([], 0)),
                patch.object(runner, "native_listener_session_postrun_attestation", return_value={"status": "PASS"}),
                patch.object(runner, "native_talker_postrun_attestation", return_value={"status": "PASS"}),
                patch.object(runner, "archive_defer_postrun_attestation", return_value={"status": "PASS"}),
                patch.object(runner, "run_analyzer", return_value={"status": "PASS", "exit_code": 0}),
                patch.object(runner, "cleanup_owned_container"),
            ):
                rc = runner.run_proof(args)

            expected_dir = f"/workspace/{output.relative_to(root).as_posix()}/listener_attestations"
            if rc != 0:
                manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
                self.fail(f"run_proof failed: {manifest.get('error')}; status={manifest.get('status')}")
            self.assertEqual(launch.call_count, 1)
            self.assertTrue(preflight.call_args.kwargs["first_input_scheduler_anchor"])
            self.assertTrue(launch.call_args.kwargs["first_input_scheduler_anchor"])
            self.assertFalse(preflight.call_args.kwargs["ready_frame_scheduler_dispatch"])
            self.assertFalse(launch.call_args.kwargs["ready_frame_scheduler_dispatch"])
            self.assertEqual(launch.call_args.kwargs["native_listener_attestation_dir"], expected_dir)
            self.assertEqual(command.call_count, 1)
            docker_argv = command.call_args.args[0]
            self.assertIn(f"PAIR_DUPLEX_LISTENER_ATTESTATION_DIR={expected_dir}", docker_argv)
            self.assertIn(
                f"{runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV}=1",
                docker_argv,
            )
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["first_input_scheduler_anchor"], {
                "requested": True,
                "status": "PASS",
                "cpu_preflight_setting": "1",
                "cpu_preflight_enabled": True,
                "cpu_preflight_status": "PASS",
                "actual": True,
            })
            self.assertIn(
                f"PAIR_DUPLEX_LISTENER_ATTESTATION_DIR={expected_dir}",
                manifest["argv"]["docker_run"],
            )

    def test_gpu_release_barrier_failure_prevents_model_launch_and_restores_managed_service(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root), "--output-dir", str(output),
                "--maintenance-container", runner.DEFAULT_MAINTENANCE_CONTAINER,
            ])
            restore_calls: list[str] = []
            observations = [{
                "container_id": "managed-container-id",
                "container_running": False,
                "memory": {"gpus": [{"index": 0, "memory_free_mib": 6400}],
                           "compute_apps": [{"pid": "101", "process_name": "python",
                                             "used_gpu_memory_mib": 17500}]},
                "blocker": "managed GPU process is still resident after stop",
            }]
            with (
                patch.object(runner, "collect_run_metadata", return_value={"git": {"sha": "abc"}, "inputs": {}}),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value={"python": "3.11", "packages": {"torch": "2.0", "transformers": "5.0"}, **_scheduler_anchor_preflight()}),
                patch.object(runner, "probe_cli_entrypoints", return_value={"server": {"usage": "usage: launch.py"}, "client": {"usage": "usage: client.py"}}),
                patch.object(runner, "snapshot_maintenance_container", return_value={"id": "managed-container-id", "was_running": True, "inspect": {}}),
                patch.object(runner, "pause_maintenance_container", return_value={"gpus": [{"index": 0, "memory_used_mib": 18000}], "compute_apps": []}),
                patch.object(runner, "wait_for_gpu_vram_release", side_effect=runner.GpuVramBarrierError(
                    "managed GPU process is still resident after stop", observations,
                )),
                patch.object(runner, "launch_server_container") as launch,
                patch.object(runner, "restore_maintenance_container", side_effect=lambda snapshot, **_kw: restore_calls.append(snapshot["id"])),
                patch.object(runner, "run_analyzer", return_value={"status": "INCOMPLETE"}),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            launch.assert_not_called()
            self.assertEqual(restore_calls, ["managed-container-id"])
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["maintenance"]["gpu_release_barrier"]["status"], "blocked")
            self.assertEqual(manifest["maintenance"]["gpu_release_barrier"]["observations"], observations)

    def test_fused_readback_cuda_smoke_uses_isolated_offline_container_and_binds_report(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository_root) as temp_dir:
            root = Path(temp_dir)
            script = root / runner.NATIVE_CONTROL6_FUSED_READBACK_SMOKE_RELATIVE_PATH
            script.parent.mkdir(parents=True)
            script.write_text("# test fixture\n", encoding="utf-8")
            output = root / "artifacts" / "smoke.json"
            argv = runner.build_native_control6_fused_readback_cuda_smoke_argv(
                root, "gemma4-omni:test", container_name="owned-fused-smoke-test",
                output_path=output,
            )
            for required in (
                "--network", "none", "--user", "1000:1000", "--ipc", "host", "--gpus", "all",
                "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
                "--pull=never", "/workspace/.venv_e2e_runtime/bin/python",
            ):
                self.assertIn(required, argv)
            self.assertNotIn("--privileged", argv)

            report = {
                "status": "PASS", "torch_version": "2.12.1+cu128",
                "finite_comparisons": 126, "nonfinite_failures_compared": 63,
                "per_successful_step_cuda_readbacks": {
                    "baseline": {"bool": 2, "item": 3, "cpu": 0},
                    "fused": {"bool": 1, "item": 0, "cpu": 1},
                },
            }
            cleanup_names: list[str] = []

            def fake_command(command_argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(kwargs["timeout"], runner.NATIVE_CONTROL6_FUSED_READBACK_SMOKE_TIMEOUT_S)
                output_container_path = command_argv[command_argv.index("--output") + 1]
                output_local_path = root / output_container_path.removeprefix("/workspace/")
                output_local_path.parent.mkdir(parents=True, exist_ok=True)
                output_local_path.write_text(json.dumps(report), encoding="utf-8")
                return subprocess.CompletedProcess(command_argv, 0, "smoke complete\n", "")

            with (
                patch.object(runner, "_command", side_effect=fake_command),
                patch.object(runner, "cleanup_owned_container", side_effect=lambda _cid, name, _root: cleanup_names.append(name)),
            ):
                result = runner.run_native_control6_fused_readback_cuda_smoke(
                    root, "gemma4-omni:test", output,
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["report"]["finite_comparisons"], 126)
            self.assertEqual(len(result["report_sha256"]), 64)
            self.assertEqual(len(cleanup_names), 1)
            self.assertTrue(cleanup_names[0].startswith("control6-fused-smoke-"))

    def test_fused_readback_smoke_runs_after_pause_and_before_server_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "fused-smoke-order"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root), "--output-dir", str(output),
                "--maintenance-container", runner.DEFAULT_MAINTENANCE_CONTAINER,
                "--native-control6-fused-readback",
            ])
            native_talker = {
                "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
                "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
                "size_bytes": 123,
                "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
            }
            runtime = {
                "python": "3.11",
                "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
                "native_talker": native_talker,
                "talker_native_q1_setting": "1",
                "talker_native_q1_graph_setting": "1",
                "native_control6_trace_setting": "0",
                **_scheduler_anchor_preflight(),
                "native_control6_fused_readback_setting": "1",
                "native_control6_fused_readback_enabled": True,
            }
            restore_calls: list[str] = []
            operation_order: list[str] = []
            barrier_result = {
                "released": True,
                "required_free_mib": runner.GPU_RELEASE_MIN_FREE_MIB,
                "observations": [{
                    "container_id": "managed-container-id", "container_running": False,
                    "memory": {"gpus": [{"index": 0, "memory_free_mib": 24000}], "compute_apps": []},
                }],
            }
            with (
                patch.object(runner, "collect_run_metadata", return_value={"git": {"sha": "abc"}, "inputs": {}}),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value=runtime),
                patch.object(runner, "probe_cli_entrypoints", return_value={
                    "server": {"usage": "usage: launch.py"}, "client": {"usage": "usage: client.py"},
                }),
                patch.object(runner, "snapshot_maintenance_container", return_value={
                    "id": "managed-container-id", "was_running": True, "inspect": {},
                }),
                patch.object(runner, "pause_maintenance_container", return_value=None),
                patch.object(runner, "wait_for_gpu_vram_release", side_effect=lambda *_a, **_k: operation_order.append("barrier") or barrier_result),
                patch.object(runner, "run_native_control6_fused_readback_cuda_smoke", create=True,
                             side_effect=lambda *_a, **_k: operation_order.append("smoke") or {
                                 "status": "PASS", "finite_comparisons": 126,
                                 "nonfinite_failures_compared": 63,
                             }) as smoke,
                patch.object(runner, "launch_server_container", side_effect=lambda *_a, **_k: operation_order.append("launch") or "task-id") as launch,
                patch.object(runner, "start_server_log_capture", return_value=None),
                patch.object(runner, "wait_for_server_ready", side_effect=RuntimeError("unexpected launch")),
                patch.object(runner, "cleanup_owned_container"),
                patch.object(runner, "restore_maintenance_container", side_effect=lambda snapshot, **_kw: restore_calls.append(snapshot["id"])),
                patch.object(runner, "run_analyzer", return_value={"status": "INCOMPLETE"}),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            smoke.assert_called_once()
            launch.assert_called_once()
            self.assertEqual(operation_order, ["barrier", "smoke", "barrier", "launch"])
            self.assertEqual(restore_calls, ["managed-container-id"])
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["native_control6_fused_readback"]["cuda_smoke"]["status"], "PASS")

    def test_fused_readback_smoke_failure_blocks_server_and_restores_managed_service(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "fused-smoke-failure"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root), "--output-dir", str(output),
                "--maintenance-container", runner.DEFAULT_MAINTENANCE_CONTAINER,
                "--native-control6-fused-readback",
            ])
            native_talker = {
                "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
                "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
                "size_bytes": 123,
                "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
            }
            runtime = {
                "python": "3.11",
                "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
                "native_talker": native_talker,
                "talker_native_q1_setting": "1",
                "talker_native_q1_graph_setting": "1",
                "native_control6_trace_setting": "0",
                **_scheduler_anchor_preflight(),
                "native_control6_fused_readback_setting": "1",
                "native_control6_fused_readback_enabled": True,
            }
            restore_calls: list[str] = []
            barrier_calls: list[str] = []
            barrier_result = {
                "released": True,
                "required_free_mib": runner.GPU_RELEASE_MIN_FREE_MIB,
                "observations": [{
                    "container_id": "managed-container-id", "container_running": False,
                    "memory": {"gpus": [{"index": 0, "memory_free_mib": 24000}], "compute_apps": []},
                }],
            }
            with (
                patch.object(runner, "collect_run_metadata", return_value={"git": {"sha": "abc"}, "inputs": {}}),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value=runtime),
                patch.object(runner, "probe_cli_entrypoints", return_value={
                    "server": {"usage": "usage: launch.py"}, "client": {"usage": "usage: client.py"},
                }),
                patch.object(runner, "snapshot_maintenance_container", return_value={
                    "id": "managed-container-id", "was_running": True, "inspect": {},
                }),
                patch.object(runner, "pause_maintenance_container", return_value=None),
                patch.object(runner, "wait_for_gpu_vram_release", side_effect=lambda *_a, **_k: barrier_calls.append("barrier") or barrier_result),
                patch.object(runner, "run_native_control6_fused_readback_cuda_smoke", create=True,
                             side_effect=RuntimeError("synthetic CUDA parity failure")) as smoke,
                patch.object(runner, "launch_server_container") as launch,
                patch.object(runner, "restore_maintenance_container", side_effect=lambda snapshot, **_kw: restore_calls.append(snapshot["id"])),
                patch.object(runner, "run_analyzer", return_value={"status": "INCOMPLETE"}),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            smoke.assert_called_once()
            launch.assert_not_called()
            self.assertEqual(barrier_calls, ["barrier"])
            self.assertEqual(restore_calls, ["managed-container-id"])
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["native_control6_fused_readback"]["cuda_smoke"]["status"], "FAIL")
            self.assertIn("synthetic CUDA parity failure", manifest["native_control6_fused_readback"]["cuda_smoke"]["error"])

    def test_pause_rechecks_idle_state_immediately_before_stopping_exact_id(self) -> None:
        snapshot = {"id": "managed-container-id", "was_running": True}
        with (
            patch.object(runner, "_assert_maintenance_idle", side_effect=RuntimeError("active peer detected")) as idle_check,
            patch.object(runner, "_command") as command,
        ):
            with self.assertRaisesRegex(RuntimeError, "active peer detected"):
                runner.pause_maintenance_container(snapshot, Path("/repo"))

        idle_check.assert_called_once_with("managed-container-id", Path("/repo"))
        command.assert_not_called()

    def test_proc_tcp_parser_finds_only_established_port_8000_for_ipv4_and_ipv6(self) -> None:
        header = "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        ipv4 = header + (
            "  0: 0100007F:1F40 0200007F:C350 01 00000000:00000000 00:00000000 00000000 1000 0 10 1\n"
            "  1: 0100007F:1F40 0200007F:C350 06 00000000:00000000 00:00000000 00000000 1000 0 11 1\n"
            "  2: 0100007F:1F41 0200007F:C350 01 00000000:00000000 00:00000000 00000000 1000 0 12 1\n"
        )
        ipv6 = header + (
            "  0: 00000000000000000000000001000000:1F40 00000000000000000000000002000000:C350 01 00000000:00000000 00:00000000 00000000 1000 0 13 1\n"
        )

        active = runner.parse_proc_net_tcp_tables({"tcp": ipv4, "tcp6": ipv6})

        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["table"], "tcp")
        self.assertEqual(active[1]["table"], "tcp6")

    def test_proc_tcp_parser_fails_closed_on_malformed_rows(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "malformed /proc/net/tcp"):
            runner.parse_proc_net_tcp_tables({"tcp": "header\ninvalid-row\n", "tcp6": None})

    def test_idle_gate_probes_proc_inside_container_without_container_ss(self) -> None:
        calls: list[list[str]] = []

        def fake_command(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(argv))
            if argv[0] == "ss":
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[:2] == ["docker", "exec"]:
                header = "sl local_address rem_address st\n"
                tables = {"tcp": header, "tcp6": header}
                return subprocess.CompletedProcess(argv, 0, json.dumps(tables), "")
            if argv[:2] == ["nvidia-smi", "--query-compute-apps=pid"]:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[:2] == ["docker", "top"]:
                return subprocess.CompletedProcess(argv, 0, "PID COMMAND\n1 python\n", "")
            if argv[:2] == ["nvidia-smi", "--query-gpu=utilization.gpu"]:
                return subprocess.CompletedProcess(argv, 0, "0\n", "")
            if argv[:2] == ["ps", "-eo"]:
                return subprocess.CompletedProcess(argv, 0, "", "")
            raise AssertionError(f"unexpected command: {argv}")

        with patch.object(runner, "_command", side_effect=fake_command):
            runner._assert_maintenance_idle("managed-container-id", Path("/repo"))

        self.assertEqual(calls[1][0:4], [
            "docker", "exec", "managed-container-id", "/workspace/.venv_e2e_runtime/bin/python",
        ])
        self.assertIn("/proc/net", calls[1][-1])
        self.assertIn("tcp6", calls[1][-1])

    def test_cleanup_failure_after_capture_records_cleanup_failed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args(["--repo-root", str(root), "--output-dir", str(output)])
            rc = self._run_successful_client(
                args,
                cleanup_side_effect=RuntimeError("owned container cleanup failed"),
                analyzer_result={"status": "PASS", "exit_code": 0},
            )

            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertEqual(manifest["status"], "cleanup_failed")
            self.assertEqual(manifest["status_before_cleanup_failure"], "captured")
            self.assertEqual(
                manifest["effective_route"]["response_protocol"], "controller_guided_say_v1",
            )
            self.assertEqual(manifest["response_bundle_manifest"]["metadata"]["bundle_tag"], "fixture")
            self.assertNotEqual(manifest["effective_route"], manifest["response_bundle_manifest"])
            self.assertIn("cleanup failed", manifest["cleanup_errors"][0])

    def test_missing_analyzer_cannot_mark_capture_complete(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args(["--repo-root", str(root), "--output-dir", str(output)])
            rc = self._run_successful_client(args)

            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertEqual(manifest["analysis"]["status"], "INCOMPLETE")
            self.assertEqual(manifest["analysis"]["legacy_analyzer_status"], "not_installed")
            self.assertTrue((output / "physical_runtime_gate.json").is_file())
            self.assertTrue((output / "pair_duplex_certification_report.md").is_file())
            self.assertEqual(manifest["status"], "analysis_failed")

    def test_skipping_analyzer_leaves_capture_uncertified(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root), "--output-dir", str(output), "--skip-analyzer",
            ])
            rc = self._run_successful_client(args)

            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertEqual(manifest["analysis"]["status"], "skipped")
            self.assertEqual(manifest["status"], "captured")

    def test_client_failure_still_runs_analyzer_and_final_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
            root = Path(temp_dir)
            output = root / "artifacts" / "proof"
            args = runner.build_parser().parse_args(["--repo-root", str(root), "--output-dir", str(output)])
            analyzer_calls: list[Path] = []
            cleanup_calls: list[str] = []

            with (
                patch.object(runner, "collect_run_metadata", return_value={"git": {"sha": "abc"}, "inputs": {}}),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value={"python": "3.11", "packages": {"torch": "2.0", "transformers": "5.0"}, **_scheduler_anchor_preflight()}),
                patch.object(runner, "probe_cli_entrypoints", return_value={"server": {"usage": "usage: launch.py"}, "client": {"usage": "usage: client.py"}}),
                patch.object(runner, "nvidia_smi_memory_snapshot", return_value={
                    "gpus": [{"index": 0, "memory_total_mib": 24576, "memory_used_mib": 512,
                              "memory_free_mib": 24064, "utilization_pct": 0}],
                    "compute_apps": [],
                }),
                patch.object(runner, "launch_server_container", return_value="task-container-id"),
                patch.object(runner, "start_server_log_capture", return_value=None),
                patch.object(runner, "wait_for_server_ready", return_value=_controller_guided_health()),
                patch.object(runner, "record_runtime_versions", return_value={"python": "3.11"}),
                patch.object(runner, "attest_native_talker_binary", return_value={
                    "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
                    "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
                    "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
                }),
                patch.object(runner, "run_client", return_value=subprocess.CompletedProcess([], 3)),
                patch.object(runner, "run_analyzer", side_effect=lambda _root, out: analyzer_calls.append(out) or {"status": "INCOMPLETE"}),
                patch.object(runner, "cleanup_owned_container", side_effect=lambda cid, _name, _root: cleanup_calls.append(cid)),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            self.assertEqual(analyzer_calls, [output])
            self.assertEqual(cleanup_calls, ["task-container-id"])
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "client_failed")
            self.assertEqual(manifest["client_exit_code"], 3)
            self.assertEqual(manifest["analysis"]["status"], "INCOMPLETE")
            self.assertEqual(manifest["git_commit"], "abc")
            self.assertEqual(manifest["python_version"], "3.11")
            self.assertEqual(manifest["packages"]["torch"], "2.0")
            self.assertEqual(manifest["server_port"], 8092)
            self.assertEqual(manifest["turn_policy"], "gemma_native_control6")
            self.assertIs(manifest["action_steering_enabled"], False)
            self.assertEqual(set(manifest["checkpoints"]), set(runner.ANALYZER_CHECKPOINTS))

    def test_preflight_failure_still_runs_host_analyzer_and_writes_report(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository_root) as temp_dir:
            output = Path(temp_dir) / "proof"
            args = runner.build_parser().parse_args([
                "--repo-root", str(repository_root), "--output-dir", str(output),
            ])
            with patch.object(runner, "collect_run_metadata", side_effect=FileNotFoundError("checkpoint absent")):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "certification_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("checkpoint absent", manifest["error"])
            self.assertIn(manifest["analysis"]["status"], {"INCOMPLETE", "FAIL"})
            self.assertEqual(manifest["analysis"]["legacy_analyzer_status"], metrics["status"])
            self.assertEqual(
                manifest["analysis"]["report_path"],
                str(output / "pair_duplex_certification_report.md"),
            )
            self.assertEqual(
                manifest["analysis"]["physical_gate_path"],
                str(output / "physical_runtime_gate.json"),
            )
            self.assertEqual(metrics["manifest"]["status"], "INCOMPLETE")
            self.assertIn(
                "**Status: INCOMPLETE**",
                (output / "pair_duplex_certification_report.md").read_text(encoding="utf-8"),
            )

    def test_cli_preflight_failure_does_not_pause_managed_service(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository_root) as temp_dir:
            root = Path(temp_dir)
            output = root / "proof"
            args = runner.build_parser().parse_args([
                "--repo-root", str(root), "--output-dir", str(output),
                "--maintenance-container", runner.DEFAULT_MAINTENANCE_CONTAINER,
            ])
            with (
                patch.object(runner, "collect_run_metadata", return_value={"git": {"sha": "abc"}, "inputs": {}}),
                patch.object(runner, "docker_image_identity", return_value={"id": "sha256:test"}),
                patch.object(runner, "probe_container_runtime", return_value={"python": "3.11", "packages": {"torch": "2.0", "transformers": "5.0"}, **_scheduler_anchor_preflight()}),
                patch.object(runner, "probe_cli_entrypoints", side_effect=RuntimeError("client CLI smoke failed (exit=1): ModuleNotFoundError: client")),
                patch.object(runner, "snapshot_maintenance_container") as snapshot,
                patch.object(runner, "pause_maintenance_container") as pause,
                patch.object(runner, "run_analyzer", return_value={"status": "INCOMPLETE"}),
            ):
                rc = runner.run_proof(args)

            self.assertEqual(rc, 1)
            snapshot.assert_not_called()
            pause.assert_not_called()
            manifest = json.loads((output / "server_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("client CLI smoke failed", manifest["error"])

    def test_host_analyzer_autodiscovers_asr_artifact(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository_root) as temp_dir:
            output = Path(temp_dir)
            (output / "asr_transcripts.json").write_text("{}", encoding="utf-8")
            captured: list[list[str]] = []
            completed = subprocess.CompletedProcess([], 1, stdout="INCOMPLETE", stderr="")
            with patch.object(runner, "_command", side_effect=lambda argv, **_kwargs: captured.append(list(argv)) or completed):
                result = runner.run_analyzer(repository_root, output)

            self.assertIn("--asr-json", captured[0])
            self.assertEqual(captured[0][0], runner.sys.executable)
            self.assertEqual(captured[0][-1], str(output / "asr_transcripts.json"))
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["legacy_analyzer_status"], "failed")
            self.assertEqual(
                result["report_path"],
                str(output / "pair_duplex_certification_report.md"),
            )


if __name__ == "__main__":
    unittest.main()
