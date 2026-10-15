from __future__ import annotations

import pytest

from tools import run_pair_duplex_proof_3090 as runner


def test_ready_frame_flag_is_default_off_and_mutually_exclusive_with_anchor() -> None:
    assert runner.build_parser().parse_args([]).ready_frame_scheduler_dispatch is False
    args = runner.build_parser().parse_args(["--ready-frame-scheduler-dispatch"])
    assert args.ready_frame_scheduler_dispatch is True
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.validate_scheduler_mode(
            first_input_scheduler_anchor=True,
            ready_frame_scheduler_dispatch=True,
        )


def test_ready_frame_preflight_and_server_share_all_requested_runtime_settings() -> None:
    ready_env = runner.READY_FRAME_SCHEDULER_DISPATCH_ENV
    preflight = runner.build_runtime_probe_argv(
        "/repo",
        "gemma4-omni:latest",
        code2wav_graph_family=True,
        code2wav_graph_odd_tail=True,
        native_listener_capacity_candidate=True,
        first_input_scheduler_anchor=False,
        ready_frame_scheduler_dispatch=True,
    )
    server = runner.build_docker_run_argv(
        repository_root="/home/administrator/gemma-4-omni",
        container_name="ready-contract-test",
        image="gemma4-omni:latest",
        server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        code2wav_graph_family=True,
        code2wav_graph_odd_tail=True,
        native_listener_capacity_candidate=True,
        native_listener_attestation_dir="/workspace/artifacts/ready-contract-test/listener",
        first_input_scheduler_anchor=False,
        ready_frame_scheduler_dispatch=True,
    )
    for argv in (preflight, server):
        assert f"{ready_env}=1" in argv
        assert f"{runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV}=0" in argv
        assert "DUPLEX_NATIVE_CONTROL6_TRACE=0" in argv
        assert "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1" in argv
        assert "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2" in argv
        assert "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1" in argv
        assert "DUPLEX_TALKER_NATIVE_AUDIO_Q1=1" in argv
        assert "DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH=1" in argv
        assert runner.NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH in " ".join(argv)
    assert "--gpus" not in preflight


def test_ready_frame_health_requires_requested_global_and_each_session_mode() -> None:
    record = runner.ready_frame_scheduler_dispatch_record(
        {
            "ready_frame_scheduler_dispatch_enabled": True,
            "ready_frame_scheduler_dispatch_sessions": [
                {"session_id": "a", "enabled": True},
                {"session_id": "b", "enabled": True},
            ],
        },
        requested=True,
    )
    assert record == {"requested": True, "actual": True, "status": "PASS"}
    with pytest.raises(RuntimeError, match="health differs"):
        runner.ready_frame_scheduler_dispatch_record({}, requested=True)
    with pytest.raises(RuntimeError, match="session mode"):
        runner.ready_frame_scheduler_dispatch_record(
            {
                "ready_frame_scheduler_dispatch_enabled": True,
                "ready_frame_scheduler_dispatch_sessions": [
                    {"session_id": "a", "enabled": False},
                ],
            },
            requested=True,
        )
