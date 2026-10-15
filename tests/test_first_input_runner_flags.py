from tools import run_pair_duplex_proof_3090 as runner


def test_first_input_anchor_is_default_off_and_identical_in_preflight_and_server():
    env = runner.FIRST_INPUT_SCHEDULER_ANCHOR_ENV
    default_preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
    default_server = runner.build_docker_run_argv(
        repository_root="/home/administrator/gemma-4-omni",
        container_name="pair-duplex-default-anchor",
        image="gemma4-omni:latest",
        server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
    )
    enabled_preflight = runner.build_runtime_probe_argv(
        "/repo", "gemma4-omni:latest", first_input_scheduler_anchor=True,
    )
    enabled_server = runner.build_docker_run_argv(
        repository_root="/home/administrator/gemma-4-omni",
        container_name="pair-duplex-enabled-anchor",
        image="gemma4-omni:latest",
        server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
        first_input_scheduler_anchor=True,
    )

    assert f"{env}=0" in default_preflight
    assert f"{env}=0" in default_server
    assert f"{env}=1" in enabled_preflight
    assert f"{env}=1" in enabled_server
    assert "DUPLEX_NATIVE_CONTROL6_TRACE=0" in default_preflight
    assert "DUPLEX_NATIVE_CONTROL6_TRACE=0" in default_server
    assert "DUPLEX_NATIVE_CONTROL6_TRACE=0" in enabled_preflight
    assert "DUPLEX_NATIVE_CONTROL6_TRACE=0" in enabled_server
    assert "--gpus" not in enabled_preflight
    probe = runner.runtime_version_probe_code()
    assert "first_input_scheduler_anchor_setting" in probe
    assert "first_input_scheduler_anchor_enabled" in probe
    assert not runner.build_parser().parse_args([]).first_input_scheduler_anchor
    assert runner.build_parser().parse_args([
        "--first-input-scheduler-anchor"
    ]).first_input_scheduler_anchor
