from __future__ import annotations

import json
import subprocess


SOURCE = "gemma4-listener-control6-q2cachemeta-build6-phaseonly-20261020"
DEFAULT = "gemma4-listener-control6-control6-restore-default-20261020"
PROFILE = "gemma4-listener-control6-q2profile-build6-20261020"
TRACE = (
    "/workspace/artifacts/conversation_quality_20261020/"
    "control6_q2cache_build6_torch_profile_20261020/listener_cpu_cuda_trace.json"
)


def output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    source = json.loads(output(["docker", "inspect", SOURCE]))[0]
    config = source["Config"]
    host = source["HostConfig"]
    if config.get("User") != "1000:1000":
        raise SystemExit("source container is not appuser")
    if host.get("Privileged") or host.get("NetworkMode") == "host" or host.get("IpcMode") != "host":
        raise SystemExit("source container violates the isolation contract")
    if not any(
        bind.startswith("/home/administrator/gemma-4-omni:/workspace")
        for bind in host.get("Binds") or ()
    ):
        raise SystemExit("source container does not mount the repository at /workspace")
    if subprocess.run(
        ["docker", "inspect", PROFILE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        raise SystemExit("profile container already exists; preserve it and choose another name")

    default_state = json.loads(output(["docker", "inspect", DEFAULT]))[0]["State"]
    if default_state.get("Running"):
        print(output(["docker", "stop", "--time", "20", DEFAULT]), flush=True)

    args = [
        "docker", "run", "--detach", "--name", PROFILE,
        "--user", config["User"], "--ipc", host["IpcMode"],
        "--network", host["NetworkMode"], "--restart", host["RestartPolicy"]["Name"],
        "--workdir", config["WorkingDir"],
    ]
    for option in host.get("SecurityOpt") or ():
        args.extend(["--security-opt", option])
    for bind in host.get("Binds") or ():
        args.extend(["--volume", bind])
    for port, mappings in (host.get("PortBindings") or {}).items():
        for mapping in mappings or ():
            host_ip = mapping.get("HostIp", "")
            host_port = mapping["HostPort"]
            container_port = port.split("/", 1)[0]
            published = (
                f"{host_ip}:{host_port}:{container_port}"
                if host_ip else f"{host_port}:{container_port}"
            )
            args.extend(["--publish", published])
    requests = host.get("DeviceRequests") or ()
    if requests:
        if not any(
            "gpu" in capability
            for request in requests
            for group in (request.get("Capabilities") or ())
            for capability in group
        ):
            raise SystemExit("source GPU request is not recognizable")
        args.extend(["--gpus", "all"])
    for key, value in (config.get("Labels") or {}).items():
        args.extend(["--label", f"{key}={value}"])

    env = {}
    for item in config.get("Env") or ():
        key, separator, value = item.partition("=")
        if separator:
            env[key] = value
    env["DUPLEX_LISTENER_TORCH_PROFILE_TRACE_PATH"] = TRACE
    env["DUPLEX_LISTENER_TORCH_PROFILE_FRAME_IDS"] = "24,25,26"
    for key, value in env.items():
        args.extend(["--env", f"{key}={value}"])
    if config.get("Entrypoint"):
        args.extend(["--entrypoint", config["Entrypoint"][0]])
    args.append(config["Image"])
    args.extend(config.get("Cmd") or ())

    container_id = output(args)
    print(json.dumps({
        "container": PROFILE,
        "id": container_id[:12],
        "image": config["Image"],
        "user": config["User"],
        "ipc": host["IpcMode"],
        "network": host["NetworkMode"],
        "trace": TRACE,
        "frame_ids": env["DUPLEX_LISTENER_TORCH_PROFILE_FRAME_IDS"],
    }, indent=2))


if __name__ == "__main__":
    main()
