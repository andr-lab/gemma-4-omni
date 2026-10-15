#!/usr/bin/env python3
"""Run the bounded two-peer duplex proof on the Linux RTX 3090 host.

Sources: Docker's run reference (https://docs.docker.com/reference/cli/docker/container/run/)
defines ``--ipc=host`` as ``Use the host system's IPC namespace`` and documents
loopback port publication and bind mounts. Its volume section recommends
``--mount`` but says there is no plan to deprecate ``--volume``. Python 3.11's
subprocess reference (https://docs.python.org/3.11/library/subprocess.html#subprocess.run)
says ``Run the command described by args. Wait for command to complete``; this
runner uses argument lists and the documented default ``shell=False``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request
import uuid


DEFAULT_REPO_ROOT = Path("/home/administrator/gemma-4-omni")
DEFAULT_BUNDLE_RELATIVE = Path("checkpoints/experimental-v5-repair200-fphase-20261003")
DEFAULT_IMAGE = "gemma4-omni:latest"
DEFAULT_MAINTENANCE_CONTAINER = "gemma4-speech-runtime-query-film-v2-20261009"
OWNERSHIP_LABEL = "org.codex.task"
OWNERSHIP_VALUE = "pair-duplex-proof-3090"
LOAD_MARKER = "[launch] load engine: OK"
EFFECTIVE_RESPONSE_PROTOCOL = "controller_guided_say_v1"
EFFECTIVE_FRAME_RESIDUAL_ARCHITECTURE = "response_hidden_q1_query_film_v2"
EXPECTED_NATIVE_TALKER_ABI = 4
EXPECTED_NATIVE_TALKER_TRACE_ABI = 3
EXPECTED_NATIVE_TALKER_LIBRARY_SHA256 = "0b2303a0902ca61c375b5babe159f6ce6afd93d77291d935bf16e2469cbeff44"
NATIVE_TALKER_LIBRARY_CONTAINER_PATH = "/workspace/native/qwen_talker/libqwen_talker.so"
NATIVE_LISTENER_LIBRARY_PATH_ENV = "GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"
NATIVE_LISTENER_ATTESTATION_DIRECTORY_ENV = "PAIR_DUPLEX_LISTENER_ATTESTATION_DIR"
NATIVE_LISTENER_CANONICAL_LIBRARY_CONTAINER_PATH = "/workspace/native/gemma4_listener/libgemma4_listener.so"
NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH = (
    "/workspace/.tmp/listener_sliding_capacity_candidate_20261011/native/"
    "gemma4_listener/libgemma4_listener.so"
)
NATIVE_LISTENER_CANDIDATE_SOURCE_ROOT = (
    "/workspace/.tmp/listener_sliding_capacity_candidate_20261011/native/gemma4_listener"
)
NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST = (
    "/workspace/.tmp/listener_sliding_capacity_candidate_20261011/candidate_source_sha256.txt"
)
EXPECTED_NATIVE_LISTENER_CANONICAL_SHA256 = "268d937c9191921c0dc04a5635ea14764ef4dafd770db1fb02ddc9df129d90fa"
EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256 = "0bf885240d63bf6581142613a6b1ded45c2290efe0e809041e4a6e22a6ed94eb"
EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST_SHA256 = (
    "bb302325040dc54c1fcef3380def5915ee477cb3f2fcc8322b1102f35348f691"
)
EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_FILE_COUNT = 22
EXPECTED_NATIVE_LISTENER_ABI = 3
NATIVE_TALKER_Q1_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1=1"
NATIVE_TALKER_Q1_GRAPH_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH=1"
CODE2WAV_GRAPH_FAMILY_ENV = "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1"
CODE2WAV_DECODE_FRAMES_ENV = "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2"
CODE2WAV_GRAPH_ODD_TAIL_ENV = "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1"
CODE2WAV_GRAPH_EVEN_ONLY_ENV = "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=0"
FIRST_INPUT_SCHEDULER_ANCHOR_ENV = "GEMMA4_OMNI_FIRST_INPUT_SCHEDULER_ANCHOR"
READY_FRAME_SCHEDULER_DISPATCH_ENV = "GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH"
LISTENER_PAIR_LOCK_HANDOFF_ENV = "GEMMA4_OMNI_LISTENER_PAIR_LOCK_HANDOFF"
NATIVE_LISTENER_CUDA_PROFILE_ENV = "GEMMA4_NATIVE_LISTENER_CUDA_PROFILE"
NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV = "GEMMA4_NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY"
NATIVE_CONTROL6_FUSED_READBACK_ENV = "GEMMA4_OMNI_NATIVE_CONTROL6_FUSED_READBACK"
NATIVE_CONTROL6_FUSED_READBACK_SMOKE_RELATIVE_PATH = (
    ".tmp/native_control6_fused_readback_cuda_smoke.py"
)
GPU_RELEASE_MIN_FREE_MIB = 12 * 1024
GPU_RELEASE_TIMEOUT_S = 15.0
GPU_RELEASE_POLL_S = 0.5
NATIVE_CONTROL6_FUSED_READBACK_SMOKE_TIMEOUT_S = 300.0
ANALYZER_CHECKPOINTS = {
    "native_control6": "checkpoints/native_control6_gemma_native_rows_rank16_real_mixed_600.pt",
    "gemma_qlora": "checkpoints/gemma4_action_qlora_v1",
    "frame_residual": "checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt",
}

RUNNER_SOURCE = Path(__file__).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_RELATIVE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--maintenance-container", default=None)
    parser.add_argument("--server-startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--readiness-poll-s", type=float, default=2.0)
    parser.add_argument("--maintenance-restore-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-duration-s", type=float, default=60.0)
    parser.add_argument("--skip-analyzer", action="store_true")
    parser.add_argument(
        "--code2wav-graph-family",
        action="store_true",
        help="opt in to two-slot, 37-window Code2Wav CUDA graph-family capture",
    )
    parser.add_argument(
        "--code2wav-graph-odd-tail",
        action="store_true",
        help="capture exact odd Code2Wav flush windows 1..73; requires --code2wav-graph-family",
    )
    parser.add_argument(
        "--native-listener-capacity-candidate",
        action="store_true",
        help="proof-only opt in to the pinned native listener capacity candidate DSO",
    )
    parser.add_argument(
        "--native-listener-cuda-profile",
        action="store_true",
        help="opt in to bounded native Q4 CUDA-event timing samples",
    )
    parser.add_argument(
        "--first-input-scheduler-anchor",
        action="store_true",
        help="proof-only opt in to anchoring the 80 ms scheduler epoch on each peer's first complete input frame",
    )
    parser.add_argument(
        "--ready-frame-scheduler-dispatch",
        action="store_true",
        help="proof-only opt in to dispatching complete queued frames immediately without empty scheduler ticks",
    )
    parser.add_argument(
        "--listener-pair-lock-handoff",
        action="store_true",
        help="proof-only opt in to grant at most two consecutive listener lock quanta before a waiting compute quantum",
    )
    parser.add_argument(
        "--listener-policy-handoff",
        action="store_true",
        help="proof-only opt in to briefly protect a fresh listener frame's policy callback from a new compute grant",
    )
    parser.add_argument(
        "--native-control6-fused-readback",
        action="store_true",
        help="opt in to one packed host readback for normal native Control6 scoring",
    )
    return parser


def validate_code2wav_graph_options(
    *, code2wav_graph_family: bool, code2wav_graph_odd_tail: bool,
) -> None:
    if code2wav_graph_odd_tail and not code2wav_graph_family:
        raise ValueError("--code2wav-graph-odd-tail requires --code2wav-graph-family")


def validate_scheduler_mode(
    *, first_input_scheduler_anchor: bool, ready_frame_scheduler_dispatch: bool,
) -> None:
    if first_input_scheduler_anchor and ready_frame_scheduler_dispatch:
        raise ValueError(
            "--first-input-scheduler-anchor and "
            "--ready-frame-scheduler-dispatch are mutually exclusive"
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _relative_output(root: Path, output_dir: Path) -> str:
    resolved = output_dir if output_dir.is_absolute() else root / output_dir
    try:
        return resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("output directory must be inside the mounted repository") from exc


def validated_native_listener_cuda_profile_directory(directory: str | None) -> str:
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("native listener CUDA profile requires an explicit mounted output directory")
    path = PurePosixPath(directory)
    if not path.is_absolute() or len(path.parts) < 3 or path.parts[1] != "workspace" or ".." in path.parts:
        raise ValueError("native listener CUDA profile directory must be inside /workspace")
    return path.as_posix()


def native_listener_attestation_filename(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id or len(session_id) > 512:
        raise ValueError("listener attestation requires a nonempty bounded session ID")
    if any(ord(character) < 32 or ord(character) == 127 for character in session_id):
        raise ValueError("listener attestation session ID contains control characters")
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest() + ".json"


def _validated_listener_attestation_container_path(path: str | None) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("candidate mode requires a per-run listener attestation directory")
    candidate = PurePosixPath(path)
    normalized = PurePosixPath(posixpath.normpath(path))
    workspace = PurePosixPath("/workspace")
    if (
        not candidate.is_absolute()
        or candidate != normalized
        or candidate == workspace
        or workspace not in candidate.parents
        or ".." in candidate.parts
    ):
        raise ValueError("listener attestation directory must be a normalized child of /workspace")
    return str(candidate)


def build_server_argv(*, bundle_path: str) -> list[str]:
    return [
        "/workspace/.venv_e2e_runtime/bin/python",
        "-m",
        "duplex_kernel.launch",
        "--workspace", "/workspace",
        "--host", "0.0.0.0",
        "--port", "8092",
        "--enable-gemma-native-control6",
        "--native-control6-adapter",
        "checkpoints/native_control6_gemma_native_rows_rank16_real_mixed_600.pt",
        "--gemma-qlora-adapter-dir", "checkpoints/gemma4_action_qlora_v1",
        "--frame-residual-adapter-checkpoint",
        "checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt",
        "--thinker-prefill-chunk-tokens", "16",
        "--thinker-audio-prefill-chunk-tokens", "0",
        "--thinker-compile-mode", "default",
        "--duplex-response-bundle-dir", bundle_path,
        "--ablate-response-lora",
        "--controller-guided-talk-context-pool",
        "--enforce-v9-training-voice",
        "--v9-conditioning-source", "live_generated",
        "--runtime-backend", "native",
    ]


def build_client_argv(*, output_dir: str, max_turns: int = 4, max_duration_s: float = 60.0) -> list[str]:
    return [
        "/workspace/.venv_e2e_runtime/bin/python",
        "client/duplex_stream_e2e.py",
        "--mode", "pair",
        "--ws-url", "ws://127.0.0.1:8092/v1/realtime?duplex=1",
        "--audio", "artifacts/scenario_audio_qwen3tts/mode_b_kickoff.wav",
        "--output-dir", output_dir,
        "--turn-policy", "gemma_native_control6",
        "--max-turns", str(max_turns),
        "--max-duration-s", format(max_duration_s, "g"),
    ]


def build_docker_run_argv(
    *,
    repository_root: str | Path,
    container_name: str,
    image: str,
    server_argv: Sequence[str],
    code2wav_graph_family: bool = False,
    code2wav_graph_odd_tail: bool = False,
    native_listener_capacity_candidate: bool = False,
    native_listener_attestation_dir: str | None = None,
    native_listener_cuda_profile: bool = False,
    native_listener_cuda_profile_directory: str | None = None,
    first_input_scheduler_anchor: bool = False,
    ready_frame_scheduler_dispatch: bool = False,
    listener_pair_lock_handoff: bool = False,
    native_control6_fused_readback: bool = False,
) -> list[str]:
    validate_code2wav_graph_options(
        code2wav_graph_family=code2wav_graph_family,
        code2wav_graph_odd_tail=code2wav_graph_odd_tail,
    )
    validate_scheduler_mode(
        first_input_scheduler_anchor=first_input_scheduler_anchor,
        ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
    )
    profile_directory = (
        validated_native_listener_cuda_profile_directory(native_listener_cuda_profile_directory)
        if native_listener_cuda_profile else None
    )
    if not native_listener_cuda_profile and native_listener_cuda_profile_directory is not None:
        raise ValueError("native listener CUDA profile directory requires profile opt-in")
    graph_environment = (
        [
            "--env", CODE2WAV_GRAPH_FAMILY_ENV,
            "--env", CODE2WAV_DECODE_FRAMES_ENV,
            "--env", CODE2WAV_GRAPH_ODD_TAIL_ENV if code2wav_graph_odd_tail
            else CODE2WAV_GRAPH_EVEN_ONLY_ENV,
        ]
        if code2wav_graph_family else []
    )
    native_listener_environment = (
        ["--env", f"{NATIVE_LISTENER_LIBRARY_PATH_ENV}={NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH}"]
        if native_listener_capacity_candidate else []
    )
    attestation_environment = (
        [
            "--env",
            f"{NATIVE_LISTENER_ATTESTATION_DIRECTORY_ENV}="
            f"{_validated_listener_attestation_container_path(native_listener_attestation_dir)}",
        ]
        if native_listener_capacity_candidate else []
    )
    native_listener_profile_environment = [
        "--env", f"{NATIVE_LISTENER_CUDA_PROFILE_ENV}={int(native_listener_cuda_profile)}",
    ]
    if native_listener_cuda_profile:
        native_listener_profile_environment.extend([
            "--env",
            f"{NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV}={profile_directory}",
        ])
    scheduler_anchor_environment = (
        f"{FIRST_INPUT_SCHEDULER_ANCHOR_ENV}={int(first_input_scheduler_anchor)}",
    )
    return [
        "docker", "run", "--detach",
        "--name", container_name,
        "--label", f"{OWNERSHIP_LABEL}={OWNERSHIP_VALUE}",
        "--user", "1000:1000",
        "--ipc", "host",
        "--gpus", "all",
        "--env", "DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU=0",
        "--env", "DUPLEX_AUDIO_HISTORY_ARCHIVE_DEFER=1",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "HF_DATASETS_OFFLINE=1",
        "--env", "DUPLEX_NATIVE_CONTROL6_TRACE=0",
        "--env", NATIVE_TALKER_Q1_ENV,
        "--env", NATIVE_TALKER_Q1_GRAPH_ENV,
        *graph_environment,
        *native_listener_environment,
        *attestation_environment,
        *native_listener_profile_environment,
        "--env", *scheduler_anchor_environment,
        "--env", f"{READY_FRAME_SCHEDULER_DISPATCH_ENV}={int(ready_frame_scheduler_dispatch)}",
        "--env", f"{LISTENER_PAIR_LOCK_HANDOFF_ENV}={int(listener_pair_lock_handoff)}",
        "--env", f"{NATIVE_CONTROL6_FUSED_READBACK_ENV}={int(native_control6_fused_readback)}",
        "--env", f"QWEN_NATIVE_LIBRARY_PATH={NATIVE_TALKER_LIBRARY_CONTAINER_PATH}",
        "--publish", "127.0.0.1:8092:8092",
        "--volume", f"{str(repository_root).replace(chr(92), '/') }:/workspace",
        "--workdir", "/workspace",
        "--pull=never",
        "--init",
        image,
        *server_argv,
    ]


def is_server_ready(log_text: str, health: Mapping[str, Any] | None) -> bool:
    return bool(
        LOAD_MARKER in log_text
        and isinstance(health, Mapping)
        and health.get("status") == "ok"
        and health.get("engine_loaded") is True
    )


def effective_route_record(health: Mapping[str, Any], server_argv: Sequence[str]) -> dict[str, Any]:
    """Validate and attest the runtime route selected for this pair capture."""
    response_protocol = health.get("response_protocol")
    if response_protocol != EFFECTIVE_RESPONSE_PROTOCOL:
        raise RuntimeError(
            "unexpected effective response protocol: "
            f"expected {EFFECTIVE_RESPONSE_PROTOCOL!r}, got {response_protocol!r}"
        )
    if health.get("direct_audio_candidate") is not True:
        raise RuntimeError("controller-guided pair route requires direct audio generation")
    if health.get("audio_history_archive_defer_offline") is not True:
        raise RuntimeError("pair route requires proof-only offline archive ASR deferral")
    if health.get("audio_history_archive_cpu_startup_available") is not False:
        raise RuntimeError("pair route must disable automatic CPU archive ASR during dialogue")
    if health.get("talker_q1_backend") != "native_audio_q1":
        raise RuntimeError(
            "pair route requires native_audio_q1 Talker backend, got "
            f"{health.get('talker_q1_backend')!r}"
        )
    if health.get("talker_q1_native_status") != "ready":
        raise RuntimeError(
            "pair route requires ready native_audio_q1 Talker backend, got "
            f"{health.get('talker_q1_native_status')!r}"
        )
    native_action_selection = health.get("native_action_selection")
    if native_action_selection != "controller_authoritative":
        raise RuntimeError(
            "controller-guided pair route requires controller_authoritative action selection"
        )
    residual = health.get("frame_residual_adapter")
    if not isinstance(residual, Mapping) or residual.get("enabled") is not True:
        raise RuntimeError("controller-guided pair route requires an enabled frame residual adapter")
    architecture = residual.get("architecture")
    if architecture != EFFECTIVE_FRAME_RESIDUAL_ARCHITECTURE:
        raise RuntimeError(
            "unexpected frame residual architecture: "
            f"expected {EFFECTIVE_FRAME_RESIDUAL_ARCHITECTURE!r}, got {architecture!r}"
        )
    try:
        compile_mode = str(server_argv[server_argv.index("--thinker-compile-mode") + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("pair runner must explicitly set --thinker-compile-mode") from exc
    if compile_mode != "default":
        raise RuntimeError(f"pair runner requires thinker compile mode 'default', got {compile_mode!r}")
    if "--ablate-response-lora" not in server_argv:
        raise RuntimeError("frame residual pair route requires --ablate-response-lora")
    return {
        "response_protocol": response_protocol,
        "direct_audio_candidate": True,
        "audio_history_archive_defer_offline": True,
        "audio_history_archive_cpu_startup_available": False,
        "talker_q1_backend": "native_audio_q1",
        "talker_q1_native_status": "ready",
        "native_action_selection": native_action_selection,
        "frame_residual_adapter": {
            "enabled": True,
            "architecture": architecture,
            "checkpoint": residual.get("checkpoint"),
        },
        "response_lora": "ablated",
        "thinker_compile_mode": compile_mode,
    }


def code2wav_graph_family_record(
    health: Mapping[str, Any], *, requested: bool, odd_tail_requested: bool = False,
) -> dict[str, Any]:
    """Fail closed unless each requested exact-window family was captured in both slots."""
    validate_code2wav_graph_options(
        code2wav_graph_family=requested,
        code2wav_graph_odd_tail=odd_tail_requested,
    )
    if not requested:
        return {"requested": False, "status": "not_requested"}
    pool = health.get("controller_guided_talk_context_pool")
    if not isinstance(pool, Mapping) or pool.get("enabled") is not True or pool.get("capacity") != 2:
        raise RuntimeError("Code2Wav graph family requires an enabled two-slot response context pool")
    memory = pool.get("memory")
    if not isinstance(memory, Mapping) or memory.get("code2wav_graph_family_enabled") is not True:
        raise RuntimeError("Code2Wav graph family was requested but startup health did not attest it enabled")
    if memory.get("code2wav_graph_odd_tail_enabled") is not odd_tail_requested:
        raise RuntimeError("Code2Wav graph family odd-tail setting differs from the requested mode")
    slots = memory.get("code2wav_graph_capture_status_by_slot")
    expected_windows = list(range(1, 75)) if odd_tail_requested else list(range(2, 75, 2))
    expected_graph_count = len(expected_windows)
    if not isinstance(slots, list) or len(slots) != 2:
        raise RuntimeError("Code2Wav graph family startup health must report exactly two slots")
    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            raise RuntimeError(f"Code2Wav graph family slot {index} telemetry is missing")
        if (
            slot.get("captured") is not True
            or slot.get("graph_count") != expected_graph_count
            or slot.get("window_frames") != expected_windows
            or "capture_error" not in slot
            or slot.get("capture_error") is not None
        ):
            raise RuntimeError(
                f"Code2Wav graph family slot {index} is incomplete or fell back: {dict(slot)!r}"
            )
    return {
        "requested": True,
        "status": "PASS",
        "odd_tail_requested": odd_tail_requested,
        "slots": 2,
        "graphs_per_slot": expected_graph_count,
        "window_frames": expected_windows,
    }


def first_input_scheduler_anchor_record(
    health: Mapping[str, Any], *, requested: bool,
) -> dict[str, Any]:
    actual = health.get("first_input_scheduler_anchor_enabled")
    if actual is not requested:
        raise RuntimeError(
            "server first-input scheduler anchor health differs from the requested mode: "
            f"requested={requested!r}, actual={actual!r}"
        )
    return {"requested": requested, "actual": actual, "status": "PASS"}


def ready_frame_scheduler_dispatch_record(
    health: Mapping[str, Any], *, requested: bool,
) -> dict[str, Any]:
    actual = health.get("ready_frame_scheduler_dispatch_enabled")
    if actual is not requested:
        raise RuntimeError(
            "server ready-frame scheduler dispatch health differs from the requested mode: "
            f"requested={requested!r}, actual={actual!r}"
        )
    sessions = health.get("ready_frame_scheduler_dispatch_sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("server health omitted ready-frame scheduler session attestations")
    if any(
        not isinstance(row, Mapping)
        or row.get("enabled") is not requested
        for row in sessions
    ):
        raise RuntimeError("server ready-frame scheduler session mode differs from the requested mode")
    return {"requested": requested, "actual": actual, "status": "PASS"}


def listener_pair_lock_handoff_record(
    health: Mapping[str, Any], *, requested: bool,
) -> dict[str, Any]:
    actual = health.get("listener_pair_lock_handoff_enabled")
    expected_setting = "1" if requested else "0"
    actual_setting = health.get("listener_pair_lock_handoff_setting")
    if actual is not requested or actual_setting != expected_setting:
        raise RuntimeError(
            "server listener pair lock handoff health differs from the requested mode: "
            f"requested={requested!r}, actual={actual!r}, "
            f"setting={actual_setting!r}"
        )
    return {"requested": requested, "actual": actual, "status": "PASS"}


def native_listener_cuda_profile_record(
    health: Mapping[str, Any], *, requested: bool, directory: str | None,
) -> dict[str, Any]:
    actual = health.get("native_listener_cuda_profile_enabled")
    lock_timeline = health.get("listener_model_lock_host_timeline_enabled")
    actual_directory = health.get("native_listener_cuda_profile_directory")
    if actual is not requested:
        raise RuntimeError(
            "server native listener CUDA profile health differs from the requested mode: "
            f"requested={requested!r}, actual={actual!r}"
        )
    if lock_timeline is not requested:
        raise RuntimeError(
            "server model-lock host timeline health differs from native profile request: "
            f"requested={requested!r}, actual={lock_timeline!r}"
        )
    if requested and actual_directory != directory:
        raise RuntimeError(
            "server native listener CUDA profile directory differs from the requested directory: "
            f"requested={directory!r}, actual={actual_directory!r}"
        )
    return {
        "requested": requested,
        "actual": actual,
        "host_timeline_enabled": lock_timeline,
        "actual_directory": actual_directory,
        "status": "PASS",
        "directory_in_container": actual_directory,
        "requested_directory_in_container": directory,
        "sampling_frequency_frames": 8,
        "sample_cap_per_context": 64,
    }


def native_control6_fused_readback_record(
    health: Mapping[str, Any], *, requested: bool,
) -> dict[str, Any]:
    actual = health.get("native_control6_fused_readback_enabled")
    actual_setting = health.get("native_control6_fused_readback_setting")
    expected_setting = "1" if requested else "0"
    if actual is not requested or actual_setting != expected_setting:
        raise RuntimeError(
            "server native Control6 fused-readback health differs from the requested mode: "
            f"requested={requested!r}, actual={actual!r}, setting={actual_setting!r}"
        )
    return {"requested": requested, "actual": actual, "status": "PASS"}


def native_talker_binary_probe_code() -> str:
    return (
        "import ctypes, hashlib, json, os, pathlib, torch\n"
        "configured_path = os.environ.get('QWEN_NATIVE_LIBRARY_PATH')\n"
        "if not configured_path:\n"
        "    raise RuntimeError('QWEN_NATIVE_LIBRARY_PATH is required for the pair proof')\n"
        "path = pathlib.Path(configured_path).resolve()\n"
        f"expected_path = pathlib.Path({NATIVE_TALKER_LIBRARY_CONTAINER_PATH!r}).resolve()\n"
        "if path != expected_path:\n"
        "    raise RuntimeError(f'QWEN_NATIVE_LIBRARY_PATH resolved to {path}, expected {expected_path}')\n"
        "digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "library = ctypes.CDLL(str(path))\n"
        "library.qwen_talker_abi_version.restype = ctypes.c_uint32\n"
        "library.qwen_talker_trace_abi_version.restype = ctypes.c_uint32\n"
        "native_talker_library = {'path': str(path), 'configured_path': configured_path, "
        "'size_bytes': path.stat().st_size, 'sha256': digest, "
        "'native_abi_version': int(library.qwen_talker_abi_version()), "
        "'trace_abi_version': int(library.qwen_talker_trace_abi_version())}\n"
    )


def validate_native_talker_binary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    expected_path = NATIVE_TALKER_LIBRARY_CONTAINER_PATH
    if evidence.get("path") != expected_path:
        raise RuntimeError(
            "native Talker library path differs from the canonical ABI4 library: "
            f"{evidence.get('path')!r}"
        )
    if evidence.get("configured_path") != expected_path:
        raise RuntimeError(
            "native Talker configured path differs from the canonical ABI4 library: "
            f"{evidence.get('configured_path')!r}"
        )
    if evidence.get("sha256") != EXPECTED_NATIVE_TALKER_LIBRARY_SHA256:
        raise RuntimeError(
            "native Talker shared-library hash differs from the ABI4 rebuild: "
            f"{evidence.get('sha256')!r}"
        )
    if evidence.get("native_abi_version") != EXPECTED_NATIVE_TALKER_ABI:
        raise RuntimeError(
            f"native Talker ABI must be {EXPECTED_NATIVE_TALKER_ABI}, "
            f"got {evidence.get('native_abi_version')!r}"
        )
    if evidence.get("trace_abi_version") != EXPECTED_NATIVE_TALKER_TRACE_ABI:
        raise RuntimeError(
            f"native Talker trace ABI must be {EXPECTED_NATIVE_TALKER_TRACE_ABI}, "
            f"got {evidence.get('trace_abi_version')!r}"
        )
    return dict(evidence)


def native_listener_candidate_probe_code() -> str:
    """Attest the isolated listener candidate, its ABI structs, and pinned source rows."""
    return (
        "import ctypes, hashlib, json, os, pathlib\n"
        "from duplex_kernel.native_listener import (Gemma4ImportedCache, Gemma4ImportedKV, "
        "Gemma4LayerWeights, Gemma4ListenerConfig, Gemma4ListenerWeights, Gemma4TensorMeta)\n"
        f"expected_path = pathlib.Path({NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH!r}).resolve()\n"
        f"expected_source_root = pathlib.Path({NATIVE_LISTENER_CANDIDATE_SOURCE_ROOT!r}).resolve()\n"
        f"manifest_path = pathlib.Path({NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST!r}).resolve()\n"
        f"expected_manifest_sha = {EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST_SHA256!r}\n"
        f"expected_library_sha = {EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256!r}\n"
        f"configured_path = os.environ.get({NATIVE_LISTENER_LIBRARY_PATH_ENV!r})\n"
        "if not configured_path or pathlib.Path(configured_path).resolve() != expected_path:\n"
        "    raise RuntimeError('native listener candidate env/path differs from pinned candidate')\n"
        "if not expected_path.is_file() or not manifest_path.is_file():\n"
        "    raise RuntimeError('pinned native listener candidate or source manifest is missing')\n"
        "library_sha = hashlib.sha256(expected_path.read_bytes()).hexdigest()\n"
        "if library_sha != expected_library_sha:\n"
        "    raise RuntimeError(f'native listener candidate SHA mismatch: {library_sha}')\n"
        "manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()\n"
        "if manifest_sha != expected_manifest_sha:\n"
        "    raise RuntimeError(f'native listener source manifest SHA mismatch: {manifest_sha}')\n"
        "verified_sources = 0\n"
        "seen_sources = set()\n"
        "for line_number, line in enumerate(manifest_path.read_text(encoding='utf-8').splitlines(), 1):\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    parts = line.split(maxsplit=1)\n"
        "    if len(parts) != 2 or len(parts[0]) != 64:\n"
        "        raise RuntimeError(f'malformed native listener source manifest row {line_number}')\n"
        "    relative = parts[1].lstrip('* ')\n"
        "    if relative in seen_sources:\n"
        "        raise RuntimeError(f'duplicate native listener source manifest row: {relative}')\n"
        "    seen_sources.add(relative)\n"
        "    source = (expected_source_root / relative).resolve()\n"
        "    if expected_source_root not in source.parents or not source.is_file():\n"
        "        raise RuntimeError(f'native listener source path missing or outside candidate root: {relative}')\n"
        "    if hashlib.sha256(source.read_bytes()).hexdigest() != parts[0].lower():\n"
        "        raise RuntimeError(f'native listener source hash mismatch: {relative}')\n"
        "    verified_sources += 1\n"
        f"if verified_sources != {EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_FILE_COUNT}:\n"
        "    raise RuntimeError(f'native listener source count mismatch: {verified_sources}')\n"
        "library = ctypes.CDLL(str(expected_path))\n"
        "library.gemma4_listener_abi_version.restype = ctypes.c_int\n"
        "if int(library.gemma4_listener_abi_version()) != 3:\n"
        "    raise RuntimeError('native listener candidate must expose ABI version 3')\n"
        "abi_size_types = {\n"
        "    'config': ('gemma4_listener_abi_config_size', Gemma4ListenerConfig),\n"
        "    'tensor_meta': ('gemma4_listener_abi_tensor_meta_size', Gemma4TensorMeta),\n"
        "    'layer_weights': ('gemma4_listener_abi_layer_weights_size', Gemma4LayerWeights),\n"
        "    'weights': ('gemma4_listener_abi_weights_size', Gemma4ListenerWeights),\n"
        "    'imported_kv': ('gemma4_listener_abi_imported_kv_size', Gemma4ImportedKV),\n"
        "    'imported_cache': ('gemma4_listener_abi_imported_cache_size', Gemma4ImportedCache),\n"
        "}\n"
        "binary_sizes = {}\n"
        "python_sizes = {}\n"
        "for name, (symbol, struct_type) in abi_size_types.items():\n"
        "    function = getattr(library, symbol)\n"
        "    function.restype = ctypes.c_size_t\n"
        "    binary_sizes[name] = int(function())\n"
        "    python_sizes[name] = int(ctypes.sizeof(struct_type))\n"
        "if binary_sizes != python_sizes:\n"
        "    raise RuntimeError(f'native listener ABI struct sizes differ: {binary_sizes} != {python_sizes}')\n"
        "native_listener_candidate = {\n"
        "    'configured_path': configured_path, 'path': str(expected_path), 'sha256': library_sha,\n"
        "    'abi_version': 3, 'source_manifest_path': str(manifest_path),\n"
        "    'source_manifest_sha256': manifest_sha, 'source_root': str(expected_source_root),\n"
        "    'source_file_count': verified_sources, 'abi_struct_sizes': binary_sizes,\n"
        "    'python_struct_sizes': python_sizes,\n"
        "}\n"
    )


def validate_native_listener_candidate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "configured_path": NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
        "path": NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
        "sha256": EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
        "abi_version": EXPECTED_NATIVE_LISTENER_ABI,
        "source_manifest_path": NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST,
        "source_manifest_sha256": EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST_SHA256,
        "source_root": NATIVE_LISTENER_CANDIDATE_SOURCE_ROOT,
        "source_file_count": EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_FILE_COUNT,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise RuntimeError(
                f"native listener candidate {key} mismatch: expected {value!r}, "
                f"got {evidence.get(key)!r}"
            )
    binary_sizes = evidence.get("abi_struct_sizes")
    python_sizes = evidence.get("python_struct_sizes")
    expected_structs = {"config", "tensor_meta", "layer_weights", "weights", "imported_kv", "imported_cache"}
    if (
        not isinstance(binary_sizes, Mapping)
        or not isinstance(python_sizes, Mapping)
        or set(binary_sizes) != expected_structs
        or set(python_sizes) != expected_structs
        or dict(binary_sizes) != dict(python_sizes)
        or any(type(size) is not int or size <= 0 for size in binary_sizes.values())
    ):
        raise RuntimeError("native listener candidate ABI3 struct layout attestation is missing or mismatched")
    return dict(evidence)


def attest_native_talker_binary(container_id: str) -> dict[str, Any]:
    result = _command([
        "docker", "exec", container_id,
        "/workspace/.venv_e2e_runtime/bin/python", "-c",
        native_talker_binary_probe_code() + "print(json.dumps(native_talker_library, sort_keys=True))\n",
    ])
    try:
        evidence = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("native Talker ABI probe returned invalid JSON") from exc
    if not isinstance(evidence, dict):
        raise RuntimeError("native Talker ABI probe returned a non-object")
    return validate_native_talker_binary(evidence)


def native_talker_postrun_attestation(report_path: Path) -> dict[str, Any]:
    """Require per-audible-response evidence that the native Q1 backend executed."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": f"client_report_unavailable:{type(exc).__name__}"}
    raw_rows = report.get("responses") if isinstance(report, dict) else None
    responses = [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
    audible = [row for row in responses if int(row.get("audio_samples", 0) or 0) > 0]
    rows: list[dict[str, Any]] = []
    for row in audible:
        runtime = row.get("runtime_metrics")
        talker = runtime.get("talker") if isinstance(runtime, Mapping) else None
        backend = talker.get("talker_q1_backend") if isinstance(talker, Mapping) else None
        calls = talker.get("native_talker_q1_calls") if isinstance(talker, Mapping) else None
        rows.append({
            "response_id": str(row.get("response_id", "")),
            "terminal_event": row.get("terminal_event"),
            "audio_samples": int(row.get("audio_samples", 0) or 0),
            "talker_q1_backend": backend,
            "native_talker_q1_calls": calls,
            "pass": backend == "native_audio_q1" and type(calls) is int and calls > 0,
        })
    passed = bool(rows) and all(item["pass"] for item in rows)
    return {
        "status": "PASS" if passed else "FAIL",
        "audible_response_count": len(rows),
        "responses": rows,
        "reason": None if passed else "every_audible_response_must_report_native_audio_q1_calls",
    }


def archive_defer_postrun_attestation(report_path: Path) -> dict[str, Any]:
    """Verify both session ACKs and per-response metrics for proof-only ASR deferral."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": f"client_report_unavailable:{type(exc).__name__}"}
    participants = report.get("participants") if isinstance(report, dict) else None
    peer_settings: dict[str, Mapping[str, Any]] = {}
    peer_session_ids: set[str] = set()
    participant_rows_valid = isinstance(participants, list) and len(participants) == 2
    if participant_rows_valid:
        for item in participants:
            if not isinstance(item, Mapping):
                participant_rows_valid = False
                continue
            speaker = item.get("speaker")
            session_id = item.get("session_id")
            settings = item.get("session_updated")
            if (
                not isinstance(speaker, str)
                or speaker not in {"A", "B"}
                or speaker in peer_settings
                or not isinstance(session_id, str)
                or not session_id
                or session_id in peer_session_ids
                or not isinstance(settings, Mapping)
            ):
                participant_rows_valid = False
                continue
            peer_settings[str(speaker)] = settings
            peer_session_ids.add(session_id)
    sessions_pass = (
        participant_rows_valid
        and set(peer_settings) == {"A", "B"}
        and len(peer_session_ids) == 2
        and all(item.get("audio_history_archive_defer_offline") is True for item in peer_settings.values())
        and all(item.get("audio_history_archive_cpu_enabled") is False for item in peer_settings.values())
    )
    raw_responses = report.get("responses") if isinstance(report, dict) else None
    responses = [row for row in raw_responses if isinstance(row, Mapping)] if isinstance(raw_responses, list) else []
    archives = []
    for row in responses:
        runtime = row.get("runtime_metrics")
        archive = runtime.get("audio_history_archive") if isinstance(runtime, Mapping) else None
        if isinstance(archive, Mapping) and archive.get("source") == "post_response_asr":
            archives.append({
                "response_id": str(row.get("response_id", "")),
                "archive_id": archive.get("archive_id"),
                "status": archive.get("status"),
                "transcript_committed": archive.get("transcript_committed"),
            })
    archives_pass = bool(archives) and all(
        row.get("status") == "deferred_offline"
        and row.get("transcript_committed") is False
        for row in archives
    )
    passed = sessions_pass and archives_pass
    return {
        "status": "PASS" if passed else "FAIL",
        "session_acknowledgements_pass": sessions_pass,
        "deferred_archive_count": len(archives),
        "archives": archives,
        "reason": None if passed else "both_peers_must_ack_defer_and_each_post_response_archive_must_remain_untranscribed",
    }


def native_listener_session_postrun_attestation(
    report_path: Path,
    server_log_path: Path,
    *,
    attestation_dir: Path | None = None,
    expected_library_path: str,
    expected_library_sha256: str,
) -> dict[str, Any]:
    """Bind atomic per-session sidecars to both actual native listener contexts."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": f"client_report_unavailable:{type(exc).__name__}"}
    participants = report.get("participants") if isinstance(report, dict) else None
    participant_ids: dict[str, str] = {}
    if not isinstance(participants, list) or len(participants) != 2:
        return {"status": "FAIL", "reason": "exactly_two_participant_rows_required", "sessions": []}
    for participant in participants:
        if not isinstance(participant, Mapping):
            return {"status": "FAIL", "reason": "participant_row_malformed", "sessions": []}
        speaker = participant.get("speaker")
        session_id = participant.get("session_id")
        if (
            speaker not in {"A", "B"}
            or speaker in participant_ids
            or not isinstance(session_id, str)
            or not session_id
            or session_id in participant_ids.values()
        ):
            return {"status": "FAIL", "reason": "participant_identity_invalid", "sessions": []}
        participant_ids[str(speaker)] = session_id
    if set(participant_ids) != {"A", "B"}:
        return {"status": "FAIL", "reason": "peer_A_B_sessions_required", "sessions": []}

    malformed_metric_rows = 0
    server_log_diagnostic_error: str | None = None
    try:
        log_lines = server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        log_lines = []
        server_log_diagnostic_error = f"{type(exc).__name__}: {exc}"
    marker = "[realtime_session_metrics] "
    for line in log_lines:
        _, found, payload = line.partition(marker)
        if not found:
            continue
        try:
            json.loads(payload)
        except json.JSONDecodeError:
            malformed_metric_rows += 1

    source = "atomic per-session sidecars from native_runtime_info_at_close"
    if attestation_dir is None:
        return {
            "status": "FAIL",
            "reason": "per_session_listener_attestation_directory_required",
            "session_attestation_source": source,
            "malformed_metric_rows": malformed_metric_rows,
            "sessions": [],
        }
    output_root = report_path.parent.resolve()
    requested_directory = attestation_dir if attestation_dir.is_absolute() else output_root / attestation_dir
    resolved_directory = requested_directory.resolve()
    expected_directory = (output_root / "listener_attestations").resolve()
    if resolved_directory != expected_directory:
        return {
            "status": "FAIL",
            "reason": "listener_attestation_directory_not_bound_to_this_run",
            "session_attestation_source": source,
            "attestation_directory": str(resolved_directory),
            "malformed_metric_rows": malformed_metric_rows,
            "sessions": [],
        }
    if not resolved_directory.is_dir():
        return {
            "status": "FAIL",
            "reason": "listener_attestation_directory_missing",
            "session_attestation_source": source,
            "attestation_directory": str(resolved_directory),
            "malformed_metric_rows": malformed_metric_rows,
            "sessions": [],
        }

    expected_names = {
        native_listener_attestation_filename(session_id)
        for session_id in participant_ids.values()
    }
    try:
        directory_entries = list(resolved_directory.iterdir())
    except OSError as exc:
        return {
            "status": "FAIL",
            "reason": f"listener_attestation_directory_unreadable:{type(exc).__name__}",
            "session_attestation_source": source,
            "attestation_directory": str(resolved_directory),
            "malformed_metric_rows": malformed_metric_rows,
            "sessions": [],
        }
    directory_names = {entry.name for entry in directory_entries}
    directory_shape_valid = (
        directory_names == expected_names
        and all(entry.is_file() and not entry.is_symlink() for entry in directory_entries)
    )

    sessions = []
    for speaker in ("A", "B"):
        session_id = participant_ids[speaker]
        sidecar_path = resolved_directory / native_listener_attestation_filename(session_id)
        record: Mapping[str, Any] | None = None
        parse_error: str | None = None
        if sidecar_path.is_file() and not sidecar_path.is_symlink():
            try:
                decoded = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if isinstance(decoded, Mapping):
                    record = decoded
                else:
                    parse_error = "sidecar_json_not_object"
            except (OSError, json.JSONDecodeError) as exc:
                parse_error = f"sidecar_unreadable:{type(exc).__name__}"
        allowed_fields = {
            "schema_version", "session_id", "available", "library_path",
            "library_sha256", "abi_version",
        }
        valid_record_shape = (
            isinstance(record, Mapping)
            and set(record).issubset(allowed_fields)
            and type(record.get("schema_version")) is int
            and record.get("schema_version") == 1
            and record.get("session_id") == session_id
        )
        library_sha256 = record.get("library_sha256") if isinstance(record, Mapping) else None
        valid_hash = (
            isinstance(library_sha256, str)
            and len(library_sha256) == 64
            and all(character in "0123456789abcdefABCDEF" for character in library_sha256)
        )
        abi_version = record.get("abi_version") if isinstance(record, Mapping) else None
        abi_valid = abi_version is None or (
            type(abi_version) is int and abi_version == EXPECTED_NATIVE_LISTENER_ABI
        )
        availability_valid = (
            isinstance(record, Mapping)
            and (
                "available" not in record
                or (type(record.get("available")) is bool and record.get("available") is True)
            )
        )
        all_match = (
            directory_shape_valid
            and valid_record_shape
            and availability_valid
            and record.get("library_path") == expected_library_path
            and valid_hash
            and library_sha256.lower() == expected_library_sha256.lower()
            and abi_valid
        )
        sessions.append({
            "speaker": speaker,
            "session_id": session_id,
            "sidecar_path": str(sidecar_path),
            "sidecar_present": record is not None,
            "sidecar": dict(record) if isinstance(record, Mapping) else None,
            "sidecar_error": parse_error,
            "runtime_info_count": 1 if record is not None else 0,
            "pass": all_match,
        })
    passed = directory_shape_valid and all(row["pass"] for row in sessions)
    return {
        "status": "PASS" if passed else "FAIL",
        "expected_library_path": expected_library_path,
        "expected_library_sha256": expected_library_sha256,
        "session_attestation_source": source,
        "attestation_directory": str(resolved_directory),
        "directory_shape_valid": directory_shape_valid,
        "directory_entry_names": sorted(directory_names),
        "malformed_metric_rows": malformed_metric_rows,
        "server_log_diagnostic_error": server_log_diagnostic_error,
        "server_log_used_for_attestation": False,
        "sessions": sessions,
        "reason": None if passed else "both_actual_peer_sessions_need_unique_bound_candidate_sidecars",
    }


def _command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    text: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=str(cwd) if cwd else None, check=check,
        stdout=stdout, stderr=stderr, text=text, timeout=timeout,
    )


def build_native_control6_fused_readback_cuda_smoke_argv(
    repository_root: str | Path,
    image: str,
    *,
    container_name: str,
    output_path: str | Path,
) -> list[str]:
    """Build the isolated, offline CUDA parity smoke invocation."""
    root = Path(repository_root).expanduser().resolve()
    script_path = root / NATIVE_CONTROL6_FUSED_READBACK_SMOKE_RELATIVE_PATH
    if not script_path.is_file() or script_path.is_symlink():
        raise FileNotFoundError(f"fused-readback CUDA smoke source is missing or unsafe: {script_path}")
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        output_rel = output.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("fused-readback CUDA smoke output must be inside the mounted repository") from exc
    script_container_path = "/workspace/" + Path(
        NATIVE_CONTROL6_FUSED_READBACK_SMOKE_RELATIVE_PATH
    ).as_posix()
    return [
        "docker", "run", "--rm",
        "--name", container_name,
        "--label", f"{OWNERSHIP_LABEL}={OWNERSHIP_VALUE}",
        "--user", "1000:1000",
        "--ipc", "host",
        "--network", "none",
        "--gpus", "all",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "HF_DATASETS_OFFLINE=1",
        "--volume", f"{str(root).replace(chr(92), '/') }:/workspace",
        "--workdir", "/workspace",
        "--pull=never",
        "--entrypoint", "/workspace/.venv_e2e_runtime/bin/python",
        image,
        script_container_path,
        "--output", "/workspace/" + output_rel,
    ]


def run_native_control6_fused_readback_cuda_smoke(
    repository_root: str | Path,
    image: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run and bind the tiny real-CUDA parity smoke before the pair server starts."""
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    container_name = f"control6-fused-smoke-{_timestamp().lower()}-{uuid.uuid4().hex[:8]}"
    argv = build_native_control6_fused_readback_cuda_smoke_argv(
        root, image, container_name=container_name, output_path=output,
    )
    try:
        result = _command(
            argv, cwd=root, check=False,
            timeout=NATIVE_CONTROL6_FUSED_READBACK_SMOKE_TIMEOUT_S,
        )
        log_path.write_text(
            "STDOUT\n" + (result.stdout or "") + "\nSTDERR\n" + (result.stderr or ""),
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"fused-readback CUDA parity smoke exited {result.returncode}; see {log_path}"
            )
        if not output.is_file():
            raise RuntimeError("fused-readback CUDA parity smoke did not write its report")
        report = json.loads(output.read_text(encoding="utf-8"))
        expected_reads = {
            "baseline": {"bool": 2, "item": 3, "cpu": 0},
            "fused": {"bool": 1, "item": 0, "cpu": 1},
        }
        if (
            not isinstance(report, Mapping)
            or report.get("status") != "PASS"
            or str(report.get("torch_version", "")).split("+", 1)[0] != "2.12.1"
            or report.get("finite_comparisons") != 126
            or report.get("nonfinite_failures_compared") != 63
            or report.get("per_successful_step_cuda_readbacks") != expected_reads
        ):
            raise RuntimeError(
                "fused-readback CUDA smoke report failed its bound PASS/count/version/readback contract"
            )
        return {
            "status": "PASS",
            "container_name": container_name,
            "argv": argv,
            "timeout_s": NATIVE_CONTROL6_FUSED_READBACK_SMOKE_TIMEOUT_S,
            "report_path": output.relative_to(root).as_posix(),
            "report_sha256": _sha256(output),
            "log_path": log_path.relative_to(root).as_posix(),
            "report": report,
        }
    finally:
        cleanup_owned_container(None, container_name, root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path, *, repository_root: Path) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise RuntimeError(f"refusing symlink artifact path: {root}")
    if root.is_file():
        files = [root]
    elif root.is_dir():
        contents = sorted(root.rglob("*"))
        linked = [path for path in contents if path.is_symlink()]
        if linked:
            raise RuntimeError(f"artifact tree contains symlinks: {linked[:5]}")
        files = [path for path in contents if path.is_file()]
    else:
        raise FileNotFoundError(root)
    if not files:
        raise RuntimeError(f"artifact contains no files: {root}")
    entries = []
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink artifact file: {path}")
        entries.append({
            "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return entries


def _source_hashes(repository_root: Path) -> list[dict[str, Any]]:
    candidates = [
        *sorted((repository_root / "duplex_kernel").rglob("*.py")),
        repository_root / "client" / "duplex_stream_e2e.py",
        repository_root / "tools" / "analyze_pair_duplex_certification.py",
        RUNNER_SOURCE,
    ]
    sources = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            relative = path.name
        sources.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return sources


def _git_state(repository_root: Path, *, exclude_relative: str | None = None) -> dict[str, Any]:
    rev = _command(["git", "rev-parse", "HEAD"], cwd=repository_root)
    status = _command(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repository_root)
    rows = []
    for line in (status.stdout or "").splitlines():
        if not line.strip():
            continue
        entry_path = line[3:].split(" -> ")[-1].replace("\\", "/")
        if exclude_relative and (entry_path == exclude_relative or entry_path.startswith(exclude_relative.rstrip("/") + "/")):
            continue
        rows.append(line)
    return {
        "sha": (rev.stdout or "").strip(),
        "dirty": bool(rows),
        "status_sha256": hashlib.sha256((status.stdout or "").encode("utf-8")).hexdigest(),
        "dirty_entries": rows,
    }


def collect_run_metadata(repository_root: Path, bundle_dir: Path, *, output_relative: str | None = None) -> dict[str, Any]:
    root = repository_root.resolve()
    bundle = bundle_dir if bundle_dir.is_absolute() else root / bundle_dir
    files = {
        "native_control6": root / "checkpoints/native_control6_gemma_native_rows_rank16_real_mixed_600.pt",
        "gemma_qlora": root / "checkpoints/gemma4_action_qlora_v1",
        "frame_residual": root / "checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt",
        "response_bundle": bundle,
        "pair_audio": root / "artifacts/scenario_audio_qwen3tts/mode_b_kickoff.wav",
        "server_python": root / ".venv_e2e_runtime/bin/python",
        "launch_source": root / "duplex_kernel/launch.py",
        "client_source": root / "client/duplex_stream_e2e.py",
    }


    artifact_records: dict[str, Any] = {}
    for label in ("native_control6", "gemma_qlora", "frame_residual", "response_bundle"):
        artifact_records[label] = _hash_tree(files[label], repository_root=root)
    for label in ("pair_audio", "server_python", "launch_source", "client_source"):
        path = files[label]
        if label == "server_python":
            artifact_records[label] = {
                "path": path.relative_to(root).as_posix(),
                "container_path": "/workspace/.venv_e2e_runtime/bin/python",
                "host_exists": path.exists(),
                "resolved_target": str(path.resolve()) if path.exists() else None,
                "resolution_scope": "container_image_or_nested_mount",
            }
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_records[label] = {
            "path": path.resolve().relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"response bundle manifest missing: {manifest_path}")
    try:
        bundle_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid response bundle manifest: {manifest_path}: {exc}") from exc
    if not isinstance(bundle_metadata, dict):
        raise RuntimeError(f"response bundle manifest must be a JSON object: {manifest_path}")
    return {
        "git": _git_state(root, exclude_relative=output_relative),
        "source_files": _source_hashes(root),
        "artifacts": artifact_records,
        "response_bundle_manifest": {
            "path": manifest_path.resolve().relative_to(root).as_posix(),
            "sha256": _sha256(manifest_path),
            "metadata": bundle_metadata,
        },
    }


def analyzer_checkpoint_attestations(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the detailed artifact inventory to the analyzer's manifest contract."""
    artifacts = metadata.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    attestations: dict[str, Any] = {}
    for name, expected_path in ANALYZER_CHECKPOINTS.items():
        entries = artifacts.get(name)
        entries = entries if isinstance(entries, list) else []
        digest: str | None = None
        if entries and all(isinstance(row, Mapping) for row in entries):
            if name == "gemma_qlora":
                # Directory checkpoints have no single file digest. Hash the ordered
                # relative-path/content-digest inventory deterministically.
                directory_digest = hashlib.sha256()
                prefix = expected_path.rstrip("/") + "/"
                valid = True
                for row in sorted(entries, key=lambda value: str(value.get("path", ""))):
                    path = str(row.get("path", "")).replace("\\", "/")
                    content_digest = str(row.get("sha256", ""))
                    if not path.startswith(prefix) or not re.fullmatch(r"[0-9a-fA-F]{64}", content_digest):
                        valid = False
                        break
                    directory_digest.update(path[len(prefix):].encode("utf-8"))
                    directory_digest.update(b"\0")
                    directory_digest.update(content_digest.lower().encode("ascii"))
                    directory_digest.update(b"\0")
                if valid:
                    digest = directory_digest.hexdigest()
            elif len(entries) == 1 and str(entries[0].get("path", "")).replace("\\", "/") == expected_path:
                candidate = str(entries[0].get("sha256", ""))
                if re.fullmatch(r"[0-9a-fA-F]{64}", candidate):
                    digest = candidate.lower()
        attestations[name] = {"path": expected_path, "sha256": digest}
    return attestations


def docker_image_identity(image: str) -> dict[str, Any]:
    result = _command(["docker", "image", "inspect", "--format", "{{json .}}", image])
    try:
        info = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"docker image inspect returned invalid JSON for {image}") from exc
    if not isinstance(info, dict) or not info.get("Id"):
        raise RuntimeError(f"docker image identity unavailable for {image}")
    return {"reference": image, "id": info["Id"], "repo_digests": info.get("RepoDigests", []), "created": info.get("Created")}


def _health(url: str, *, timeout_s: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            if response.status != 200:
                return None
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    return body if isinstance(body, dict) else None


def wait_for_server_ready(
    repository_root: Path,
    container_id: str,
    log_path: Path,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_health: dict[str, Any] | None = None
    log_text = ""
    while time.monotonic() < deadline:
        state_result = _command(
            ["docker", "inspect", "--format", "{{json .State}}", container_id],
            cwd=repository_root,
            check=False,
        )
        if state_result.returncode != 0:
            raise RuntimeError(
                f"unable to inspect server container {container_id}: "
                f"{(state_result.stderr or state_result.stdout or '').strip()}; inspect {log_path}"
            )
        try:
            container_state = json.loads(state_result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid Docker state for server container {container_id}") from exc
        if not isinstance(container_state, Mapping):
            raise RuntimeError(f"invalid Docker state for server container {container_id}")
        if container_state.get("Status") in {"exited", "dead"} or (
            container_state.get("Running") is False
            and int(container_state.get("ExitCode", 0) or 0) != 0
        ):
            raise RuntimeError(
                f"server container exited: Status={container_state.get('Status')} "
                f"ExitCode={container_state.get('ExitCode')}; inspect {log_path}"
            )
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            log_text = ""
        last_health = _health("http://127.0.0.1:8092/health")
        if is_server_ready(log_text, last_health):
            return last_health or {}
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"server not ready after {timeout_s:g}s; load_marker={LOAD_MARKER in log_text}, "
        f"health={last_health!r}; inspect {log_path}"
    )


def snapshot_maintenance_container(container_name: str, repository_root: Path) -> dict[str, Any]:
    inspected = _command(["docker", "inspect", container_name], cwd=repository_root)
    try:
        data = json.loads(inspected.stdout or "[]")
        original = data[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unable to inspect maintenance container {container_name}") from exc
    if original.get("State", {}).get("Running") is not True:
        if str(original.get("Name", "")).lstrip("/") != container_name:
            raise RuntimeError("maintenance container name did not resolve to the requested managed service")
        return {"id": original.get("Id"), "name": container_name, "was_running": False, "inspect": original}
    if str(original.get("Name", "")).lstrip("/") != container_name:
        raise RuntimeError("maintenance container name did not resolve to the requested managed service")
    container_id = str(original.get("Id") or "")
    if not container_id:
        raise RuntimeError("maintenance container inspect did not contain its immutable ID")
    _assert_maintenance_idle(container_id, repository_root)
    return {"id": container_id, "name": container_name, "was_running": True, "inspect": original}


def pause_maintenance_container(snapshot: dict[str, Any], repository_root: Path) -> dict[str, Any] | None:
    if snapshot.get("was_running"):
        container_id = str(snapshot.get("id") or "")
        if not container_id:
            raise RuntimeError("maintenance snapshot has no immutable container ID")
        # Recheck immediately before stop so a peer that connected after the
        # initial snapshot prevents maintenance from interrupting that session.
        _assert_maintenance_idle(container_id, repository_root)
        memory_before = nvidia_smi_memory_snapshot(repository_root)
        top = _command(["docker", "top", container_id, "-eo", "pid,args"], cwd=repository_root)
        container_pids = {
            fields[0] for line in (top.stdout or "").splitlines()
            if (fields := line.strip().split(maxsplit=1)) and fields[0].isdigit()
        }
        gpu_apps = memory_before["compute_apps"]
        snapshot["gpu_memory_before_stop"] = memory_before
        snapshot["gpu_process_pids_before_stop"] = sorted(
            app["pid"] for app in gpu_apps if app["pid"] in container_pids
        )
        outside = [app for app in gpu_apps if app["pid"] not in container_pids]
        if outside:
            snapshot["gpu_processes_outside_container_before_stop"] = outside
            raise RuntimeError(
                f"GPU ownership changed before managed stop; refusing to launch: {outside!r}"
            )
        _command(["docker", "stop", "--time", "20", container_id], cwd=repository_root)
        return memory_before
    return None


def container_tcp_tables_probe_code() -> str:
    return (
        "import json\n"
        "from pathlib import Path\n"
        "tables = {}\n"
        "for name in ('tcp', 'tcp6'):\n"
        "    path = Path('/proc/net') / name\n"
        "    try:\n"
        "        tables[name] = path.read_text(encoding='ascii')\n"
        "    except FileNotFoundError:\n"
        "        tables[name] = None\n"
        "print(json.dumps(tables))\n"
    )


def parse_proc_net_tcp_tables(tables: Mapping[str, Any], *, port: int = 8000) -> list[dict[str, str]]:
    """Return established sockets whose local port matches, failing closed on malformed tables."""
    ipv4 = tables.get("tcp")
    if not isinstance(ipv4, str):
        raise RuntimeError("/proc/net/tcp table is missing or unreadable")
    active: list[dict[str, str]] = []
    for name in ("tcp", "tcp6"):
        content = tables.get(name)
        if content is None and name == "tcp6":
            continue
        if not isinstance(content, str):
            raise RuntimeError(f"/proc/net/{name} table is unreadable")
        lines = content.splitlines()
        if not lines or not {"local_address", "st"}.issubset(set(lines[0].split())):
            raise RuntimeError(f"malformed /proc/net/{name} header")
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 4:
                raise RuntimeError(f"malformed /proc/net/{name} row: {line!r}")
            try:
                _local_host, local_port = fields[1].rsplit(":", 1)
                local_port_number = int(local_port, 16)
                state_number = int(fields[3], 16)
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"malformed /proc/net/{name} socket fields: {line!r}") from exc
            if local_port_number == port and state_number == 1:
                active.append({
                    "table": name,
                    "local_address": fields[1],
                    "remote_address": fields[2],
                    "state": fields[3].upper(),
                })
    return active


def _assert_maintenance_idle(container_id: str, repository_root: Path) -> None:
    host_connections = _command(
        ["ss", "-Htnp", "state", "established", "( sport = :8000 )"], cwd=repository_root,
    ).stdout or ""
    container_tables_raw = _command(
        [
            "docker", "exec", container_id,
            "/workspace/.venv_e2e_runtime/bin/python", "-c", container_tcp_tables_probe_code(),
        ],
        cwd=repository_root,
    ).stdout or ""
    try:
        container_tables = json.loads(container_tables_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("container /proc/net/tcp probe returned invalid JSON") from exc
    if not isinstance(container_tables, dict):
        raise RuntimeError("container /proc/net/tcp probe returned an invalid payload")
    container_connections = parse_proc_net_tcp_tables(container_tables)
    if host_connections.strip() or container_connections:
        raise RuntimeError("maintenance service has established port 8000 clients; refusing to stop it")

    pids_result = _command(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], cwd=repository_root,
    )
    gpu_pids = set()
    for line in (pids_result.stdout or "").splitlines():
        value = line.strip()
        if not value or "no running processes" in value.casefold():
            continue
        if not value.isdigit():
            raise RuntimeError(f"cannot verify GPU process ownership from nvidia-smi output: {value!r}")
        gpu_pids.add(value)
    top = _command(["docker", "top", container_id, "-eo", "pid,args"], cwd=repository_root)
    container_pids = {
        fields[0] for line in (top.stdout or "").splitlines()
        if (fields := line.strip().split(maxsplit=1)) and fields[0].isdigit()
    }
    outside = gpu_pids - container_pids
    if outside:
        raise RuntimeError(f"GPU processes outside maintenance container are active: {sorted(outside)}")

    utilization = _command(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], cwd=repository_root,
    )
    util_values = []
    for line in (utilization.stdout or "").splitlines():
        match = re.fullmatch(r"\s*(\d+)\s*", line)
        if not match:
            raise RuntimeError(f"cannot verify GPU idle state from nvidia-smi output: {line!r}")
        util_values.append(int(match.group(1)))
    if not util_values or any(value > 5 for value in util_values):
        raise RuntimeError(f"GPU is not idle enough to pause maintenance container: {util_values!r}")

    processes = _command(["ps", "-eo", "pid=,args="], cwd=repository_root).stdout or ""
    training = [line.strip() for line in processes.splitlines() if re.search(
        r"(?:train[^/\s]*\.py|run_3090\.py[^\n]*--phase\s+train|accelerate\s+launch[^\n]*train)",
        line, re.IGNORECASE,
    )]
    if training:
        raise RuntimeError(f"training process is active; refusing to pause service: {training[:5]}")


def parse_nvidia_smi_snapshot(gpu_memory_csv: str, compute_apps_csv: str) -> dict[str, Any]:
    """Parse the no-header, no-units CSV form of the two nvidia-smi queries."""
    gpus: list[dict[str, int]] = []
    for row in csv.reader(gpu_memory_csv.splitlines(), skipinitialspace=True):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) != 5:
            raise RuntimeError(f"cannot parse nvidia-smi GPU memory row: {row!r}")
        try:
            index, total, used, free, utilization = (int(field.strip()) for field in row)
        except ValueError as exc:
            raise RuntimeError(f"cannot parse nvidia-smi GPU memory row: {row!r}") from exc
        gpus.append({
            "index": index,
            "memory_total_mib": total,
            "memory_used_mib": used,
            "memory_free_mib": free,
            "utilization_pct": utilization,
        })
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU memory rows")

    compute_apps: list[dict[str, Any]] = []
    for row in csv.reader(compute_apps_csv.splitlines(), skipinitialspace=True):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) == 1 and "no running processes" in row[0].casefold():
            continue
        if len(row) != 3:
            raise RuntimeError(f"cannot parse nvidia-smi compute-app row: {row!r}")
        pid, process_name, used_memory = (field.strip() for field in row)
        if not pid.isdigit() or not used_memory.isdigit():
            raise RuntimeError(f"cannot parse nvidia-smi compute-app row: {row!r}")
        compute_apps.append({
            "pid": pid,
            "process_name": process_name,
            "used_gpu_memory_mib": int(used_memory),
        })
    return {"gpus": gpus, "compute_apps": compute_apps}


def nvidia_smi_memory_snapshot(
    repository_root: Path,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    def command_timeout() -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("GPU release barrier deadline expired before nvidia-smi completed")
        return remaining

    gpu_memory = _command(
        [
            "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], cwd=repository_root, timeout=command_timeout(),
    )
    compute_apps = _command(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], cwd=repository_root, timeout=command_timeout(),
    )
    snapshot = parse_nvidia_smi_snapshot(gpu_memory.stdout or "", compute_apps.stdout or "")
    snapshot["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    return snapshot


def assert_gpu_vram_released(
    maintenance_snapshot: Mapping[str, Any],
    memory_snapshot: Mapping[str, Any],
    *,
    min_free_mib: int = GPU_RELEASE_MIN_FREE_MIB,
) -> None:
    managed_pids = {
        str(pid) for pid in maintenance_snapshot.get("gpu_process_pids_before_stop", [])
    }
    compute_apps = memory_snapshot.get("compute_apps")
    if not isinstance(compute_apps, list):
        raise RuntimeError("GPU memory snapshot has no valid compute-app list")
    resident_managed = [
        app for app in compute_apps
        if isinstance(app, Mapping) and str(app.get("pid", "")) in managed_pids
    ]
    if resident_managed:
        raise RuntimeError(f"managed GPU process is still resident after stop: {resident_managed!r}")
    unrelated = [
        app for app in compute_apps
        if isinstance(app, Mapping) and str(app.get("pid", "")) not in managed_pids
    ]
    if unrelated:
        raise RuntimeError(f"unrelated GPU compute process remains after managed stop: {unrelated!r}")
    gpus = memory_snapshot.get("gpus")
    if not isinstance(gpus, list):
        raise RuntimeError("GPU memory snapshot has no valid GPU list")
    primary_gpu = next((gpu for gpu in gpus if isinstance(gpu, Mapping) and gpu.get("index") == 0), None)
    if primary_gpu is None:
        raise RuntimeError("GPU memory snapshot did not include GPU 0")
    free_mib = primary_gpu.get("memory_free_mib")
    if not isinstance(free_mib, int) or free_mib < min_free_mib:
        raise RuntimeError(
            f"GPU 0 has only {free_mib!r} MiB free after managed stop; "
            f"need at least {min_free_mib} MiB before model load"
        )


def assert_gpu_ready_without_maintenance(
    memory_snapshot: Mapping[str, Any],
    *,
    min_free_mib: int = GPU_RELEASE_MIN_FREE_MIB,
) -> None:
    compute_apps = memory_snapshot.get("compute_apps")
    if not isinstance(compute_apps, list):
        raise RuntimeError("GPU memory snapshot has no valid compute-app list")
    if compute_apps:
        raise RuntimeError(
            "GPU compute process exists without a maintenance container; refusing model load: "
            f"{compute_apps!r}"
        )
    gpus = memory_snapshot.get("gpus")
    if not isinstance(gpus, list):
        raise RuntimeError("GPU memory snapshot has no valid GPU list")
    primary_gpu = next((gpu for gpu in gpus if isinstance(gpu, Mapping) and gpu.get("index") == 0), None)
    if primary_gpu is None:
        raise RuntimeError("GPU memory snapshot did not include GPU 0")
    free_mib = primary_gpu.get("memory_free_mib")
    if not isinstance(free_mib, int) or free_mib < min_free_mib:
        raise RuntimeError(
            f"GPU 0 has only {free_mib!r} MiB free without a managed runtime; "
            f"need at least {min_free_mib} MiB before model load"
        )


class GpuVramBarrierError(RuntimeError):
    def __init__(self, message: str, observations: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.observations = [dict(observation) for observation in observations]


def wait_for_gpu_vram_release(
    maintenance_snapshot: Mapping[str, Any],
    repository_root: Path,
    *,
    timeout_s: float = GPU_RELEASE_TIMEOUT_S,
    poll_interval_s: float = GPU_RELEASE_POLL_S,
) -> dict[str, Any]:
    if maintenance_snapshot.get("was_running") is not True:
        raise RuntimeError("refusing GPU launch without a previously running managed service")
    container_id = str(maintenance_snapshot.get("id") or "")
    if not container_id:
        raise RuntimeError("GPU release barrier requires the immutable managed container ID")
    deadline = time.monotonic() + timeout_s
    observations: list[dict[str, Any]] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            blocker = observations[-1].get("blocker", "GPU release verification exceeded its deadline") if observations else "GPU release verification exceeded its deadline"
            raise GpuVramBarrierError(str(blocker), observations)
        try:
            inspected = _command(
                ["docker", "inspect", container_id], cwd=repository_root, timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            raise GpuVramBarrierError(
                "GPU release barrier timed out inspecting the managed container", observations,
            ) from exc
        try:
            data = json.loads(inspected.stdout or "[]")
            state = data[0]["State"]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            raise GpuVramBarrierError(
                "unable to verify managed container state after stop", observations,
            ) from exc
        if not data or data[0].get("Id") != container_id:
            raise GpuVramBarrierError("managed container identity changed after stop", observations)
        if state.get("Running") is not False:
            raise GpuVramBarrierError(
                f"managed container {container_id} is still running after stop", observations,
            )
        try:
            memory_snapshot = nvidia_smi_memory_snapshot(repository_root, deadline=deadline)
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise GpuVramBarrierError(
                "GPU release barrier timed out collecting nvidia-smi state", observations,
            ) from exc
        live_managed_pids = []
        for pid in maintenance_snapshot.get("gpu_process_pids_before_stop", []):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GpuVramBarrierError(
                    "GPU release barrier deadline expired checking managed PIDs", observations,
                )
            try:
                live = _command(
                    ["ps", "-p", str(pid), "-o", "pid="], cwd=repository_root,
                    check=False, timeout=remaining,
                )
            except subprocess.TimeoutExpired as exc:
                raise GpuVramBarrierError(
                    "GPU release barrier timed out checking managed PIDs", observations,
                ) from exc
            if (live.stdout or "").strip():
                live_managed_pids.append(str(pid))
        observation = {
            "container_id": container_id,
            "container_running": False,
            "memory": memory_snapshot,
            "managed_pids_still_alive": live_managed_pids,
        }
        observations.append(observation)
        if live_managed_pids:
            observation["blocker"] = f"managed process PID(s) still alive after stop: {live_managed_pids!r}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GpuVramBarrierError(observation["blocker"], observations)
            time.sleep(min(poll_interval_s, remaining))
            continue
        try:
            assert_gpu_vram_released(maintenance_snapshot, memory_snapshot)
        except RuntimeError as exc:
            observation["blocker"] = str(exc)
            if str(exc).startswith("unrelated GPU compute process remains"):
                raise GpuVramBarrierError(str(exc), observations) from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GpuVramBarrierError(str(exc), observations) from exc
            time.sleep(min(poll_interval_s, remaining))
            continue
        return {
            "released": True,
            "required_free_mib": GPU_RELEASE_MIN_FREE_MIB,
            "observations": observations,
        }


def restore_maintenance_container(snapshot: Mapping[str, Any], *, timeout_s: float = 300.0) -> None:
    if not snapshot.get("was_running"):
        return
    container_id = str(snapshot.get("id") or "")
    if not container_id:
        raise RuntimeError("maintenance snapshot has no original container ID")
    inspect = _command(["docker", "inspect", container_id])
    data = json.loads(inspect.stdout or "[]")
    if not data or data[0].get("Id") != container_id:
        raise RuntimeError("maintenance container identity changed; refusing to restart by name")
    if data[0].get("State", {}).get("Running") is not True:
        _command(["docker", "start", container_id])
        inspect = _command(["docker", "inspect", container_id])
        data = json.loads(inspect.stdout or "[]")
        if not data or data[0].get("Id") != container_id or data[0].get("State", {}).get("Running") is not True:
            raise RuntimeError("maintenance container did not restart with its original ID")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        health = _health("http://127.0.0.1:8000/health")
        if health and health.get("status") == "ok" and health.get("engine_loaded") is True:
            return
        time.sleep(2.0)
    raise TimeoutError(f"restored container {container_id} did not become healthy on port 8000")


def launch_server_container(
    repository_root: Path,
    container_name: str,
    image: str,
    server_argv: Sequence[str],
    code2wav_graph_family: bool = False,
    code2wav_graph_odd_tail: bool = False,
    native_listener_capacity_candidate: bool = False,
    native_listener_attestation_dir: str | None = None,
    native_listener_cuda_profile: bool = False,
    native_listener_cuda_profile_directory: str | None = None,
    first_input_scheduler_anchor: bool = False,
    ready_frame_scheduler_dispatch: bool = False,
    listener_pair_lock_handoff: bool = False,
    native_control6_fused_readback: bool = False,
) -> str:
    docker_argv = build_docker_run_argv(
        repository_root=repository_root,
        container_name=container_name,
        image=image,
        server_argv=server_argv,
        code2wav_graph_family=code2wav_graph_family,
        code2wav_graph_odd_tail=code2wav_graph_odd_tail,
        native_listener_capacity_candidate=native_listener_capacity_candidate,
        native_listener_attestation_dir=native_listener_attestation_dir,
        native_listener_cuda_profile=native_listener_cuda_profile,
        native_listener_cuda_profile_directory=native_listener_cuda_profile_directory,
        first_input_scheduler_anchor=first_input_scheduler_anchor,
        ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
        listener_pair_lock_handoff=listener_pair_lock_handoff,
        native_control6_fused_readback=native_control6_fused_readback,
    )
    result = _command(docker_argv, cwd=repository_root)
    container_id = (result.stdout or "").strip()
    if not container_id:
        raise RuntimeError("docker run succeeded without returning the new container ID")
    return container_id


def start_server_log_capture(container_id: str, repository_root: Path, log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            ["docker", "logs", "--follow", "--timestamps", container_id],
            cwd=repository_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        handle.close()
        raise
    return process, handle


def stop_server_log_capture(capture: tuple[subprocess.Popen[bytes], Any] | None) -> None:
    if capture is None:
        return
    process, handle = capture
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle.close()


def cleanup_owned_container(container_id: str | None, container_name: str, repository_root: Path) -> None:
    if not container_id:
        inspected = _command(["docker", "inspect", container_name], cwd=repository_root, check=False)
        if inspected.returncode != 0:
            return
        data = json.loads(inspected.stdout or "[]")
        if not data:
            return
        labels = data[0].get("Config", {}).get("Labels") or {}
        if labels.get(OWNERSHIP_LABEL) != OWNERSHIP_VALUE:
            raise RuntimeError("refusing cleanup: unique container name is not owned by this task")
        container_id = str(data[0].get("Id") or "")
    if not container_id:
        return
    inspected = _command(["docker", "inspect", container_id], cwd=repository_root, check=False)
    if inspected.returncode == 0:
        data = json.loads(inspected.stdout or "[]")
        if data:
            labels = data[0].get("Config", {}).get("Labels") or {}
            if labels.get(OWNERSHIP_LABEL) != OWNERSHIP_VALUE:
                raise RuntimeError(f"refusing cleanup of container without task ownership label: {container_id}")
            _command(["docker", "stop", "--time", "20", container_id], cwd=repository_root, check=False)
            _command(["docker", "rm", container_id], cwd=repository_root)


def record_runtime_versions(
    container_id: str, *, native_listener_capacity_candidate: bool = False,
) -> dict[str, Any]:
    result = _command([
        "docker", "exec", container_id,
        "/workspace/.venv_e2e_runtime/bin/python", "-c",
        runtime_version_probe_code(
            native_listener_capacity_candidate=native_listener_capacity_candidate,
        ),
    ])
    try:
        value = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("container runtime version probe returned invalid JSON") from exc
    if not isinstance(value, dict) or not value.get("python"):
        raise RuntimeError("container runtime version probe omitted Python version")
    if native_listener_capacity_candidate:
        candidate = value.get("native_listener_candidate")
        if not isinstance(candidate, Mapping):
            raise RuntimeError("server container omitted native listener candidate attestation")
        value["native_listener_candidate"] = validate_native_listener_candidate(candidate)
    return value


def build_runtime_probe_argv(
    repository_root: str | Path,
    image: str,
    *,
    code2wav_graph_family: bool = False,
    code2wav_graph_odd_tail: bool = False,
    native_listener_capacity_candidate: bool = False,
    native_listener_cuda_profile: bool = False,
    native_listener_cuda_profile_directory: str | None = None,
    first_input_scheduler_anchor: bool = False,
    ready_frame_scheduler_dispatch: bool = False,
    listener_pair_lock_handoff: bool = False,
    native_control6_fused_readback: bool = False,
) -> list[str]:
    """Probe the exact image-side interpreter with the planned workspace mount, without a GPU."""
    validate_code2wav_graph_options(
        code2wav_graph_family=code2wav_graph_family,
        code2wav_graph_odd_tail=code2wav_graph_odd_tail,
    )
    validate_scheduler_mode(
        first_input_scheduler_anchor=first_input_scheduler_anchor,
        ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
    )
    profile_directory = (
        validated_native_listener_cuda_profile_directory(native_listener_cuda_profile_directory)
        if native_listener_cuda_profile else None
    )
    if not native_listener_cuda_profile and native_listener_cuda_profile_directory is not None:
        raise ValueError("native listener CUDA profile directory requires profile opt-in")
    graph_environment = (
        [
            "--env", CODE2WAV_GRAPH_FAMILY_ENV,
            "--env", CODE2WAV_DECODE_FRAMES_ENV,
            "--env", CODE2WAV_GRAPH_ODD_TAIL_ENV if code2wav_graph_odd_tail
            else CODE2WAV_GRAPH_EVEN_ONLY_ENV,
        ]
        if code2wav_graph_family else []
    )
    native_listener_environment = (
        ["--env", f"{NATIVE_LISTENER_LIBRARY_PATH_ENV}={NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH}"]
        if native_listener_capacity_candidate else []
    )
    native_listener_profile_environment = [
        "--env", f"{NATIVE_LISTENER_CUDA_PROFILE_ENV}={int(native_listener_cuda_profile)}",
    ]
    if native_listener_cuda_profile:
        native_listener_profile_environment.extend([
            "--env",
            f"{NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV}={profile_directory}",
        ])
    return [
        "docker", "run", "--rm",
        "--user", "1000:1000",
        "--ipc", "host",
        "--env", NATIVE_TALKER_Q1_ENV,
        "--env", NATIVE_TALKER_Q1_GRAPH_ENV,
        "--env", "DUPLEX_NATIVE_CONTROL6_TRACE=0",
        *graph_environment,
        *native_listener_environment,
        *native_listener_profile_environment,
        "--env", f"{FIRST_INPUT_SCHEDULER_ANCHOR_ENV}={int(first_input_scheduler_anchor)}",
        "--env", f"{READY_FRAME_SCHEDULER_DISPATCH_ENV}={int(ready_frame_scheduler_dispatch)}",
        "--env", f"{LISTENER_PAIR_LOCK_HANDOFF_ENV}={int(listener_pair_lock_handoff)}",
        "--env", f"{NATIVE_CONTROL6_FUSED_READBACK_ENV}={int(native_control6_fused_readback)}",
        "--env", f"QWEN_NATIVE_LIBRARY_PATH={NATIVE_TALKER_LIBRARY_CONTAINER_PATH}",
        "--volume", f"{str(Path(repository_root).resolve()).replace(chr(92), '/') }:/workspace",
        "--workdir", "/workspace",
        "--pull=never",
        "--entrypoint", "/workspace/.venv_e2e_runtime/bin/python",
        image,
        "-c", runtime_version_probe_code(
            native_listener_capacity_candidate=native_listener_capacity_candidate,
        ),
    ]


def probe_container_runtime(
    repository_root: str | Path,
    image: str,
    *,
    code2wav_graph_family: bool = False,
    code2wav_graph_odd_tail: bool = False,
    native_listener_capacity_candidate: bool = False,
    native_listener_cuda_profile: bool = False,
    native_listener_cuda_profile_directory: str | None = None,
    first_input_scheduler_anchor: bool = False,
    ready_frame_scheduler_dispatch: bool = False,
    listener_pair_lock_handoff: bool = False,
    native_control6_fused_readback: bool = False,
) -> dict[str, Any]:
    """Verify runtime metadata in the same image/mount boundary used by the server."""
    result = _command(build_runtime_probe_argv(
        repository_root, image,
        code2wav_graph_family=code2wav_graph_family,
        code2wav_graph_odd_tail=code2wav_graph_odd_tail,
        native_listener_capacity_candidate=native_listener_capacity_candidate,
        native_listener_cuda_profile=native_listener_cuda_profile,
        native_listener_cuda_profile_directory=native_listener_cuda_profile_directory,
        first_input_scheduler_anchor=first_input_scheduler_anchor,
        ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
        listener_pair_lock_handoff=listener_pair_lock_handoff,
        native_control6_fused_readback=native_control6_fused_readback,
    ))
    try:
        value = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("container runtime preflight returned invalid version JSON") from exc
    if not isinstance(value, dict) or not value.get("python"):
        raise RuntimeError("container runtime preflight omitted Python version")
    packages = value.get("packages")
    if not isinstance(packages, dict) or not packages.get("torch") or not packages.get("transformers"):
        raise RuntimeError("container runtime preflight omitted torch/transformers versions")
    native_talker = value.get("native_talker")
    if not isinstance(native_talker, Mapping):
        raise RuntimeError("container runtime preflight omitted native Talker library attestation")
    value["native_talker"] = validate_native_talker_binary(native_talker)
    if native_listener_capacity_candidate:
        native_listener = value.get("native_listener_candidate")
        if not isinstance(native_listener, Mapping):
            raise RuntimeError("candidate mode preflight omitted native listener DSO attestation")
        value["native_listener_candidate"] = validate_native_listener_candidate(native_listener)
    if value.get("talker_native_q1_setting") != "1":
        raise RuntimeError("native Talker Q1 must be enabled for the pair proof")
    if value.get("talker_native_q1_graph_setting") != "1":
        raise RuntimeError("native Q1 graph mode is required for the pair proof")
    expected_profile_setting = "1" if native_listener_cuda_profile else "0"
    expected_profile_directory = (
        validated_native_listener_cuda_profile_directory(native_listener_cuda_profile_directory)
        if native_listener_cuda_profile else None
    )
    if (
        value.get("native_listener_cuda_profile_setting") != expected_profile_setting
        or value.get("native_listener_cuda_profile_enabled") is not native_listener_cuda_profile
        or value.get("native_listener_cuda_profile_directory_setting") != expected_profile_directory
    ):
        raise RuntimeError(
            "native listener CUDA profile CPU preflight differs from the requested flag/directory"
        )
    if value.get("native_control6_trace_setting") != "0":
        raise RuntimeError("DUPLEX_NATIVE_CONTROL6_TRACE must be explicitly disabled for the pair proof")
    if code2wav_graph_family and (
        value.get("code2wav_graph_family_setting") != "1"
        or value.get("code2wav_decode_frames_setting") != "2"
        or value.get("code2wav_graph_odd_tail_setting")
        != ("1" if code2wav_graph_odd_tail else "0")
    ):
        raise RuntimeError(
            "Code2Wav graph family CPU preflight must attest family=1, decode-frames=2, "
            f"and odd-tail={int(code2wav_graph_odd_tail)}"
        )
    expected_anchor_setting = "1" if first_input_scheduler_anchor else "0"
    if (
        value.get("first_input_scheduler_anchor_setting") != expected_anchor_setting
        or value.get("first_input_scheduler_anchor_enabled") is not first_input_scheduler_anchor
    ):
        raise RuntimeError(
            "first-input scheduler anchor CPU preflight must attest the requested environment "
            f"and resolved setting {expected_anchor_setting}"
        )
    expected_ready_setting = "1" if ready_frame_scheduler_dispatch else "0"
    if (
        value.get("ready_frame_scheduler_dispatch_setting") != expected_ready_setting
        or value.get("ready_frame_scheduler_dispatch_enabled") is not ready_frame_scheduler_dispatch
    ):
        raise RuntimeError(
            "ready-frame scheduler dispatch CPU preflight must attest the requested environment "
            f"and resolved setting {expected_ready_setting}"
        )
    expected_pair_handoff_setting = "1" if listener_pair_lock_handoff else "0"
    if (
        value.get("listener_pair_lock_handoff_setting") != expected_pair_handoff_setting
        or value.get("listener_pair_lock_handoff_enabled") is not listener_pair_lock_handoff
    ):
        raise RuntimeError(
            "listener pair lock handoff CPU preflight must attest the requested environment "
            f"and resolved setting {expected_pair_handoff_setting}"
        )
    expected_fused_readback_setting = "1" if native_control6_fused_readback else "0"
    if (
        value.get("native_control6_fused_readback_setting") != expected_fused_readback_setting
        or value.get("native_control6_fused_readback_enabled") is not native_control6_fused_readback
    ):
        raise RuntimeError(
            "native Control6 fused-readback CPU preflight must attest the requested environment "
            f"and resolved setting {expected_fused_readback_setting}"
        )
    return value


def build_cli_smoke_argv(
    repository_root: str | Path,
    image: str,
    script: str,
    *,
    workspace_pythonpath: bool = False,
    module: bool = False,
) -> list[str]:
    """Build a disposable CPU-only `--help` invocation for an actual entrypoint."""
    argv = [
        "docker", "run", "--rm",
        "--user", "1000:1000",
        "--ipc", "host",
        "--volume", f"{str(Path(repository_root).resolve()).replace(chr(92), '/') }:/workspace",
        "--workdir", "/workspace",
        "--pull=never",
    ]
    if workspace_pythonpath:
        argv.extend(["--env", "PYTHONPATH=/workspace"])
    argv.extend(["--entrypoint", "/workspace/.venv_e2e_runtime/bin/python", image])
    if module:
        argv.extend(["-m", script])
    else:
        argv.append(script)
    argv.append("--help")
    return argv


def probe_cli_entrypoints(repository_root: str | Path, image: str) -> dict[str, Any]:
    """Verify server/client CLI imports before any managed-service pause."""
    entrypoints = {
        "server": ("duplex_kernel.launch", False, True),
        "client": ("client/duplex_stream_e2e.py", True, False),
    }
    result: dict[str, Any] = {}
    for name, (script, workspace_pythonpath, module) in entrypoints.items():
        argv = build_cli_smoke_argv(
            repository_root,
            image,
            script,
            workspace_pythonpath=workspace_pythonpath,
            module=module,
        )
        completed = _command(argv, cwd=repository_root, check=False)
        output = (completed.stdout or "") + (completed.stderr or "")
        first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
        if completed.returncode != 0 or not first_line.startswith("usage:"):
            detail = output[-1200:].strip()
            raise RuntimeError(
                f"{name} CLI smoke failed (exit={completed.returncode}): {detail or 'usage output missing'}"
            )
        result[name] = {
            "script": script,
            "module": module,
            "workspace_pythonpath": workspace_pythonpath,
            "usage": first_line,
        }
    return result


def runtime_version_probe_code(*, native_listener_capacity_candidate: bool = False) -> str:
    candidate_probe = (
        native_listener_candidate_probe_code()
        + "native_listener_candidate_record = native_listener_candidate\n"
        if native_listener_capacity_candidate
        else "native_listener_candidate_record = None\n"
    )
    return (
        native_talker_binary_probe_code()
        + candidate_probe
        + "import importlib.metadata as metadata, json, platform\n"
        + "from duplex_kernel.config import FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED\n"
        + "from duplex_kernel.config import READY_FRAME_SCHEDULER_DISPATCH_ENABLED\n"
        + "from duplex_kernel.config import LISTENER_PAIR_LOCK_HANDOFF_ENABLED\n"
        + "from duplex_kernel.config import NATIVE_LISTENER_CUDA_PROFILE_ENABLED\n"
        + "from duplex_kernel.config import NATIVE_CONTROL6_FUSED_READBACK_ENABLED\n"
        "names = ['torch', 'transformers', 'peft', 'unsloth', 'websockets', 'fastapi', 'uvicorn']\n"
        "versions = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        versions[name] = metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        versions[name] = None\n"
        "print(json.dumps({'python': platform.python_version(), 'packages': versions, "
        "'native_talker': native_talker_library, "
        "'native_listener_candidate': native_listener_candidate_record, "
        "'talker_native_q1_setting': os.environ.get('DUPLEX_TALKER_NATIVE_AUDIO_Q1'), "
        "'talker_native_q1_graph_setting': os.environ.get('DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH'), "
        "'native_control6_trace_setting': os.environ.get('DUPLEX_NATIVE_CONTROL6_TRACE'), "
        "'code2wav_graph_family_setting': os.environ.get('GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY'), "
        "'code2wav_decode_frames_setting': os.environ.get('GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES'), "
        "'code2wav_graph_odd_tail_setting': os.environ.get('GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL'), "
        "'first_input_scheduler_anchor_setting': os.environ.get('GEMMA4_OMNI_FIRST_INPUT_SCHEDULER_ANCHOR'), "
        "'first_input_scheduler_anchor_enabled': FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED, "
        "'ready_frame_scheduler_dispatch_setting': os.environ.get('GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH'), "
        "'ready_frame_scheduler_dispatch_enabled': READY_FRAME_SCHEDULER_DISPATCH_ENABLED, "
        "'listener_pair_lock_handoff_setting': os.environ.get('GEMMA4_OMNI_LISTENER_PAIR_LOCK_HANDOFF'), "
        "'listener_pair_lock_handoff_enabled': LISTENER_PAIR_LOCK_HANDOFF_ENABLED, "
        "'native_listener_cuda_profile_setting': os.environ.get('GEMMA4_NATIVE_LISTENER_CUDA_PROFILE'), "
        "'native_listener_cuda_profile_directory_setting': os.environ.get('GEMMA4_NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY'), "
        "'native_listener_cuda_profile_enabled': NATIVE_LISTENER_CUDA_PROFILE_ENABLED, "
        "'native_control6_fused_readback_setting': os.environ.get('GEMMA4_OMNI_NATIVE_CONTROL6_FUSED_READBACK'), "
        "'native_control6_fused_readback_enabled': NATIVE_CONTROL6_FUSED_READBACK_ENABLED}, "
        "sort_keys=True))\n"
    )


def run_client(
    repository_root: Path,
    container_id: str,
    argv: Sequence[str],
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the client with the mounted workspace importable as a package.

    Docker's exec reference says ``--env`` adds or overrides environment
    variables for only that exec process: https://docs.docker.com/reference/cli/docker/container/exec/.
    """
    with log_path.open("w", encoding="utf-8") as log:
        return _command(["docker", "exec", "--env", "PYTHONPATH=/workspace", container_id, *argv], cwd=repository_root,
                        check=False, stdout=log, stderr=subprocess.STDOUT)


def run_analyzer(
    repository_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    script = repository_root / "tools" / "analyze_pair_duplex_certification.py"
    argv = [sys.executable, str(script), "--input-dir", str(output_dir)]
    asr_artifact = output_dir / "asr_transcripts.json"
    if asr_artifact.is_file():
        argv.extend(["--asr-json", str(asr_artifact)])
    analyzer_status = "not_installed" if not script.is_file() else "unavailable"
    analyzer_error: str | None = None
    analyzer_stdout = ""
    analyzer_stderr = ""
    try:
        if script.is_file():
            result = _command(argv, cwd=repository_root, check=False)
            analyzer_stdout = (result.stdout or "")[-4000:]
            analyzer_stderr = (result.stderr or "")[-4000:]
            analyzer_status = "completed" if result.returncode == 0 else "failed"
    except Exception as exc:
        analyzer_error = f"{type(exc).__name__}: {exc}"
    report = output_dir / "certification_metrics.json"
    if report.is_file():
        try:
            metrics = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
    else:
        metrics = {}
    legacy_report_path = output_dir / "legacy_pair_duplex_certification_report.md"
    analyzer_markdown = output_dir / "pair_duplex_certification_report.md"
    if analyzer_markdown.is_file() and not legacy_report_path.exists():
        analyzer_markdown.replace(legacy_report_path)
    physical_gate = write_physical_gate_artifacts(output_dir)
    status = str(physical_gate.get("status", "INCOMPLETE"))
    result_record = {
        "exit_code": 0 if status == "PASS" else 1,
        "status": status,
        "physical_gate_path": str(output_dir / "physical_runtime_gate.json"),
        "report_path": str(analyzer_markdown),
        "legacy_analyzer_status": metrics.get("status", analyzer_status),
        "legacy_metrics_path": str(report) if report.is_file() else None,
        "legacy_report_path": str(legacy_report_path) if legacy_report_path.is_file() else None,
        "physical_gate": physical_gate,
        "asr_json": str(asr_artifact) if asr_artifact.is_file() else None,
        "argv": argv,
        "stdout": analyzer_stdout,
        "stderr": analyzer_stderr,
    }
    if analyzer_error:
        result_record["legacy_analyzer_error"] = analyzer_error
    elif not script.is_file():
        result_record["legacy_analyzer_reason"] = "tools/analyze_pair_duplex_certification.py missing"
    return result_record


def _physical_gate_markdown(result: Mapping[str, Any]) -> str:
    from tools.verify_pair_physical_runtime import _aggregate_gate_status

    status = str(result.get("status", "INCOMPLETE"))
    lines = [
        "# Pair duplex certification report",
        "",
        f"**Status: {status}**",
        "",
        "This verdict is rendered from physical_runtime_gate.json by "
        "tools/verify_pair_physical_runtime.py. The legacy analyzer output is "
        "preserved separately as legacy_pair_duplex_certification_report.md.",
        "",
        "Per-peer input WAVs are recordings of frames actually sent, not a replay "
        "of only the kickoff WAV; they may also contain live/endpoint silence and "
        "routed peer PCM. Their total duration does not identify a different seed.",
        "",
        "Latency semantics: `append_to_ack` measures receipt of the listener-ingress "
        "`input_audio_buffer.consumed` ACK after frame consumption; it is not a "
        "policy-completion ACK. Policy latency is gated separately by "
        "`append_to_native_policy_decision` and `append_to_accepted_native_policy_decision`, "
        "with the same 80 ms p95 threshold. ACK receipt alone cannot satisfy either policy gate.",
        "",
        "| Physical gate | Status |",
        "|---|---|",
    ]
    gates = result.get("gates")
    if isinstance(gates, Mapping):
        for name, value in gates.items():
            gate_status = value.get("status") if isinstance(value, Mapping) else None
            if not isinstance(gate_status, str):
                nested_sources = result.get("nested_status_sources")
                prefix = f"{name}."
                child_statuses = [
                    child_status
                    for path, child_status in nested_sources.items()
                    if isinstance(path, str) and path.startswith(prefix)
                ] if isinstance(nested_sources, Mapping) else []
                gate_status = (
                    _aggregate_gate_status(child_statuses)
                    if child_statuses else "INCOMPLETE"
                )
            lines.append(f"| {name} | **{gate_status}** |")
    else:
        lines.append("| physical_runtime_gate | **INCOMPLETE** |")
    lines.extend(["", "## Gate details", ""])
    if isinstance(gates, Mapping):
        for name, value in gates.items():
            lines.extend([f"### {name}", ""])
            if not isinstance(value, Mapping):
                lines.append(f"- Result: {value!r}")
                lines.append("")
                continue
            failures = value.get("failures")
            missing = value.get("missing")
            if isinstance(failures, list) and failures:
                lines.extend(f"- Failure: {item}" for item in failures)
            if isinstance(missing, list) and missing:
                lines.extend(f"- Missing: {item}" for item in missing)
            if not failures and not missing:
                lines.append(f"- Status: {value.get('status', 'INCOMPLETE')}")
            lines.extend([
                "",
                "~~~json",
                json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True),
                "~~~",
            ])
            lines.append("")
    error = result.get("error")
    if error:
        lines.extend(["## Verification error", "", str(error), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_physical_gate_artifacts(output_dir: Path) -> dict[str, Any]:
    """Write the authoritative gate JSON and canonical Markdown report."""
    from tools.verify_pair_physical_runtime import verify_capture

    required = ("report.json", "events.jsonl", "server_run_manifest.json")
    missing_files = [name for name in required if not (output_dir / name).is_file()]
    if missing_files:
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "INCOMPLETE",
            "gates": {},
            "error": "required capture files missing: " + ", ".join(missing_files),
        }
    else:
        try:
            result = verify_capture(output_dir.resolve())
        except Exception as exc:
            result = {
                "schema_version": 1,
                "status": "INCOMPLETE",
                "gates": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
    gate_path = output_dir / "physical_runtime_gate.json"
    gate_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    legacy_report = output_dir / "legacy_pair_duplex_certification_report.md"
    canonical_report = output_dir / "pair_duplex_certification_report.md"
    if canonical_report.is_file() and not legacy_report.exists():
        canonical_report.replace(legacy_report)
    canonical_report.write_text(_physical_gate_markdown(result), encoding="utf-8")
    return result


def _write_manifest(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default_output(root: Path) -> Path:
    return root / "artifacts" / f"pair_duplex_e2e_proof_{_timestamp()}_{uuid.uuid4().hex[:8]}"


def run_proof(args: argparse.Namespace) -> int:
    repository_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output(repository_root)
    if not output_dir.is_absolute():
        output_dir = repository_root / output_dir
    output_dir = output_dir.resolve()
    output_rel = _relative_output(repository_root, output_dir)
    manifest_path = output_dir / "server_run_manifest.json"
    container_name = f"pair-duplex-proof-{_timestamp().lower()}-{uuid.uuid4().hex[:8]}"
    native_listener_attestation_dir = (
        f"/workspace/{output_rel}/listener_attestations"
        if bool(getattr(args, "native_listener_capacity_candidate", False)) else None
    )
    first_input_scheduler_anchor = bool(
        getattr(args, "first_input_scheduler_anchor", False)
    )
    ready_frame_scheduler_dispatch = bool(
        getattr(args, "ready_frame_scheduler_dispatch", False)
    )
    listener_pair_lock_handoff = bool(
        getattr(args, "listener_pair_lock_handoff", False)
    )
    native_control6_fused_readback = bool(
        getattr(args, "native_control6_fused_readback", False)
    )
    native_listener_cuda_profile = bool(
        getattr(args, "native_listener_cuda_profile", False)
    )
    native_listener_cuda_profile_directory = (
        f"/workspace/{output_rel}/native_listener_cuda_profile"
        if native_listener_cuda_profile else None
    )
    validate_scheduler_mode(
        first_input_scheduler_anchor=first_input_scheduler_anchor,
        ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
    )
    bundle_dir = Path(args.bundle_dir).expanduser()
    if not bundle_dir.is_absolute():
        bundle_dir = repository_root / bundle_dir

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "pair_duplex_proof_3090",
        "status": "preflight",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repository_root),
        "output_dir": str(output_dir),
        "output_dir_in_container": f"/workspace/{output_rel}",
        "runner_argv": list(getattr(args, "runner_argv", [])),
        "container": {"name": container_name, "id": None, "image": args.image},
        "maintenance": None,
        # The analyzer consumes these top-level fields. Null version/hash values
        # remain explicitly incomplete when preflight fails before runtime loading.
        "git_commit": None,
        "python_version": None,
        "packages": {"torch": None, "transformers": None},
        "server_port": 8092,
        "turn_policy": "gemma_native_control6",
        "action_steering_enabled": False,
        "checkpoints": {
            name: {"path": path, "sha256": None}
            for name, path in ANALYZER_CHECKPOINTS.items()
        },
        "argv": {},
        "code2wav_graph_family": {
            "requested": bool(getattr(args, "code2wav_graph_family", False)),
            "status": "pending" if getattr(args, "code2wav_graph_family", False) else "not_requested",
        },
        "code2wav_graph_odd_tail": {
            "requested": bool(getattr(args, "code2wav_graph_odd_tail", False)),
            "status": "pending" if getattr(args, "code2wav_graph_odd_tail", False) else "not_requested",
        },
        "native_listener_capacity_candidate": {
            "requested": bool(getattr(args, "native_listener_capacity_candidate", False)),
            "status": "pending" if getattr(args, "native_listener_capacity_candidate", False) else "not_requested",
            "expected_library_path": (
                NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH
                if getattr(args, "native_listener_capacity_candidate", False) else None
            ),
            "expected_library_sha256": (
                EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256
                if getattr(args, "native_listener_capacity_candidate", False) else None
            ),
            "expected_source_manifest_sha256": (
                EXPECTED_NATIVE_LISTENER_CANDIDATE_SOURCE_MANIFEST_SHA256
                if getattr(args, "native_listener_capacity_candidate", False) else None
            ),
            "session_attestation_directory": (
                f"/workspace/{output_rel}/listener_attestations"
                if getattr(args, "native_listener_capacity_candidate", False) else None
            ),
        },
        "first_input_scheduler_anchor": {
            "requested": first_input_scheduler_anchor,
            "status": "pending" if first_input_scheduler_anchor else "not_requested",
        },
        "ready_frame_scheduler_dispatch": {
            "requested": ready_frame_scheduler_dispatch,
            "status": "pending" if ready_frame_scheduler_dispatch else "not_requested",
        },
        "listener_pair_lock_handoff": {
            "requested": listener_pair_lock_handoff,
            "status": "pending" if listener_pair_lock_handoff else "not_requested",
            "environment_variable": LISTENER_PAIR_LOCK_HANDOFF_ENV,
            "cpu_preflight_setting": None,
        },
        "native_listener_cuda_profile": {
            "requested": native_listener_cuda_profile,
            "status": "pending" if native_listener_cuda_profile else "not_requested",
            "environment_variable": NATIVE_LISTENER_CUDA_PROFILE_ENV,
            "cpu_preflight_setting": None,
            "directory_in_container": native_listener_cuda_profile_directory,
            "sampling_frequency_frames": 8,
            "sample_cap_per_context": 64,
        },
        "native_control6_fused_readback": {
            "requested": native_control6_fused_readback,
            "status": "pending" if native_control6_fused_readback else "not_requested",
            "environment_variable": NATIVE_CONTROL6_FUSED_READBACK_ENV,
            "cpu_preflight_setting": None,
        },
        "logs": {
            "server": str(output_dir / "server.log"),
            "client": str(output_dir / "client.log"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_manifest(manifest_path, manifest)

    own_container_id: str | None = None
    launch_attempted = False
    log_capture = None
    maintenance_snapshot: dict[str, Any] | None = None
    primary_error: str | None = None
    client_exit_code: int | None = None
    success = False
    cleanup_errors: list[str] = []
    try:
        metadata = collect_run_metadata(repository_root, bundle_dir, output_relative=output_rel)
        manifest.update(metadata)
        git_record = metadata.get("git")
        manifest["git_commit"] = git_record.get("sha") if isinstance(git_record, Mapping) else None
        manifest["checkpoints"] = analyzer_checkpoint_attestations(metadata)
        image_identity = docker_image_identity(args.image)
        manifest["container"]["image_identity"] = image_identity
        bundle_container_path = "/workspace/" + bundle_dir.resolve().relative_to(repository_root).as_posix()
        server_argv = build_server_argv(bundle_path=bundle_container_path)
        client_argv = build_client_argv(
            output_dir=output_rel,
            max_turns=args.max_turns,
            max_duration_s=args.max_duration_s,
        )
        docker_argv = build_docker_run_argv(
            repository_root=repository_root,
            container_name=container_name,
            image=args.image,
            server_argv=server_argv,
            code2wav_graph_family=bool(getattr(args, "code2wav_graph_family", False)),
            code2wav_graph_odd_tail=bool(getattr(args, "code2wav_graph_odd_tail", False)),
            native_listener_capacity_candidate=bool(getattr(args, "native_listener_capacity_candidate", False)),
            native_listener_attestation_dir=native_listener_attestation_dir,
            native_listener_cuda_profile=native_listener_cuda_profile,
            native_listener_cuda_profile_directory=native_listener_cuda_profile_directory,
            first_input_scheduler_anchor=first_input_scheduler_anchor,
            ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
            listener_pair_lock_handoff=listener_pair_lock_handoff,
            native_control6_fused_readback=native_control6_fused_readback,
        )
        manifest["argv"] = {
            "server": server_argv,
            "client": client_argv,
            "docker_run": docker_argv,
        }
        runtime_preflight = probe_container_runtime(
            repository_root,
            args.image,
            code2wav_graph_family=bool(getattr(args, "code2wav_graph_family", False)),
            code2wav_graph_odd_tail=bool(getattr(args, "code2wav_graph_odd_tail", False)),
            native_listener_capacity_candidate=bool(getattr(args, "native_listener_capacity_candidate", False)),
            native_listener_cuda_profile=native_listener_cuda_profile,
            native_listener_cuda_profile_directory=native_listener_cuda_profile_directory,
            first_input_scheduler_anchor=first_input_scheduler_anchor,
            ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
            listener_pair_lock_handoff=listener_pair_lock_handoff,
            native_control6_fused_readback=native_control6_fused_readback,
        )
        manifest["runtime_preflight"] = runtime_preflight
        manifest["first_input_scheduler_anchor"].update({
            "cpu_preflight_setting": runtime_preflight["first_input_scheduler_anchor_setting"],
            "cpu_preflight_enabled": runtime_preflight["first_input_scheduler_anchor_enabled"],
            "cpu_preflight_status": "PASS",
        })
        manifest["ready_frame_scheduler_dispatch"].update({
            "cpu_preflight_setting": runtime_preflight["ready_frame_scheduler_dispatch_setting"],
            "cpu_preflight_enabled": runtime_preflight["ready_frame_scheduler_dispatch_enabled"],
            "cpu_preflight_status": "PASS",
        })
        manifest["listener_pair_lock_handoff"].update({
            "cpu_preflight_setting": runtime_preflight["listener_pair_lock_handoff_setting"],
            "cpu_preflight_enabled": runtime_preflight["listener_pair_lock_handoff_enabled"],
            "cpu_preflight_status": "PASS",
        })
        manifest["native_listener_cuda_profile"].update({
            "cpu_preflight_setting": runtime_preflight["native_listener_cuda_profile_setting"],
            "cpu_preflight_enabled": runtime_preflight["native_listener_cuda_profile_enabled"],
            "cpu_preflight_directory": runtime_preflight[
                "native_listener_cuda_profile_directory_setting"
            ],
            "cpu_preflight_status": "PASS",
        })
        manifest["native_control6_fused_readback"].update({
            "cpu_preflight_setting": runtime_preflight[
                "native_control6_fused_readback_setting"
            ],
            "cpu_preflight_enabled": runtime_preflight[
                "native_control6_fused_readback_enabled"
            ],
            "cpu_preflight_status": "PASS",
        })
        manifest["python_version"] = runtime_preflight["python"]
        manifest["packages"] = dict(runtime_preflight["packages"])
        manifest["entrypoint_smoke"] = probe_cli_entrypoints(repository_root, args.image)
        if bool(getattr(args, "native_listener_capacity_candidate", False)):
            manifest["native_listener_capacity_candidate"]["status"] = "CPU_preflight_PASS"
            manifest["native_listener_capacity_candidate"]["cpu_preflight"] = runtime_preflight[
                "native_listener_candidate"
            ]
        if args.maintenance_container:
            maintenance_snapshot = snapshot_maintenance_container(args.maintenance_container, repository_root)
            manifest["maintenance"] = {
                "container_name": args.maintenance_container,
                "container_id": maintenance_snapshot.get("id"),
                "was_running": maintenance_snapshot.get("was_running"),
                "snapshot_path": str(output_dir / "maintenance_container_before.json"),
            }
            snapshot_path = output_dir / "maintenance_container_before.json"
            snapshot_path.write_text(
                json.dumps(maintenance_snapshot.get("inspect", {}), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_manifest(manifest_path, manifest)
            if maintenance_snapshot.get("was_running") is not True:
                raise RuntimeError(
                    "managed maintenance container was not running; refusing a second GPU model load"
                )
            try:
                before_memory = pause_maintenance_container(maintenance_snapshot, repository_root)
                manifest["maintenance"]["gpu_memory_before_stop"] = before_memory
                manifest["maintenance"]["gpu_process_pids_before_stop"] = list(
                    maintenance_snapshot.get("gpu_process_pids_before_stop", [])
                )
                _write_manifest(manifest_path, manifest)
                release_record = wait_for_gpu_vram_release(maintenance_snapshot, repository_root)
            except GpuVramBarrierError as exc:
                manifest["maintenance"]["gpu_release_barrier"] = {
                    "status": "blocked",
                    "observations": exc.observations,
                }
                _write_manifest(manifest_path, manifest)
                raise
            except Exception:
                if maintenance_snapshot.get("gpu_memory_before_stop") is not None:
                    manifest["maintenance"]["gpu_memory_before_stop"] = maintenance_snapshot[
                        "gpu_memory_before_stop"
                    ]
                    manifest["maintenance"]["gpu_process_pids_before_stop"] = list(
                        maintenance_snapshot.get("gpu_process_pids_before_stop", [])
                    )
                    _write_manifest(manifest_path, manifest)
                raise
            manifest["maintenance"]["gpu_release_barrier"] = release_record
            _write_manifest(manifest_path, manifest)
        else:
            prelaunch_memory = nvidia_smi_memory_snapshot(repository_root)
            manifest["gpu_prelaunch_without_maintenance"] = {
                "memory": prelaunch_memory,
                "status": "ready",
            }
            try:
                assert_gpu_ready_without_maintenance(prelaunch_memory)
            except Exception as exc:
                manifest["gpu_prelaunch_without_maintenance"]["status"] = "blocked"
                manifest["gpu_prelaunch_without_maintenance"]["error"] = f"{type(exc).__name__}: {exc}"
                _write_manifest(manifest_path, manifest)
                raise
            _write_manifest(manifest_path, manifest)
        if native_control6_fused_readback:
            if maintenance_snapshot is None or maintenance_snapshot.get("was_running") is not True:
                manifest["native_control6_fused_readback"]["cuda_smoke"] = {
                    "status": "blocked",
                    "error": "fused-readback GPU smoke requires a paused, previously running managed container",
                }
                _write_manifest(manifest_path, manifest)
                raise RuntimeError(
                    "native Control6 fused-readback opt-in requires managed GPU maintenance"
                )
            smoke_output = output_dir / "native_control6_fused_readback_cuda_smoke.json"
            try:
                smoke_record = run_native_control6_fused_readback_cuda_smoke(
                    repository_root, args.image, smoke_output,
                )
                manifest["native_control6_fused_readback"]["cuda_smoke"] = smoke_record
                _write_manifest(manifest_path, manifest)
                post_smoke_barrier = wait_for_gpu_vram_release(
                    maintenance_snapshot, repository_root,
                )
                manifest["maintenance"]["post_smoke_gpu_release_barrier"] = post_smoke_barrier
                manifest["native_control6_fused_readback"]["cuda_smoke"][
                    "post_smoke_gpu_release_barrier_status"
                ] = "PASS"
                manifest["native_control6_fused_readback"]["status"] = "cuda_smoke_PASS"
                _write_manifest(manifest_path, manifest)
            except Exception as exc:
                manifest["native_control6_fused_readback"].setdefault("cuda_smoke", {
                    "status": "FAIL",
                })
                manifest["native_control6_fused_readback"]["cuda_smoke"].update({
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                _write_manifest(manifest_path, manifest)
                raise
        manifest["status"] = "starting_server"
        _write_manifest(manifest_path, manifest)
        launch_attempted = True
        own_container_id = launch_server_container(
            repository_root,
            container_name,
            args.image,
            server_argv,
            code2wav_graph_family=bool(getattr(args, "code2wav_graph_family", False)),
            code2wav_graph_odd_tail=bool(getattr(args, "code2wav_graph_odd_tail", False)),
            native_listener_capacity_candidate=bool(getattr(args, "native_listener_capacity_candidate", False)),
            native_listener_attestation_dir=native_listener_attestation_dir,
            native_listener_cuda_profile=native_listener_cuda_profile,
            native_listener_cuda_profile_directory=native_listener_cuda_profile_directory,
            first_input_scheduler_anchor=first_input_scheduler_anchor,
            ready_frame_scheduler_dispatch=ready_frame_scheduler_dispatch,
            listener_pair_lock_handoff=listener_pair_lock_handoff,
            native_control6_fused_readback=native_control6_fused_readback,
        )
        manifest["container"]["id"] = own_container_id
        _write_manifest(manifest_path, manifest)
        log_capture = start_server_log_capture(
            own_container_id, repository_root, output_dir / "server.log",
        )
        health = wait_for_server_ready(
            repository_root,
            own_container_id,
            output_dir / "server.log",
            timeout_s=args.server_startup_timeout_s,
            poll_interval_s=args.readiness_poll_s,
        )
        manifest["server_readiness"] = {"load_marker": LOAD_MARKER, "health": health}
        manifest["first_input_scheduler_anchor"].update(
            first_input_scheduler_anchor_record(
                health, requested=first_input_scheduler_anchor,
            )
        )
        manifest["ready_frame_scheduler_dispatch"].update(
            ready_frame_scheduler_dispatch_record(
                health, requested=ready_frame_scheduler_dispatch,
            )
        )
        manifest["listener_pair_lock_handoff"].update(
            listener_pair_lock_handoff_record(
                health, requested=listener_pair_lock_handoff,
            )
        )
        manifest["native_listener_cuda_profile"].update(
            native_listener_cuda_profile_record(
                health,
                requested=native_listener_cuda_profile,
                directory=native_listener_cuda_profile_directory,
            )
        )
        manifest["native_control6_fused_readback"].update(
            native_control6_fused_readback_record(
                health, requested=native_control6_fused_readback,
            )
        )
        manifest["effective_route"] = effective_route_record(health, server_argv)
        manifest["code2wav_graph_family"] = code2wav_graph_family_record(
            health,
            requested=bool(getattr(args, "code2wav_graph_family", False)),
            odd_tail_requested=bool(getattr(args, "code2wav_graph_odd_tail", False)),
        )
        manifest["code2wav_graph_odd_tail"] = {
            "requested": bool(getattr(args, "code2wav_graph_odd_tail", False)),
            "status": manifest["code2wav_graph_family"].get("status", "not_requested"),
            "window_frames": manifest["code2wav_graph_family"].get("window_frames", []),
        }
        manifest["native_talker_binary"] = attest_native_talker_binary(own_container_id)
        runtime = record_runtime_versions(
            own_container_id,
            native_listener_capacity_candidate=bool(getattr(args, "native_listener_capacity_candidate", False)),
        )
        manifest["runtime"] = runtime
        if bool(getattr(args, "native_listener_capacity_candidate", False)):
            manifest["native_listener_capacity_candidate"]["server_container_preclient_attestation"] = (
                runtime.get("native_listener_candidate")
            )
            manifest["native_listener_capacity_candidate"]["status"] = "server_container_preclient_PASS"
        manifest["python_version"] = runtime.get("python")
        runtime_packages = runtime.get("packages")
        if isinstance(runtime_packages, Mapping):
            manifest["packages"] = dict(runtime_packages)
        manifest["status"] = "running_client"
        _write_manifest(manifest_path, manifest)

        client_result = run_client(
            repository_root, own_container_id, client_argv, output_dir / "client.log",
        )
        client_exit_code = client_result.returncode
        manifest["client_exit_code"] = client_exit_code
        if bool(getattr(args, "native_listener_capacity_candidate", False)):
            stop_server_log_capture(log_capture)
            log_capture = None
            listener_attestation = native_listener_session_postrun_attestation(
                output_dir / "report.json",
                output_dir / "server.log",
                attestation_dir=output_dir / "listener_attestations",
                expected_library_path=NATIVE_LISTENER_CANDIDATE_LIBRARY_CONTAINER_PATH,
                expected_library_sha256=EXPECTED_NATIVE_LISTENER_CANDIDATE_SHA256,
            )
            manifest["native_listener_capacity_candidate"]["session_contexts_postrun"] = listener_attestation
            manifest["native_listener_capacity_candidate"]["status"] = listener_attestation.get("status")
        manifest["native_talker_postrun"] = native_talker_postrun_attestation(
            output_dir / "report.json"
        )
        manifest["archive_asr_deferral_postrun"] = archive_defer_postrun_attestation(
            output_dir / "report.json"
        )
        runtime_gates_pass = (
            manifest["native_talker_postrun"].get("status") == "PASS"
            and manifest["archive_asr_deferral_postrun"].get("status") == "PASS"
            and (
                not bool(getattr(args, "native_listener_capacity_candidate", False))
                or manifest.get("native_listener_capacity_candidate", {}).get("status") == "PASS"
            )
        )
        manifest["status"] = (
            "captured"
            if client_exit_code == 0 and runtime_gates_pass
            else "runtime_gate_failed"
            if client_exit_code == 0
            else "client_failed"
        )
    except Exception as exc:  # The failure receipt must survive cleanup errors.
        primary_error = f"{type(exc).__name__}: {exc}"
        manifest["status"] = "failed"
        manifest["error"] = primary_error
    finally:
        if launch_attempted:
            try:
                cleanup_owned_container(own_container_id, container_name, repository_root)
                manifest["container"]["cleanup"] = "removed"
            except Exception as exc:
                cleanup_errors.append(f"container cleanup: {type(exc).__name__}: {exc}")
                manifest["container"]["cleanup"] = "failed"
        try:
            stop_server_log_capture(log_capture)
        except Exception as exc:
            cleanup_errors.append(f"server log capture: {type(exc).__name__}: {exc}")
        if maintenance_snapshot and maintenance_snapshot.get("was_running"):
            try:
                restore_maintenance_container(
                    maintenance_snapshot, timeout_s=args.maintenance_restore_timeout_s,
                )
                manifest["maintenance"]["restored"] = True
            except Exception as exc:
                cleanup_errors.append(f"maintenance restoration: {type(exc).__name__}: {exc}")
                manifest["maintenance"]["restored"] = False
                manifest["maintenance"]["restore_error"] = cleanup_errors[-1]
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if cleanup_errors:
            manifest["cleanup_errors"] = cleanup_errors
            success = False
            if manifest.get("status") in {"complete", "captured", "client_failed", "analysis_failed"}:
                manifest["status_before_cleanup_failure"] = manifest["status"]
                manifest["status"] = "cleanup_failed"
        if primary_error:
            manifest["error"] = primary_error
        _write_manifest(manifest_path, manifest)

        # Analyze on the host after the task container is removed and any managed
        # service is restored. This also emits an INCOMPLETE report for preflight,
        # readiness, or client failures while preserving the primary failure.
        if args.skip_analyzer:
            manifest["analysis"] = {"status": "skipped", "reason": "--skip-analyzer"}
            try:
                manifest["physical_gate"] = write_physical_gate_artifacts(output_dir)
            except Exception as exc:
                manifest["physical_gate"] = {
                    "schema_version": 1,
                    "status": "INCOMPLETE",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            try:
                manifest["analysis"] = run_analyzer(repository_root, output_dir)
            except Exception as exc:
                manifest["analysis"] = {
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if not primary_error and not cleanup_errors:
            if client_exit_code is None:
                manifest["status"] = "failed"
            elif client_exit_code != 0:
                manifest["status"] = "client_failed"
            elif (
                manifest.get("native_talker_postrun", {}).get("status") != "PASS"
                or manifest.get("archive_asr_deferral_postrun", {}).get("status") != "PASS"
                or (
                    bool(getattr(args, "native_listener_capacity_candidate", False))
                    and manifest.get("native_listener_capacity_candidate", {}).get("status") != "PASS"
                )
            ):
                manifest["status"] = "runtime_gate_failed"
            elif args.skip_analyzer:
                # Captured audio is retained, but an explicit analyzer skip cannot
                # certify this run.
                manifest["status"] = "captured"
            elif (
                manifest.get("analysis", {}).get("status") != "PASS"
                or manifest.get("analysis", {}).get("exit_code") not in {None, 0}
            ):
                manifest["status"] = "analysis_failed"
            else:
                manifest["status"] = "complete"
        success = manifest.get("status") == "complete"
        _write_manifest(manifest_path, manifest)
    return 0 if success and not cleanup_errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_code2wav_graph_options(
            code2wav_graph_family=args.code2wav_graph_family,
            code2wav_graph_odd_tail=args.code2wav_graph_odd_tail,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_scheduler_mode(
            first_input_scheduler_anchor=args.first_input_scheduler_anchor,
            ready_frame_scheduler_dispatch=args.ready_frame_scheduler_dispatch,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_turns < 1 or args.max_duration_s <= 0:
        parser.error("--max-turns must be positive and --max-duration-s must be greater than zero")
    if args.server_startup_timeout_s <= 0 or args.readiness_poll_s <= 0 or args.maintenance_restore_timeout_s <= 0:
        parser.error("startup, polling, and restore timeout values must be positive")
    if args.maintenance_container and args.maintenance_container != DEFAULT_MAINTENANCE_CONTAINER:
        parser.error(f"--maintenance-container must be {DEFAULT_MAINTENANCE_CONTAINER!r}")
    args.runner_argv = list(sys.argv[1:] if argv is None else argv)
    return run_proof(args)


if __name__ == "__main__":
    raise SystemExit(main())
