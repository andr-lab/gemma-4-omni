from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native" / "gemma4_listener"


def _cpp_contract_command(tmp_path: Path) -> list[str] | tuple[str, str]:
    source = NATIVE / "sliding_repeat_capacity_test.cpp"
    executable = tmp_path / ("sliding_repeat_capacity_test.exe" if os.name == "nt" else "sliding_repeat_capacity_test")
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if compiler:
        return [
            compiler,
            "-std=c++17",
            "-I",
            str(NATIVE),
            str(source),
            "-o",
            str(executable),
        ], str(executable)

    if os.name == "nt":
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            install = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            devcmd = Path(install) / "Common7" / "Tools" / "VsDevCmd.bat"
            if devcmd.exists():
                command = (
                    f'call "{devcmd}" -arch=x64 -host_arch=x64 >nul && '
                    f'cl /nologo /std:c++17 /EHsc /I "{NATIVE}" "{source}" '
                    f'/Fe:"{executable}" && "{executable}"'
                )
                return "cmd.exe", command

    pytest.skip("A host C++17 compiler is required for the native helper contract test")


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function body for {signature}")


def test_sliding_capacity_and_guard_helpers_compile_and_hold_contract() -> None:
    scratch = ROOT / ".tmp" / f"native_sliding_contract_{uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    command = _cpp_contract_command(scratch)
    if isinstance(command, tuple):
        runner, shell_command = command
        compiled = subprocess.run(
            f'{runner} /d /s /c {shell_command}',
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    else:
        compile_command, executable = command
        compiled = subprocess.run(
            compile_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        executed = subprocess.run(
            [executable], cwd=ROOT, capture_output=True, text=True, check=False
        )
        assert executed.returncode == 0, executed.stdout + executed.stderr


def test_all_native_staging_writes_are_guarded_and_full_slots_stay_full_sized() -> None:
    source = (NATIVE / "gemma4_listener.cu").read_text(encoding="utf-8")
    cache = (NATIVE / "gemma4_listener_cache.cu").read_text(encoding="utf-8")
    assert '#include "sliding_repeat_capacity.h"' in source
    assert "gemma4_cache_valid_kv_len(const gemma4_cache *c)" in cache

    execute = _function_body(source, "static int execute_layer(")
    position_guard = execute.index("validate_position_query(")
    first_device_work = execute.index("debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps")
    assert position_guard < first_device_work

    generic_guard = execute.index("validate_visible_window(")
    generic_stage = execute.index("ctx->d_sdpa_k_repeat,", generic_guard)
    assert generic_guard < generic_stage
    assert execute.index("validate_head_dim(") < generic_stage

    shared = _function_body(source, "static int stage_shared_sliding_source13(")
    shared_window_guard = shared.index("validate_visible_window(")
    first_repeat = shared.index("kernel_repeat_kv_bf16<<<")
    assert shared_window_guard < first_repeat
    assert shared.index("validate_position_query(") < first_repeat

    allocation = source[source.index("const size_t repeat_elements =") :]
    assert "GEMMA4_NUM_ATTN_HEADS * config->max_seq_len * 512u" in allocation
    assert "sliding_staging.capacity" in allocation
    assert "sliding_staging.max_sliding_head_dim" in allocation
    assert "GEMMA4_CREATE_ALLOC(ctx->d_sdpa_full_k_repeat[slot]" in allocation
    assert "GEMMA4_CREATE_ALLOC(ctx->d_sdpa_full_v_repeat[slot]" in allocation


def test_context_create_diagnostics_are_failure_only_and_label_unknown_memory() -> None:
    source = (NATIVE / "gemma4_listener.cu").read_text(encoding="utf-8")
    cache = (NATIVE / "gemma4_listener_cache.cu").read_text(encoding="utf-8")
    formatter = (NATIVE / "listener_create_failure_log.h").read_text(encoding="utf-8")
    adapter_source = (NATIVE / "listener_cuda_create_failure_log.cuh").read_text(encoding="utf-8")
    makefile = (NATIVE / "Makefile").read_text(encoding="utf-8")
    logger = _function_body(adapter_source, "static inline void gemma4_log_create_failure(")
    assert "cudaMemGetInfo(&free_bytes, &total_bytes)" in logger
    assert "write_context_create_failure(" in logger
    assert "free_bytes=unknown total_bytes=unknown" in formatter
    assert '#include "listener_cuda_create_failure_log.cuh"' in source
    assert '#include "listener_cuda_create_failure_log.cuh"' in cache
    assert "static inline void gemma4_log_create_failure(" in adapter_source
    assert "listener_cuda_create_failure_log.cuh listener_create_failure_log.h sliding_repeat_capacity.h" in makefile
    assert '"cudaMalloc"' in source
    assert "ordered_failure.requested_bytes" in source
    assert "d_inv_freq_sliding.upload" in source
    assert "d_inv_freq_full.upload" in source
    assert "int gemma4_cache_create(gemma4_cache **out_cache, uint32_t max_seq_len, bool allocate_device_position)" in cache
    for field in (
        "cache.device_position",
        "cache.layers[%d].k_data",
        "cache.layers[%d].v_data",
        "cache.layers[%d].rebase_k_scratch",
        "cache.layers[%d].rebase_v_scratch",
    ):
        assert field in cache
    assert "gemma4_log_create_failure(" in cache
