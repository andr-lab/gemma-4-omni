from __future__ import annotations

from pathlib import Path

import pytest

from duplex_kernel.native_listener import (
    GEMMA4_ATTN_ATEN_REPEAT_EXACT,
    GEMMA4_ATTN_NATIVE_SPLITK_GQA,
    assert_production_native_runtime,
    resolve_native_attention_backend,
)


def test_production_resolver_defaults_to_ordered_backend(monkeypatch) -> None:
    monkeypatch.delenv("GEMMA4_ATTENTION_BACKEND", raising=False)
    assert resolve_native_attention_backend() == GEMMA4_ATTN_NATIVE_SPLITK_GQA


def test_ordered_full_attention_query_rows_follow_native_q2_geometry() -> None:
    root = Path(__file__).resolve().parents[2]
    ordered_header = (root / "native" / "gemma4_listener_microbatch" /
                      "gemma4_listener_attention_ordered.cuh").read_text(encoding="utf-8")
    ordered_source = (root / "native" / "gemma4_listener_microbatch" /
                     "gemma4_listener_attention_ordered.cu").read_text(encoding="utf-8")
    batch_source = (root / "native" / "gemma4_listener_microbatch" /
                    "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
    assert '#include "generated_topology.h"' in ordered_header
    assert "#define GEMMA4_ORDERED_QUERY_ROWS GEMMA4_STREAM_Q_LEN" in ordered_header
    assert "static_assert(GEMMA4_STREAM_Q_LEN == 2)" in ordered_source
    assert "GEMMA4_ORDERED_QUERY_ROWS 4" not in ordered_header
    assert "visible > GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE" in batch_source


def test_production_resolver_preserves_explicit_oracle(monkeypatch) -> None:
    monkeypatch.setenv("GEMMA4_ATTENTION_BACKEND", "garbage")
    assert resolve_native_attention_backend("ATEN_REPEAT_EXACT") == GEMMA4_ATTN_ATEN_REPEAT_EXACT
    assert resolve_native_attention_backend(1) == GEMMA4_ATTN_ATEN_REPEAT_EXACT


def test_production_resolver_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("GEMMA4_ATTENTION_BACKEND", "garbage")
    with pytest.raises(RuntimeError):
        resolve_native_attention_backend()


def test_layer_local_parity_forces_the_aten_exact_oracle_environment() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "kernel_gates"
        / "test_native_layer_local_parity.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ["GEMMA4_ATEN_SDPA_BRIDGE"] = "1"' in source
    assert 'os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"' in source
    assert 'os.environ.pop("GEMMA4_ATTENTION_BACKEND", None)' in source
    assert 'os.environ.pop("GEMMA4_ATEN_RMSNORM_BRIDGE", None)' in source


def test_production_step_does_not_capture_debug_layer_trace() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "native"
        / "gemma4_listener"
        / "gemma4_listener.cu"
    ).read_text(encoding="utf-8")
    assert "GEMMA4_NUM_LAYERS, true, false, stream" in source
    assert "captures one hidden output after each executed layer without adding any" in source


def test_masked_atten_fallback_declares_persistent_explicit_repeat_staging() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "native"
        / "gemma4_listener"
        / "gemma4_listener.cu"
    ).read_text(encoding="utf-8")
    kernels = (
        Path(__file__).resolve().parents[2]
        / "native"
        / "gemma4_listener"
        / "gemma4_listener_kernels.cuh"
    ).read_text(encoding="utf-8")
    assert "d_sdpa_k_repeat" in source
    assert "d_sdpa_v_repeat" in source
    assert "kernel_repeat_kv_bf16" in kernels
    assert "GEMMA4_NUM_ATTN_HEADS" in source
    assert "const uint64_t kv_heads = use_gqa" in source
    assert "ctx->d_sdpa_k_repeat" in source and "ctx->d_sdpa_v_repeat" in source


def test_masked_atten_fallback_reuses_shared_sliding_source13_staging() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "native"
        / "gemma4_listener"
        / "gemma4_listener.cu"
    ).read_text(encoding="utf-8")
    assert "d_sdpa_shared_sliding_k_repeat" in source
    assert "d_sdpa_shared_sliding_v_repeat" in source
    assert "kv_shared_src_index == 13" in source
    assert "!spec->is_kv_shared && layer_idx == 13" in source
    assert "debug_uses_shared_sliding_source13" in source


def test_production_runtime_assertion_checks_functional_contract_without_hashes() -> None:
    info = {
        "enabled": True,
        "cuda_graph_enabled": False,
        "checkpoint_path": "/attested/checkpoint",
        "attention_backend_id": 3,
        "attention_backend": "NATIVE_SPLITK_GQA",
        "attention_implementation": "CUTLASS_ORDERED_TILES_V1",
        "splitk_layer_mask_layers": [19, 24, 29, 34],
        "rmsnorm_backend": "NATIVE_EXACT",
    }
    assert_production_native_runtime(info)
    with pytest.raises(RuntimeError):
        assert_production_native_runtime({**info, "attention_backend_id": 1})
    with pytest.raises(RuntimeError):
        assert_production_native_runtime({**info, "checkpoint_path": ""})
