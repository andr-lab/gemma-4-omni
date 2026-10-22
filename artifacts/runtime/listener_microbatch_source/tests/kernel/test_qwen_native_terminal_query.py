from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_native_abi_declares_audio_only_terminal_query() -> None:
    header = _source("native/qwen_talker/qwen_talker.h")
    implementation = _source("native/qwen_talker/qwen_talker.cu")
    kernels = _source("native/qwen_talker/qwen_talker_kernels.cu")
    attention = _source("native/qwen_talker/qwen_talker_aten_attention.cpp")
    rmsnorm = _source("native/qwen_talker/qwen_talker_aten_rmsnorm.cpp")

    assert "qwen_talker_step_terminal_async" in header
    assert "qwen_talker_step_terminal_async" in implementation
    assert "return 2u;" in implementation
    assert "qwen_talker_runtime_terminal_prefix_cache_appends" in implementation
    assert "qwen_talker_runtime_terminal_audio_cache_appends" in implementation
    assert "qwen_build_terminal_mask" in kernels
    assert "visible_prefix_position" in kernels
    assert "query_len != 1u && query_len != 2u" in attention
    assert "qwen_talker_rmsnorm_graph_create" in implementation
    assert "&ctx->rmsnorm_graph_q1,\n            1u" in implementation
    assert "state->rows != 2u" not in rmsnorm
    assert "terminal_query_done" in implementation
    assert "audio_position != ctx->next_audio_position" in implementation


def test_native_terminal_is_audio_only_position_checked_and_not_a_joint_step() -> None:
    implementation = _source("native/qwen_talker/qwen_talker.cu")
    adapter = _source("duplex_kernel/native_qwen_talker.py")

    assert "def step_terminal_async(" in adapter
    assert "qwen_talker_step_terminal_async" in adapter
    assert "qwen_talker_abi_version() != 2" in adapter
    assert "prefix_visible_position" in adapter
    assert "segment_joint_steps == 0u" in implementation
    assert "terminal_query_done" in implementation
    assert "ctx->next_audio_position = audio_position + 1u" in implementation
    assert "++ctx->joint_steps" not in implementation.split(
        "extern \"C\" int qwen_talker_step_terminal_async", 1
    )[1].split("extern \"C\" void qwen_talker_destroy", 1)[0]


def test_future_prefix_terminal_uses_native_q1_without_consuming_an_acoustic_frame() -> None:
    source = _source("duplex_kernel/future_prefix.py")
    terminal = source.split("def _step_terminal_slot_kv", 1)[1].split(
        "def _required_max_cache_len", 1
    )[0]

    assert "self._native_qwen.step_terminal_async(" in terminal
    assert "audio_position=pos_int" in terminal
    assert "prefix_visible_position=self._prefix_start + visible_end" in terminal
    assert "self._current_frame_id +=" not in terminal
    assert "self._committed_codes.append" not in terminal
    assert "_terminal_queries_used += 1" in source.split("def step_terminal", 1)[1]
    assert "terminal query audio-only" not in source


def test_native_trace_contract_attests_terminal_q1_separately_from_joint_q2() -> None:
    source = _source("duplex_kernel/native_qwen_talker.py")
    trace_contract = _source("duplex_kernel/native_trace.py")

    assert '"terminal_query"' in source
    assert "query_length" in source
    assert "prefix_cache_appends" in trace_contract
    assert "audio_cache_appends" in trace_contract
    assert "qwen_talker_runtime_terminal_prefix_cache_appends" in source
    assert "qwen_talker_runtime_terminal_audio_cache_appends" in source
    assert "visible_prefix_position" in trace_contract


def test_native_terminal_q1_matches_pytorch_after_real_joint_trajectory() -> None:
    """Opt-in RTX smoke; requires the pinned Qwen checkpoint and native library."""
    try:
        import pytest
    except ImportError:  # permits invoking this tiny gate directly in minimal CUDA images
        pytest = None

    if os.environ.get("QWEN_NATIVE_TERMINAL_SMOKE") != "1":
        if pytest is not None:
            pytest.skip("set QWEN_NATIVE_TERMINAL_SMOKE=1 for the RTX 3090 smoke")
        raise RuntimeError("set QWEN_NATIVE_TERMINAL_SMOKE=1 for the RTX 3090 smoke")
    import torch
    if not torch.cuda.is_available():
        if pytest is not None:
            pytest.skip("native Qwen terminal parity requires CUDA")
        raise RuntimeError("native Qwen terminal parity requires CUDA")

    from duplex_kernel.future_prefix import FuturePrefixQwen
    from duplex_kernel.models.talker import load_talker
    from duplex_kernel.native_trace import NATIVE_TRACE_ABI_VERSION, validate_native_trace_contract

    workspace = Path(os.environ.get("QWEN_NATIVE_WORKSPACE", "/workspace"))
    talker = load_talker(str(workspace), device="cuda", load_code2wav=False)
    talker.talker.eval()
    common = {
        "lookahead_frames": 0,
        "max_prefix_frames": 4,
        "max_audio_frames": 4,
        "device": torch.device("cuda:0"),
        "dtype": torch.bfloat16,
    }
    reference = FuturePrefixQwen(talker, runtime_backend="pytorch", **common)
    native = FuturePrefixQwen(
        talker,
        runtime_backend="native",
        trace_enabled=True,
        trace_sequence_id=0,
        trace_expected_steps=2,
        **common,
    )

    def parity(name: str, expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
        torch.cuda.synchronize()
        left = expected.detach().float()
        right = actual.detach().float()
        assert left.shape == right.shape, f"{name}: shape mismatch {left.shape} != {right.shape}"
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), f"{name}: non-finite output"
        delta = (left - right).abs()
        cosine = torch.nn.functional.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
        metrics = {"mae": delta.mean().item(), "max_abs": delta.max().item(), "cosine": cosine}
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values()), f"{name}: invalid metrics {metrics}"
        assert metrics["mae"] <= 0.01, f"{name}: MAE gate failed: {metrics}"
        assert metrics["max_abs"] <= 0.05, f"{name}: max-abs gate failed: {metrics}"
        assert metrics["cosine"] >= 0.999, f"{name}: cosine gate failed: {metrics}"
        return metrics

    try:
        reference.begin("Frame-causal native Qwen terminal-query smoke.")
        native.begin("Frame-causal native Qwen terminal-query smoke.")
        generator = torch.Generator(device="cuda").manual_seed(7319)
        prefixes = torch.randn(
            (2, 1, 2048), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        frame_metrics: list[dict[str, float]] = []
        for frame_id in range(2):
            prefix = prefixes[frame_id]
            reference.push_v9_frame(frame_id, prefix)
            native.push_v9_frame(frame_id, prefix)
            expected = reference.step_frame()
            actual = native.step_frame()
            assert expected.frame_id == actual.frame_id == frame_id
            frame_metrics.append(parity(f"joint frame {frame_id}", expected.last_hidden, actual.last_hidden))
            codes = torch.arange(16, device="cuda", dtype=torch.long) + frame_id
            reference.commit_codes(codes)
            native.commit_codes(codes)

        assert native._slot_cache is not None and reference._slot_cache is not None
        assert native._native_qwen is not None
        terminal_audio_position = native._audio_start + 2
        visible_prefix_position = native._prefix_start + 1
        native_cache_before: dict[str, dict[int, torch.Tensor]] = {"_keys": {}, "_values": {}}
        for layer_index in range(28):
            key_cache, value_cache = native._native_qwen.debug_copy_layer_cache_async(layer_index)
            native_cache_before["_keys"][layer_index] = key_cache.clone()
            native_cache_before["_values"][layer_index] = value_cache.clone()
        expected_terminal = reference.step_terminal()
        actual_terminal = native.step_terminal()
        frame_metrics.append(parity("terminal Q=1", expected_terminal.last_hidden, actual_terminal.last_hidden))
        frame_metrics.append(parity("terminal codec logits", expected_terminal.logits, actual_terminal.logits))
        assert expected_terminal.is_terminal_query and actual_terminal.is_terminal_query
        assert native._current_frame_id == 2, "terminal query must not consume an acoustic frame"

        for layer_index in range(28):
            key_cache, value_cache = native._native_qwen.debug_copy_layer_cache_async(layer_index)
            for name, actual_cache in (("_keys", key_cache), ("_values", value_cache)):
                before = native_cache_before[name][layer_index]
                assert torch.equal(
                    actual_cache[:, :, :terminal_audio_position, :],
                    before[:, :, :terminal_audio_position, :],
                ), f"terminal query rewrote an existing {name} row at layer {layer_index}"
                assert torch.equal(
                    actual_cache[:, :, terminal_audio_position + 1 :, :],
                    before[:, :, terminal_audio_position + 1 :, :],
                ), f"terminal query wrote beyond its single {name} row at layer {layer_index}"
                expected_cache = getattr(reference._slot_cache, name)[layer_index]
                frame_metrics.append(parity(
                    f"terminal {name} layer {layer_index}",
                    expected_cache[:, :, terminal_audio_position, :],
                    actual_cache[:, :, terminal_audio_position, :],
                ))
        torch.cuda.synchronize()
        runtime = native.runtime_info()
        assert runtime["joint_steps"] == 2
        assert runtime["terminal_query_count"] == 1
        assert runtime["terminal_prefix_cache_appends"] == 0
        assert runtime["terminal_audio_cache_appends"] == 1
        assert runtime["native_terminal_query_position"] == terminal_audio_position
        assert runtime["native_terminal_visible_prefix_position"] == visible_prefix_position
        assert runtime["terminal_query_frame_id"] == 2
        assert runtime["terminal_query_position"] == 2
        assert runtime["current_audio_position"] == terminal_audio_position
        assert runtime["acoustic_kv_position"] == 1
        trace = native.native_trace_report()
        validated_trace = validate_native_trace_contract(
            trace,
            expected_step_pairs=[(0, 0), (0, 1)],
            expected_abi_version=NATIVE_TRACE_ABI_VERSION,
            expected_topology_sha256=trace["topology_sha256"],
        )
        terminal_evidence = validated_trace["native_completion"]["terminal_query"]
        assert terminal_evidence["count"] == 1
        assert terminal_evidence["query_length"] == 1
        assert terminal_evidence["prefix_cache_appends"] == 0
        assert terminal_evidence["audio_cache_appends"] == 1
        assert terminal_evidence["audio_position"] == terminal_audio_position
        print({
            "joint_q2_frames": 2,
            "terminal_query_length": 1,
            "parity_checks": len(frame_metrics),
            "mae_max": max(item["mae"] for item in frame_metrics),
            "max_abs_max": max(item["max_abs"] for item in frame_metrics),
            "cosine_min": min(item["cosine"] for item in frame_metrics),
            "native_positions": {
                key: runtime[key]
                for key in (
                    "current_prefix_position",
                    "current_audio_position",
                    "native_terminal_query_position",
                    "native_terminal_visible_prefix_position",
                    "terminal_query_count",
                    "terminal_prefix_cache_appends",
                    "terminal_audio_cache_appends",
                    "joint_steps",
                    "acoustic_kv_position",
                )
            },
            "trace_terminal_query": {
                key: terminal_evidence[key]
                for key in (
                    "query_length",
                    "visible_prefix_position",
                    "audio_position",
                    "prefix_cache_appends",
                    "audio_cache_appends",
                )
            },
        }, flush=True)
    finally:
        native.close()
        reference.close()
