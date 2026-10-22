from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_qwen_production_attention_is_exact_cuda_graph_and_allocation_free() -> None:
    executor = (ROOT / "native/qwen_talker/qwen_talker.cu").read_text(encoding="utf-8")
    aten = (ROOT / "native/qwen_talker/qwen_talker_aten_attention.cpp").read_text(encoding="utf-8")

    production = executor.split("int execute_layer(", 1)[1].split("int validate_creation(", 1)[0]
    assert "qwen_talker_attention_graph_run(" in production
    assert "qwen_talker_attention_gqa(" not in production
    assert "attention_scores" not in executor
    assert "cudaGraphLaunch" in aten
    assert "qwen_talker_attention_graph_captured" in aten


def test_qwen_atten_bridge_is_capture_only_not_the_steady_state_executor() -> None:
    executor = (ROOT / "native/qwen_talker/qwen_talker.cu").read_text(encoding="utf-8")
    aten = (ROOT / "native/qwen_talker/qwen_talker_aten_attention.cpp").read_text(encoding="utf-8")
    production = executor.split("int execute_layer(", 1)[1].split("int validate_creation(", 1)[0]

    assert "qwen_talker_attention_gqa(" not in production
    assert "qwen_talker_attention_gqa(" in aten
    assert "cudaStreamBeginCapture" in aten


def test_qwen_hidden_rmsnorm_is_exact_aten_cuda_graph() -> None:
    root = ROOT

    bridge = (
        root
        / "native/qwen_talker/qwen_talker_aten_rmsnorm.cpp"
    ).read_text(encoding="utf-8")

    executor = (
        root
        / "native/qwen_talker/qwen_talker.cu"
    ).read_text(encoding="utf-8")

    makefile = (
        root
        / "native/qwen_talker/Makefile"
    ).read_text(encoding="utf-8")

    wrapper = (
        root
        / "duplex_kernel/native_qwen_talker.py"
    ).read_text(encoding="utf-8")

    assert "state->input_bf16.to(at::kFloat)" in bridge
    assert "state->input_fp32.pow(2)" in bridge
    assert "state->squared_fp32.mean({-1}, true)" in bridge
    assert "state->variance_eps_fp32 = " in bridge
    assert "state->variance_eps_fp32.rsqrt()" in bridge
    assert "        !stream ||" not in bridge
    assert (
        "state->weight_bf16 * state->normalized_bf16"
        in bridge
    )

    assert "cudaStreamBeginCapture" in bridge
    assert "cudaGraphInstantiate" in bridge
    assert "cudaGraphLaunch" in bridge

    assert (
        "qwen_talker_rmsnorm_graph_run("
        in executor
    )

    assert (
        "qwen_talker_aten_rmsnorm.o"
        in makefile
    )

    assert (
        "QWEN_HIDDEN_ATEN_GRAPH_QK_NATIVE_V1"
        in wrapper
    )

    execute_layer = executor.split(
        "int execute_layer(", 1
    )[1].split(
        "void bind_arena_pointers", 1
    )[0]

    # Hidden input + post-attention use graph.
    assert (
        execute_layer.count(
            "hidden_rmsnorm_exact_graph("
        )
        == 2
    )

    # Q and K head-dim norms stay native.
    assert (
        execute_layer.count(
            "qwen_rmsnorm_exact("
        )
        == 2
    )
