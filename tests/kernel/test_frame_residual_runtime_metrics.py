from __future__ import annotations

from duplex_kernel.server import _filter_frame_residual_runtime_metrics


def test_frame_residual_metrics_survive_typed_response_filter() -> None:
    values = {
        "frame_residual_adapter": "response_hidden_q1_input_residual_v1",
        "adapter_mode": "response_hidden_q1_query_film_v2",
        "frame_residual_checkpoint": "/workspace/checkpoints/adapter.pt",
        "frame_residual_calls": 17,
        "frame_residual_tail_hold_calls": 2,
        "frame_residual_qwen_q1_only": True,
        "frame_residual_prefix_tokens_added": 0,
        "frame_residual_learned_scale": 0.05,
        "frame_residual_rms": 0.012,
        "frame_residual_base_rms": 0.31,
        "frame_residual_rms_to_base_ratio": 0.0387,
        "qwen_route": "plain_native_audio_q1",
        "qwen_prefix_slots_added": 0,
        "qwen_semantic_text_unchanged": True,
        "source": "gemma_assistant_response_posttoken_hidden",
        "untrusted_extra": "drop me",
    }

    result = _filter_frame_residual_runtime_metrics(values)

    assert result == {key: value for key, value in values.items() if key != "untrusted_extra"}


def test_frame_residual_metric_filter_rejects_wrong_types_and_nonfinite_values() -> None:
    values = {
        "frame_residual_calls": True,
        "frame_residual_qwen_q1_only": 1,
        "frame_residual_rms": float("nan"),
        "frame_residual_learned_scale": "0.05",
    }

    assert _filter_frame_residual_runtime_metrics(values) == {}
