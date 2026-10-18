from __future__ import annotations

from duplex_kernel.server import (
    _filter_frame_residual_runtime_metrics,
    _filter_turn_local_speech_runtime_metrics,
    _format_turn_local_speech_summary,
)


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
        "frame_residual_live_first_last_hidden_cosine": 0.519,
        "qwen_route": "plain_native_audio_q1",
        "qwen_prefix_slots_added": 0,
        "qwen_semantic_text_unchanged": True,
        "source": "gemma_assistant_response_posttoken_hidden",
        "frame_residual_live_hidden_trace": [
            {
                "qwen_frame_step": 0,
                "gemma_token_id": 1201,
                "gemma_kv_position": 261,
                "qwen_token_id": 991,
                "qwen_packet_token_index": 0,
                "qwen_packet_token_count": 1,
                "tail_hold": False,
                "gemma_hidden_norm": 38.25,
                "gemma_hidden_cosine_previous_packet": None,
            }
        ],
        "untrusted_extra": "drop me",
    }

    result = _filter_frame_residual_runtime_metrics(values)

    assert result == {key: value for key, value in values.items() if key != "untrusted_extra"}


def test_frame_residual_trace_is_scalar_bounded_and_rejects_invalid_entries() -> None:
    good = {
        "qwen_frame_step": 0,
        "gemma_token_id": 1201,
        "gemma_kv_position": 261,
        "qwen_token_id": 991,
        "qwen_packet_token_index": 0,
        "qwen_packet_token_count": 1,
        "tail_hold": False,
        "gemma_hidden_norm": 38.25,
        "gemma_hidden_cosine_previous_packet": None,
    }
    bad = {**good, "gemma_hidden_norm": float("inf")}
    result = _filter_frame_residual_runtime_metrics({
        "frame_residual_live_hidden_trace": [good, bad] * 200,
    })
    assert len(result["frame_residual_live_hidden_trace"]) == 128
    assert all(
        value is None or type(value) in {bool, int, float}
        for entry in result["frame_residual_live_hidden_trace"]
        for value in entry.values()
    )
    assert _filter_frame_residual_runtime_metrics({
        "frame_residual_live_hidden_trace": [{**good, "qwen_frame_step": True}],
    }) == {}


def test_frame_residual_metric_filter_rejects_wrong_types_and_nonfinite_values() -> None:
    values = {
        "frame_residual_calls": True,
        "frame_residual_qwen_q1_only": 1,
        "frame_residual_rms": float("nan"),
        "frame_residual_learned_scale": "0.05",
        "frame_residual_live_first_last_hidden_cosine": 1.5,
    }

    assert _filter_frame_residual_runtime_metrics(values) == {}


def test_turn_local_speech_metrics_keep_bounded_fifo_and_pacer_evidence() -> None:
    values = {
        "turn_local_fifo_capacity": 4,
        "turn_local_fifo_remaining_packets": 0,
        "turn_local_fifo_peak_packets": 4,
        "turn_local_last_vocalized_kv_position": 278,
        "turn_local_vocalized_cursor_is_playback_ack": False,
        "live_gemma_token_pacer": {
            "tick_ms": 80,
            "max_tokens_per_tick": 2,
            "total_tokens": 34,
            "peak_tokens_in_tick": 2,
        },
        "realtime_pcm_packet_queue_capacity": 4,
        "native_q1_import_audio_frame_capacity": 512,
        "native_q1_live_frame_cap": 512,
        "untrusted_extra": "drop me",
    }

    assert _filter_turn_local_speech_runtime_metrics(values) == {
        key: value for key, value in values.items() if key != "untrusted_extra"
    }


def test_turn_local_speech_metric_filter_rejects_wrong_types_and_over_cap() -> None:
    assert _filter_turn_local_speech_runtime_metrics({
        "turn_local_fifo_peak_packets": True,
        "live_gemma_token_pacer": {
            "tick_ms": 80,
            "max_tokens_per_tick": 2,
            "total_tokens": 34,
            "peak_tokens_in_tick": 3,
        },
    }) == {}


def test_turn_local_speech_summary_formats_runtime_values_without_external_scope() -> None:
    line = _format_turn_local_speech_summary("session123", {
        "turn_local_fifo_capacity": 4,
        "turn_local_fifo_peak_packets": 4,
        "turn_local_fifo_remaining_packets": 0,
        "turn_local_last_vocalized_kv_position": 278,
        "live_gemma_token_pacer": {
            "tick_ms": 80,
            "max_tokens_per_tick": 2,
            "total_tokens": 34,
            "peak_tokens_in_tick": 2,
        },
        "realtime_pcm_packet_queue_capacity": 4,
        "frame_residual_calls": 23,
        "frame_residual_tail_hold_calls": 85,
        "native_q1_live_frame_cap": 512,
        "frame_residual_live_hidden_trace": [{"qwen_frame_step": 0}],
        "frame_residual_live_first_last_hidden_cosine": 0.51,
    })
    assert "session=session123" in line
    assert "fifo_peak=4" in line
    assert "pacer_tokens=34" in line
    assert "hidden_trace_entries=1" in line
    assert "first_last_hidden_cosine=0.51" in line
