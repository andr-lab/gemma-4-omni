from __future__ import annotations


def test_counterfactual_metrics_separate_protocol_action_heard_and_brevity():
    from tools.eval_audio_response_protocol_v2 import _aggregate_counterfactual

    rows = {
        "s1<-s2": {
            "gold_action": "SPEAK", "protocol_valid": True, "action_correct": True,
            "heard_word_errors": 1, "heard_reference_words": 10, "say_word_count": 12,
            "duration_ratio": 0.05,
        },
        "b1<-b2": {
            "gold_action": "BACKCHANNEL", "protocol_valid": True, "action_correct": True,
            "heard_word_errors": 0, "heard_reference_words": 0, "heard_exact": True,
            "say_word_count": 3, "backchannel_at_most_four_words": True, "duration_ratio": 0.10,
        },
    }

    result = _aggregate_counterfactual(rows)

    assert result["examples"] == 2
    assert result["by_action"]["SPEAK"]["heard_wer"] == 0.1
    assert result["by_action"]["BACKCHANNEL"]["heard_empty_correct"] == 1
    assert result["by_action"]["BACKCHANNEL"]["backchannels_at_most_four_words"] == 1
    assert result["by_action"]["SPEAK"]["mean_duration_ratio"] == 0.05


def test_counterfactual_pairing_is_unique_class_matched_and_duration_bounded():
    from tools.eval_audio_response_protocol_v2 import _counterfactual_pairs

    rows = []
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        for index, duration in enumerate((1000, 1050, 1480, 1530)):
            rows.append({
                "task_kind": "conversation_response",
                "action": action,
                "conversation_id": f"{action}-{index}",
                "target_utterance_id": "reply",
                "input_audio_duration_ms": duration,
                "audio_path": f"{action}-{index}.wav",
                "audio_start_ms": 0,
                "audio_end_ms": duration,
                "audio_channel_index": 0,
                "sample_rate_hz": 16000,
                "source_audio_sha256": f"{index:064x}",
                "target_text": f"target-{index}",
                "history_messages": [{"role": "assistant", "content": f"history-{index}"}],
            })

    pairs = _counterfactual_pairs(rows, per_action=2, max_duration_ratio=0.20)

    assert len(pairs) == 6
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        selected = [pair for pair in pairs if pair["action"] == action]
        assert len(selected) == 2
        assert len({_record_key(pair["target"]) for pair in selected}) == 2
        assert len({_record_key(pair["donor"]) for pair in selected}) == 2
        for pair in selected:
            assert pair["target"]["conversation_id"] != pair["donor"]["conversation_id"]
            assert pair["duration_ratio"] <= 0.20


def test_counterfactual_audio_replacement_preserves_target_context_and_labels():
    from tools.eval_audio_response_protocol_v2 import _replace_counterfactual_audio

    target = {
        "audio_path": "target.wav", "audio_start_ms": 100, "audio_end_ms": 900,
        "audio_channel_index": 0, "sample_rate_hz": 16000, "source_audio_sha256": "a" * 64,
        "history_messages": [{"role": "assistant", "content": "prior"}],
        "target_text": "target response", "target_heard_text": "target words", "action": "SPEAK",
    }
    donor = {
        "audio_path": "donor.wav", "audio_start_ms": 200, "audio_end_ms": 1000,
        "audio_channel_index": 1, "sample_rate_hz": 24000, "source_audio_sha256": "b" * 64,
        "history_messages": [{"role": "assistant", "content": "other"}],
        "target_text": "donor response", "target_heard_text": "donor words", "action": "SPEAK",
    }

    swapped = _replace_counterfactual_audio(target, donor)

    for field in ("audio_path", "audio_start_ms", "audio_end_ms", "audio_channel_index", "sample_rate_hz", "source_audio_sha256"):
        assert swapped[field] == donor[field]
    for field in ("history_messages", "target_text", "target_heard_text", "action"):
        assert swapped[field] == target[field]


def _record_key(record):
    return f"{record['conversation_id']}::{record['target_utterance_id']}"
