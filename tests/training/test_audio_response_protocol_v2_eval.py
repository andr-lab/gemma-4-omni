from __future__ import annotations

import torch

from tools.eval_audio_response_protocol_v2 import _aggregate_direct, _generate_one, word_error_counts


def test_word_error_counts_uses_reference_word_denominator():
    assert word_error_counts("Il gatto salta!", "il cane salta") == (1, 3)
    assert word_error_counts("", "una parola") == (2, 0)


def test_direct_metrics_keep_backchannel_heard_and_brevity_separate():
    rows = {
        "speak": {
            "gold_action": "SPEAK", "protocol_valid": True, "action_correct": True,
            "heard_word_errors": 1, "heard_reference_words": 3, "heard_exact": False,
            "say_word_count": 2,
        },
        "backchannel": {
            "gold_action": "BACKCHANNEL", "protocol_valid": True, "action_correct": True,
            "heard_word_errors": 0, "heard_reference_words": 0, "heard_exact": True,
            "say_word_count": 5, "backchannel_at_most_four_words": False,
        },
        "replan": {
            "gold_action": "REPLAN", "protocol_valid": False,
            "heard_word_errors": 0, "heard_reference_words": 0,
        },
    }

    report = _aggregate_direct(rows)

    assert report["protocol_valid"] == 2
    assert report["action_accuracy"] == 2 / 3
    assert report["by_action"]["SPEAK"]["heard_wer"] == 1 / 3
    assert report["by_action"]["BACKCHANNEL"]["heard_empty_correct"] == 1
    assert report["by_action"]["BACKCHANNEL"]["backchannels_at_most_four_words"] == 0
    assert report["by_action"]["REPLAN"]["protocol_valid"] == 0


def test_generate_one_can_disable_kv_cache(monkeypatch):
    captured = {}

    class FakeTokenizer:
        eos_token_id = 106
        eot_id = None
        turn_token_id = None
        pad_token_id = 0

        def decode(self, token_ids, skip_special_tokens=True):
            return "recognized words" if token_ids == [9] else ""

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeModel:
        generation_config = None

        def generate(self, **kwargs):
            captured["use_cache"] = kwargs["use_cache"]
            return torch.tensor([[1, 2, 9, 106]])

    monkeypatch.setattr(
        "tools.eval_audio_response_protocol_v2._encode_prompt",
        lambda _processor, _record, *, device: {"input_ids": torch.tensor([[1, 2]], device=device)},
    )

    result = _generate_one(
        FakeModel(), FakeProcessor(), {
            "task_kind": "asr", "conversation_id": "conversation", "target_utterance_id": "utterance",
            "target_text": "reference",
        },
        device=torch.device("cpu"), max_new_tokens=20, use_cache=False,
    )

    assert captured["use_cache"] is False
    assert result["raw_generated_text"] == "recognized words"
    assert result["use_cache"] is False
