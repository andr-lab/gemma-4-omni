from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.duplex_qlora.stage1 import run_audio_text_probe


class _Tokenizer:
    all_special_ids = [2]

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert text == "Ciao mondo."
        return {"input_ids": [10, 11, 12]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        if add_generation_prompt:
            return "<user>parlato</user><assistant>"
        return "<user>parlato</user><assistant>Ciao mondo.</assistant>"

    def decode(self, token_ids, *, skip_special_tokens):
        pieces = {3: "prefix", 4: "prompt", 10: "Ciao", 11: " mondo", 12: ".", 2: "<eos>"}
        return "".join(pieces[int(token)] for token in token_ids if not (skip_special_tokens and int(token) == 2))


class _Processor:
    tokenizer = _Tokenizer()


class _ProbeModel:
    def __init__(self):
        self.training = True
        self.calls = []
        self.shared_attentions = [
            type("Attention", (), {
                "layer_idx": 15,
                "is_kv_shared_layer": True,
                "kv_shared_layer_index": 13,
            })(),
            type("Attention", (), {
                "layer_idx": 16,
                "is_kv_shared_layer": True,
                "kv_shared_layer_index": 14,
            })(),
        ]

    def named_modules(self):
        return [("", self)] + [
            (f"attention_{index}", attention)
            for index, attention in enumerate(self.shared_attentions)
        ]

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def __call__(self, **kwargs):
        self.calls.append(set(kwargs))
        labels = kwargs["labels"]
        seq_len = labels.shape[1]
        logits = torch.zeros((1, seq_len, 16), dtype=torch.float32)
        logits[..., 4] = 1.0 if "input_features" in kwargs else 0.0
        cache = type("DynamicCache", (), {"shared_layers": {13: object(), 14: object()}})()
        return type("Output", (), {
            "loss": torch.tensor(1.25 if "input_features" in kwargs else 1.5),
            "logits": logits,
            "past_key_values": cache,
        })()


def _record():
    return {
        "task_kind": "conversation_response",
        "audio_mask": True,
        "target_text": "Ciao mondo.",
        "synthetic": True,
        "language": "it",
    }


def _inputs():
    return {
        "input_ids": torch.tensor([[1, 3, 4, 10, 11, 12, 2]]),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 10, 11, 12, 2]]),
        "input_features": torch.ones((1, 2, 4), dtype=torch.float32),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }


def test_audio_text_probe_validates_assistant_only_labels_and_removes_only_audio_tensors():
    model = _ProbeModel()
    report = run_audio_text_probe(
        model,
        _Processor(),
        _record(),
        _inputs(),
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    assert report["prompt_generation_header"] is True
    assert report["decoded_target_matches"] is True
    assert report["eot_present_after_target"] is True
    assert report["assistant_target_token_count"] == 3
    assert report["causal_target_prediction_count"] == 4
    assert report["audio_loss"] == 1.25
    assert report["text_only_loss"] == 1.5
    assert report["max_target_logit_abs_diff"] > 0
    assert report["native_shared_kv_cache"]["first_shared_layer_index"] == 15
    assert report["native_shared_kv_cache"]["expected_cache_source_indices"] == [13, 14]
    assert report["native_shared_kv_cache"]["observed_cache_source_indices"] == [13, 14]
    assert len(model.calls) == 2
    assert "input_features" in model.calls[0] and "input_features_mask" in model.calls[0]
    assert "input_features" not in model.calls[1] and "input_features_mask" not in model.calls[1]
    assert model.training is True


def test_audio_text_probe_rejects_labels_that_include_nonassistant_prompt_tokens():
    inputs = _inputs()
    inputs["labels"][0, 1] = 3
    try:
        run_audio_text_probe(
            _ProbeModel(), _Processor(), _record(), inputs,
            device=torch.device("cpu"), compute_dtype=torch.bfloat16,
        )
    except ValueError as exc:
        assert "label" in str(exc).lower()
    else:
        raise AssertionError("probe ha accettato label che includono il prompt")


if __name__ == "__main__":
    test_audio_text_probe_validates_assistant_only_labels_and_removes_only_audio_tensors()
    test_audio_text_probe_rejects_labels_that_include_nonassistant_prompt_tokens()
    print("Stage1 audio/text probe CPU smoke: 2 tests PASS")
