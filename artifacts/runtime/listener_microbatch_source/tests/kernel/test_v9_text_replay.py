from types import SimpleNamespace
import threading

import torch

from duplex_kernel.engine import PipelineEngine


class _Tokenizer:
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert len(messages) == 1
        return f"chat:{messages[0]['content']}"

    def __call__(self, text, *, return_tensors, add_special_tokens=True):
        assert return_tensors == "pt"
        if text.startswith("chat:"):
            ids = [10, 11]
        else:
            assert add_special_tokens is False
            ids = [21, 22, 23]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _Thinker:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        length = int(kwargs["input_ids"].shape[1])
        hidden = torch.arange(length * 2, dtype=torch.float32).reshape(1, length, 2)
        return SimpleNamespace(hidden_states=(hidden,))


def _engine(source):
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.v9_conditioning_source = source
    engine.device = "cpu"
    engine.tokenizer = _Tokenizer()
    engine.thinker = _Thinker()
    engine._thinker_lock = threading.Lock()
    return engine


def test_live_generated_v9_conditioning_remains_the_default_path():
    engine = _engine("live_generated")
    live_hidden = torch.ones((1, 5, 2))

    hidden, prompt_len, metrics = engine.controller_say_v9_input(
        live_hidden,
        2,
        "ciao",
    )

    assert hidden is live_hidden
    assert prompt_len == 2
    assert metrics["source"] == "live_generated"
    assert engine.thinker.calls == []


def test_text_replay_uses_exact_text_only_suffix_and_no_persistent_cache():
    engine = _engine("v9_text_replay")

    hidden, prompt_len, metrics = engine.controller_say_v9_input(
        torch.ones((1, 8, 2)),
        3,
        "Ciao, ci sono.",
    )

    call = engine.thinker.calls[0]
    assert call["input_ids"].tolist() == [[10, 11, 21, 22, 23]]
    assert call["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert call["use_cache"] is False
    assert call["output_hidden_states"] is True
    assert call["return_dict"] is True
    assert call["logits_to_keep"] == 1
    assert prompt_len == 2
    assert hidden.shape == (1, 5, 2)
    assert hidden[:, prompt_len:, :].tolist() == [[[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]]
    assert metrics["source"] == "v9_text_replay"
    assert metrics["prompt_tokens"] == 2
    assert metrics["target_tokens"] == metrics["hidden_target_tokens"] == 3

