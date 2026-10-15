from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from duplex_kernel.server import (
    _append_native_action_token,
    _prepare_direct_audio_processor_inputs,
)


class _Tokenizer:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, _messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False and add_generation_prompt is True
        self.messages = _messages
        return "<start_of_turn>assistant\n"


class _Processor:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
        self.calls.append((text, audio, sampling_rate, return_tensors, padding))
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.ones((1, 4, 8)),
            "input_features_mask": torch.ones((1, 4), dtype=torch.long),
        }


def test_native_action_token_appender_is_normalized_idempotent_and_noop_for_other_actions() -> None:
    prompt = "<start_of_turn>assistant\n"

    assert _append_native_action_token(prompt, "speak") == prompt + "<unused90> "
    assert _append_native_action_token(prompt, " BACKCHANNEL ") == prompt + "<unused93> "
    assert _append_native_action_token(prompt, "replan") == prompt + "<unused94> "
    assert _append_native_action_token(prompt + "<unused90> ", "SPEAK") == prompt + "<unused90> "
    assert _append_native_action_token(prompt + "<unused90>\n", "SPEAK") == prompt + "<unused90>\n"
    assert _append_native_action_token(prompt + "<unused90>\t", "SPEAK") == prompt + "<unused90>\t"
    assert _append_native_action_token(prompt, "YIELD") == prompt
    assert _append_native_action_token(prompt, None) == prompt


def test_direct_processor_appends_controller_action_token_to_rendered_prompt() -> None:
    tokenizer = _Tokenizer()
    processor = _Processor()
    engine = SimpleNamespace(tokenizer=tokenizer, _thinker_wrapper=SimpleNamespace(processor=processor))

    _prepare_direct_audio_processor_inputs(
        engine,
        language="Italian",
        history_messages=[],
        pcm=np.ones(1280, dtype=np.float32),
        controller_action="BACKCHANNEL",
    )

    assert processor.calls[0][0] == ["<start_of_turn>assistant\n<unused93> "]


def test_controller_guided_direct_prompt_keeps_action_instruction_without_reserved_prefix() -> None:
    from duplex_kernel.audio_response_protocol import CONTROLLER_GUIDED_SAY_V1

    tokenizer = _Tokenizer()
    processor = _Processor()
    engine = SimpleNamespace(tokenizer=tokenizer, _thinker_wrapper=SimpleNamespace(processor=processor))
    instructions = "Sei una persona che organizza una cena con amici. Parla in modo naturale."

    _prepare_direct_audio_processor_inputs(
        engine,
        language="it",
        history_messages=[],
        pcm=np.ones(1280, dtype=np.float32),
        response_protocol=CONTROLLER_GUIDED_SAY_V1,
        session_instructions=instructions,
        controller_action="SPEAK",
    )

    system_text = tokenizer.messages[0]["content"]
    assert instructions in system_text
    assert "Il controller ha già selezionato SPEAK." in system_text
    assert processor.calls[0][0] == ["<start_of_turn>assistant\n"]
