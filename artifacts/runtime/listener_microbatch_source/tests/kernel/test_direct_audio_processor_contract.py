from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from duplex_kernel.server import _prepare_direct_audio_processor_inputs


class _Tokenizer:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False and add_generation_prompt is True
        self.messages = messages
        return repr(messages)


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


def test_direct_processor_receives_audio_and_completed_history_without_transcript_or_action() -> None:
    tokenizer = _Tokenizer()
    processor = _Processor()
    engine = SimpleNamespace(tokenizer=tokenizer, _thinker_wrapper=SimpleNamespace(processor=processor))
    pcm = np.linspace(-0.5, 0.5, 1280, dtype=np.float32)

    ids, tensors, prepare_ms = _prepare_direct_audio_processor_inputs(
        engine,
        language="Italian",
        history_messages=[
            {"role": "system", "content": "legacy custom instructions"},
            {"role": "user", "content": "turno precedente"},
            {"role": "assistant", "content": "risposta precedente"},
        ],
        pcm=pcm,
    )

    assert ids == [1, 2, 3]
    assert set(tensors) == {"input_ids", "attention_mask", "input_features", "input_features_mask"}
    assert prepare_ms >= 0
    assert tokenizer.messages == [
        {
            "role": "system",
            "content": "Rispondi in modo naturale e pertinente in italiano, usando la cronologia e l'audio dell'utente.",
        },
        {"role": "user", "content": "turno precedente"},
        {"role": "assistant", "content": "risposta precedente"},
        {"role": "user", "content": "<|audio|>"},
    ]
    call_text, call_audio, sample_rate, return_tensors, padding = processor.calls[0]
    assert call_text == [repr(tokenizer.messages)]
    assert np.array_equal(call_audio[0], pcm)
    assert sample_rate == 16000 and return_tensors == "pt" and padding is True
    assert "SPEAK" not in call_text[0] and "BACKCHANNEL" not in call_text[0]


def test_direct_v2_processor_keeps_session_instructions_alongside_protocol_system() -> None:
    from duplex_kernel.audio_response_protocol import (
        AUDIO_RESPONSE_PROTOCOL_V2,
        AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM,
    )

    tokenizer = _Tokenizer()
    processor = _Processor()
    engine = SimpleNamespace(tokenizer=tokenizer, _thinker_wrapper=SimpleNamespace(processor=processor))

    _prepare_direct_audio_processor_inputs(
        engine,
        language="Italian",
        history_messages=[],
        pcm=np.ones(1280, dtype=np.float32),
        response_protocol=AUDIO_RESPONSE_PROTOCOL_V2,
    )
    assert tokenizer.messages == [
        {"role": "system", "content": AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM},
        {"role": "user", "content": "<|audio|>"},
    ]

    _prepare_direct_audio_processor_inputs(
        engine,
        language="Italian",
        history_messages=[{"role": "user", "content": "Cerco un ristorante."}],
        pcm=np.ones(1280, dtype=np.float32),
        response_protocol=AUDIO_RESPONSE_PROTOCOL_V2,
        session_instructions="Stiamo organizzando una cena con amici.",
    )

    system = tokenizer.messages[0]["content"]
    assert "Stiamo organizzando una cena con amici." in system
    assert system.endswith(AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM)
    assert tokenizer.messages[-2:] == [
        {"role": "user", "content": "Cerco un ristorante."},
        {"role": "user", "content": "<|audio|>"},
    ]
