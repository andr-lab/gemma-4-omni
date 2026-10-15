from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.duplex_qlora.stage1 import (
    _format_chat,
    _training_records,
    build_training_inputs,
    run_text_conditioning_probe,
)


class _Tokenizer:
    audio_token = "<|audio|>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        self.calls.append([dict(message) for message in messages])
        serialized = "\n".join(f"{row['role']}:{row['content']}" for row in messages)
        return serialized + ("\nmodel" if add_generation_prompt else "\nassistant-target")

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert text == "Risposta finale."
        return {"input_ids": [10, 11]}

    def decode(self, token_ids, *, skip_special_tokens):
        return "Risposta finale." if [int(value) for value in token_ids] == [10, 11] else ""


class _Processor:
    audio_token = "<|audio|>"

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.calls = []

    def __call__(self, *, text, return_tensors, padding, **kwargs):
        self.calls.append({"text": text[0], "keys": set(kwargs)})
        has_target = text[0].endswith("\nassistant-target")
        ids = [1, 2, 3] if "audio" in kwargs else [1, 2]
        if has_target:
            ids += [10, 11]
        encoded = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }
        if "audio" in kwargs:
            encoded["input_features"] = torch.ones((1, 2, 4), dtype=torch.float32)
            encoded["input_features_mask"] = torch.ones((1, 2), dtype=torch.long)
        return encoded


class _TextProbeModel:
    def __init__(self):
        self.training = True
        self.calls = []

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def __call__(self, **kwargs):
        self.calls.append(set(kwargs))
        seq_len = kwargs["input_ids"].shape[1]
        logits = torch.zeros((1, seq_len, 16), dtype=torch.float32)
        return type("Output", (), {"loss": torch.tensor(2.0), "logits": logits})()


def _conversation(audio_path: Path):
    return {
        "conversation_id": "runtime-mode-test",
        "source_group_id": "family-1",
        "source": "synthetic_it_t6_v4",
        "synthetic": True,
        "audio_path": str(audio_path),
        "sample_rate_hz": 16000,
        "duration_ms": 1000,
        "language": "it",
        "channel_map": {"0": "user", "1": "assistant"},
        "segments": [
            {"role": "asr", "task_kind": "asr", "text": "frase ASR completa", "start_ms": 50, "end_ms": 950,
             "audio_channel_verified": True, "channel_index": 2, "text_mask": True, "time_mask": True},
            {"role": "user", "text": "Ciao, sono Luca.", "start_ms": 100, "end_ms": 250,
             "text_mask": True, "time_mask": True},
            {"role": "assistant", "text": "Ti ascolto.", "start_ms": 300, "end_ms": 500,
             "target_mask": False, "text_mask": True, "time_mask": True},
            {"role": "user", "text": "parole ritagliate", "start_ms": 510, "end_ms": 600,
             "text_mask": False, "time_mask": True},
            {"role": "user", "text": "Ecco la domanda.", "start_ms": 610, "end_ms": 680,
             "text_mask": True, "time_mask": True},
            {"role": "user", "text": "testo futuro non ancora pronunciato", "start_ms": 690, "end_ms": 750,
             "text_mask": True, "time_mask": True},
            {"role": "assistant", "text": "Risposta finale.", "start_ms": 700, "end_ms": 900,
             "target_mask": True, "text_mask": True, "time_mask": True},
        ],
    }


def test_runtime_asr_text_uses_complete_causal_dialog_and_asr_uses_full_target_channel(tmp_path):
    audio_path = tmp_path / "three-channel.wav"
    sf.write(audio_path, np.ones((16000, 3), dtype=np.float32) * 0.1, 16000, subtype="FLOAT")
    records = _training_records([_conversation(audio_path)], conditioning_mode="runtime_asr_text")
    response = next(row for row in records if row["task_kind"] == "conversation_response")
    asr = next(row for row in records if row["task_kind"] == "asr")

    assert response["conditioning_mode"] == "runtime_asr_text"
    assert response["audio_mask"] is False
    assert response["audio_start_ms"] is None and response["audio_end_ms"] is None
    assert [row["role"] for row in response["history_messages"]] == ["user", "assistant", "user"]
    assert response["history_messages"][0]["content"] == "Ciao, sono Luca."
    assert response["history_messages"][1]["content"] == "Ti ascolto."
    assert response["history_messages"][2]["content"] == "Ecco la domanda."
    assert "parole ritagliate" not in response["context_text"]
    assert "testo futuro non ancora pronunciato" not in response["context_text"]

    assert asr["conditioning_mode"] == "native_audio_asr"
    assert asr["audio_mask"] is True
    assert asr["audio_channel_index"] == 2
    assert (asr["audio_start_ms"], asr["audio_end_ms"]) == (50.0, 950.0)
    assert asr["audio_history_cap_ms"] is None

    processor = _Processor()
    response_for_build = dict(response, audio_path=str(tmp_path / "audio-must-not-be-read.wav"))
    response_inputs = build_training_inputs(processor, response_for_build)
    assert "input_features" not in response_inputs
    assert all("audio" not in call["keys"] for call in processor.calls)
    assert [row["role"] for row in processor.tokenizer.calls[-1][1:-1]] == ["user", "assistant", "user"]

    text_probe_model = _TextProbeModel()
    text_probe = run_text_conditioning_probe(
        text_probe_model,
        processor,
        response,
        response_inputs,
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    assert text_probe["conditioning_mode"] == "runtime_asr_text"
    assert text_probe["text_only_loss"] == 2.0
    assert text_probe["audio_placeholder_count"] == 0
    assert text_probe["input_features_present"] is False
    assert "input_features" not in text_probe_model.calls[0]
    assert "input_features_mask" not in text_probe_model.calls[0]
    assert text_probe_model.training is True

    processor.calls.clear()
    processor.tokenizer.calls.clear()
    asr_inputs = build_training_inputs(processor, asr)
    assert "input_features" in asr_inputs
    assert all("audio" in call["keys"] for call in processor.calls)
    assert "in Italian" in processor.tokenizer.calls[-1][0]["content"]


def test_conversation_instruction_matches_target_language_in_chat_messages():
    processor = _Processor()
    cases = [
        ("Italian", "Rispondi in italiano"),
        ("cn", "请用中文"),
        ("en-US", "Reply in English"),
        ("", "same language"),
    ]
    for language, expected_instruction in cases:
        record = {
            "task_kind": "conversation_response",
            "conditioning_mode": "runtime_asr_text",
            "language": language,
            "history_messages": [{"role": "user", "content": "messaggio trascritto"}],
            "target_text": "risposta",
        }
        _format_chat(processor, record, add_target=False)
        system_message = processor.tokenizer.calls[-1][0]
        assert system_message["role"] == "system"
        assert expected_instruction in system_message["content"]


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="runtime-asr-text-") as temporary:
        root = Path(temporary)
        test_runtime_asr_text_uses_complete_causal_dialog_and_asr_uses_full_target_channel(root)
    test_conversation_instruction_matches_target_language_in_chat_messages()
    print("Stage1 runtime_asr_text CPU smoke: 2 tests PASS")
