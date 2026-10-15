from __future__ import annotations

import torch
from fastapi.testclient import TestClient
from types import SimpleNamespace
import numpy as np

from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    format_audio_response_v2_target,
)
from duplex_kernel.server import (
    _direct_audio_prompt_history,
    _raw_source_audio_quality,
    create_app,
)
from duplex_kernel.session import SessionManager
from duplex_kernel.turn_policy import NoOpPolicy


class _Tokenizer:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.seen_messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.seen_messages.append([dict(message) for message in messages])
        return " | ".join(message["content"] for message in messages)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        fixed_ids = {
            "SPEAK": 41,
            "BACKCHANNEL": 42,
            "REPLAN": 43,
            "{": 100,
            '"heard":"': 101,
            '"': 102,
            ',"action":"': 103,
            ',"say":"': 104,
            "}": 105,
        }
        if text in fixed_ids:
            return [fixed_ids[text]]
        return [4096 + ord(char) for char in text]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        fixed_text = {
            41: "SPEAK",
            42: "BACKCHANNEL",
            43: "REPLAN",
            100: "{",
            101: '"heard":"',
            102: '"',
            103: ',"action":"',
            104: ',"say":"',
            105: "}",
        }
        return "".join(
            fixed_text[int(token_id)]
            if int(token_id) in fixed_text
            else chr(int(token_id) - 4096)
            for token_id in token_ids
        )


class _Processor:
    audio_token = "<|audio|>"

    def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.ones((1, 4, 8)),
            "input_features_mask": torch.ones((1, 4), dtype=torch.long),
        }


def test_raw_source_audio_quality_measures_float_pcm_before_wav_encoding():
    pcm = np.asarray([-1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25], dtype=np.float32)

    quality = _raw_source_audio_quality(pcm)

    expected_rms = round(float(np.sqrt(np.mean(np.square(pcm.astype(np.float64))))), 8)
    assert quality == {
        "finite": True,
        "sample_count": 7,
        "peak": 1.25,
        "rms": expected_rms,
        "clipping_rail_samples": 4,
    }


class _V2Engine:
    def __init__(self) -> None:
        self.loaded = True
        self.enforce_production_listener = False
        self.default_turn_policy_factory = NoOpPolicy
        self.tokenizer = _Tokenizer()
        self._thinker_wrapper = SimpleNamespace(processor=_Processor())
        self.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
        self.say_hidden_seen: torch.Tensor | None = None
        self.spoken_text: str | None = None
        self.omit_closing_brace = False
        self.no_speech = False
        self.v9_dual_audio_context_experimental = False
        self.capture_audio_context_requested = False
        self.dual_v9_raw_seen = None
        self.dual_v9_anchor_seen = None

    def think(self, _ids, **kwargs):
        self.capture_audio_context_requested = bool(kwargs.get("capture_audio_context", False))
        assert kwargs["on_generation_start"]() == 100
        target = format_audio_response_v2_target(
            heard="trascrizione privata",
            action="SPEAK",
            say="Ciao!",
        )
        if self.omit_closing_brace:
            target = target[:-1]
        body_open = target.index('"say":"') + len('"say":"')
        closing_width = 2 if target.endswith('"}') else 1
        fragments = [
            target[: body_open + 1],
            target[body_open + 1 : -closing_width],
            target[-closing_width:],
        ]
        callback = kwargs["on_generated_token"]
        candidate_callback = kwargs["on_candidate_token"]
        assert callable(kwargs["on_candidate_fallback"])
        assert kwargs["on_text_chunk"] is None
        assert kwargs["on_token_id"] is None
        for token_index, (fragment, value) in enumerate(zip(fragments, (11.0, 22.0, 33.0), strict=True)):
            assert candidate_callback(100 + token_index, fragment, False)
            callback(
                token_index,
                100 + token_index,
                fragment,
                torch.full((1, 1, 4), value),
            )
        assert candidate_callback(3, "", True)
        result = {
            "text": target,
            "hidden": torch.full((1, 1, 4), -99.0),
            "prompt_len": 5,
            "gen_ids": [100, 101, 102],
            "runtime_metrics": {},
        }
        if self.capture_audio_context_requested:
            result["audio_context_snapshot"] = SimpleNamespace(
                hidden=torch.full((1, 2, 4), 77.0),
                audio_placeholder_count=8,
                selected_frame_count=2,
                past_length=0,
            )
        return result

    def v9_prefix(self, hidden, prompt_len):
        assert prompt_len == 0
        self.say_hidden_seen = hidden.detach().clone()
        return hidden

    def v9_prefix_audio_context(self, audio_context_hidden, semantic_anchor_hidden):
        self.dual_v9_raw_seen = audio_context_hidden.detach().clone()
        self.dual_v9_anchor_seen = semantic_anchor_hidden.detach().clone()
        assert tuple(audio_context_hidden.shape) == (1, 2, 4)
        assert tuple(semantic_anchor_hidden.shape) == (1, 4)
        projected = torch.full((1, 2, 4), 88.0)
        self.say_hidden_seen = projected.detach().clone()
        return SimpleNamespace(h_pfx=projected)

    def talk(self, text, h_pfx, *, metrics=None, **_kwargs):
        self.spoken_text = text
        assert self.say_hidden_seen is not None
        assert torch.equal(h_pfx, self.say_hidden_seen)
        if self.no_speech:
            assert metrics is not None
            metrics.update({
                "eos_detected": True,
                "eos_frame_id": 0,
                "generated_cb0_frames": 0,
                "generated_pcm_samples": 0,
            })
            return np.empty(0, dtype=np.float32)
        return torch.zeros(1600).numpy()


def test_v2_runtime_only_returns_says_text_and_v9_hidden_after_json_validation(monkeypatch) -> None:
    from duplex_kernel import server as server_module

    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    engine = _V2Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "cronologia precedente"},
                {"role": "assistant", "content": "risposta precedente"},
                {"role": "user", "content": "trascrizione corrente da non ripetere nel prompt"},
            ],
            "modalities": ["text", "audio"],
            "_direct_audio_pcm": [0.1] * 1280,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "trascrizione privata" not in response.text
    assert "action" not in payload["choices"][0]["message"]
    assert payload["choices"][0]["message"]["content"] == "Ciao!"
    assert engine.spoken_text == "Ciao!"
    assert engine.say_hidden_seen is not None
    assert engine.say_hidden_seen.shape == (1, 2, 4)
    assert engine.say_hidden_seen[0, :, 0].tolist() == [11.0, 22.0]
    assert all(
        "trascrizione privata" not in str(message)
        for message in sessions.get(payload["session_id"]).history
    )
    prompt = repr(engine.tokenizer.seen_messages[-1])
    assert "trascrizione privata" not in prompt
    assert "trascrizione corrente da non ripetere nel prompt" not in prompt
    assert "cronologia precedente" in prompt
    assert "risposta precedente" in prompt
    assert "SPEAK" in prompt
    assert "<|audio|>" in prompt


def test_v2_runtime_rejects_incomplete_json_before_v9_or_talker(monkeypatch) -> None:
    from duplex_kernel import server as server_module

    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    engine = _V2Engine()
    engine.omit_closing_brace = True
    client = TestClient(create_app(engine, SessionManager()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "cronologia precedente"},
                {"role": "assistant", "content": "risposta precedente"},
                {"role": "user", "content": "trascrizione corrente da non ripetere nel prompt"},
            ],
            "modalities": ["text", "audio"],
            "_direct_audio_pcm": [0.1] * 1280,
        },
    )

    assert response.status_code == 500
    assert engine.say_hidden_seen is None
    assert engine.spoken_text is None


def test_experimental_v9_uses_prefill_audio_snapshot_and_validated_say_anchor(monkeypatch) -> None:
    from duplex_kernel import server as server_module

    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    engine = _V2Engine()
    engine.v9_dual_audio_context_experimental = True
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "turno precedente"},
                {"role": "assistant", "content": "risposta precedente"},
                {"role": "user", "content": "testo corrente privato"},
            ],
            "modalities": ["text", "audio"],
            "_direct_audio_pcm": [0.1] * 1280,
        },
    )

    assert response.status_code == 200
    assert engine.capture_audio_context_requested is True
    assert torch.equal(engine.dual_v9_raw_seen, torch.full((1, 2, 4), 77.0))
    # The SAY collector supplied post-token values 11 and 22, never the
    # private transcript or JSON/action-token hidden states.
    assert torch.equal(engine.dual_v9_anchor_seen, torch.full((1, 4), 16.5))
    assert engine.spoken_text == "Ciao!"
    assert "77.0" not in response.text and "BACKCHANNEL" not in response.text


def test_v2_empty_first_codec_frame_returns_typed_no_speech_without_history_commit(monkeypatch) -> None:
    from duplex_kernel import server as server_module

    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    engine = _V2Engine()
    engine.no_speech = True
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "contesto precedente"},
                {"role": "assistant", "content": "risposta precedente"},
                {"role": "user", "content": "audio corrente"},
            ],
            "modalities": ["text", "audio"],
            "_direct_audio_pcm": [0.1] * 1280,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert message["content"] == ""
    assert message["audio_status"] == "no_speech"
    assert "audio" not in message
    session = sessions.get(payload["session_id"])
    assert all(
        not (entry.get("role") == "assistant" and entry.get("content") == "Ciao!")
        for entry in session.history
    )
    assert engine.spoken_text == "Ciao!"


def test_direct_audio_history_trims_only_uncommitted_api_current_user() -> None:
    messages = [
        {"role": "user", "content": "user precedente"},
        {"role": "assistant", "content": "assistant precedente"},
        {"role": "user", "content": "user corrente o turno utente senza risposta dopo cancel"},
    ]

    assert _direct_audio_prompt_history(messages, history_is_committed=False) == messages[:2]
    assert _direct_audio_prompt_history(messages, history_is_committed=True) == messages
