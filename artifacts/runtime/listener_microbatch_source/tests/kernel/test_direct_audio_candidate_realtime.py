from __future__ import annotations

import base64
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from duplex_kernel.server import create_app
from duplex_kernel.session import SessionManager
from tests.kernel.test_realtime_control_plane import FRAME_SAMPLES, _Engine, _send_pcm_level, _receive_until


class _Processor:
    audio_token = "<|audio|>"

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
        self.calls.append((text, np.array(audio[0], dtype=np.float32, copy=True), sampling_rate))
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.ones((1, 4, 8)),
            "input_features_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _DirectAudioEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.processor = _Processor()
        self._thinker_wrapper = SimpleNamespace(processor=self.processor)
        self.order = []
        self.direct_inputs = None
        self.archive_done = threading.Event()

    def think(self, _ids, **kwargs):
        self.order.append("thinker")
        self.direct_inputs = kwargs.get("input_audio")
        assert isinstance(self.direct_inputs, dict)
        assert "input_features" in self.direct_inputs
        result = super().think(_ids, **kwargs)
        result["runtime_metrics"] = {
            "thinker_cache_hit": True,
            "thinker_cache_reused_tokens": 7,
            "thinker_cache_reason": "exact_text_prefix_before_audio",
            "thinker_cache_publish": "committed",
        }
        return result

    def transcribe_audio(self, pcm):
        self.order.append("archive_asr")
        self.transcriptions.append(np.array(pcm, dtype=np.float32, copy=True))
        self.transcription_started.set()
        self.archive_done.set()
        return {"text": "testo utente archiviato", "phase_timings_ms": {"total_asr_ms": 1.0}}


def test_opt_in_realtime_uses_audio_for_thinker_then_archives_after_audio_packet(monkeypatch) -> None:
    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    engine = _DirectAudioEngine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {"turn_policy": "none"},
        })
        assert ws.receive_json()["type"] == "session.updated"
        _send_pcm_level(ws, 1000)
        _send_pcm_level(ws, 1000)
        ws.send_json({"type": "input_audio_buffer.commit"})
        events = _receive_until(ws, "response.done")

        assert engine.order[0] == "thinker"
        assert engine.archive_done.wait(timeout=3.0)
        session = sessions.all()[0]
        deadline = time.monotonic() + 2.0
        while session.pending_audio_history_archive is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.history[-2:] == [
            {"role": "user", "content": "testo utente archiviato"},
            {"role": "assistant", "content": "reply 1"},
        ]
        assert engine.order == ["thinker", "archive_asr"]
        assert engine.processor.calls[0][2] == 16000
        assert engine.processor.calls[0][1].size == 2 * FRAME_SAMPLES
        assert engine.direct_inputs["input_features"].shape == (1, 4, 8)
        done = next(event for event in events if event["type"] == "response.done")
        thinker_metrics = done["runtime_metrics"]["thinker"]
        assert thinker_metrics["thinker_cache_hit"] is True
        assert thinker_metrics["thinker_cache_reused_tokens"] == 7
        assert thinker_metrics["thinker_cache_reason"] == "exact_text_prefix_before_audio"
        assert thinker_metrics["thinker_cache_publish"] == "committed"
        prompt = repr(engine.tokenizer.seen_messages[-1])
        assert "<|audio|>" in prompt
        assert "recognized turn" not in prompt
        assert "Current turn action" not in prompt
        assert any(event["type"] == "response.audio.delta" for event in events)
        ws.send_json({"type": "session.close"})


def test_empty_audio_archive_skips_history_and_keeps_realtime_session_alive(monkeypatch) -> None:
    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    import duplex_kernel.server as server_module

    monkeypatch.setattr(server_module, "resolve_voice_conditioning", lambda *_args, **_kwargs: None)
    engine = _DirectAudioEngine()

    def transcribe_silence(_pcm):
        engine.archive_done.set()
        return {"text": "", "phase_timings_ms": {"total_asr_ms": 1.0}}

    engine.transcribe_audio = transcribe_silence
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "none",
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"

        for _ in range(2):
            engine.archive_done.clear()
            for _frame in range(2):
                _send_pcm_level(ws, 1000)
                _receive_until(ws, "input_audio_buffer.consumed")
            ws.send_json({"type": "input_audio_buffer.commit"})
            events = _receive_until(ws, "response.done")
            assert engine.archive_done.wait(timeout=3.0)
            assert not any(event["type"] == "response.failed" for event in events)

        session = sessions.all()[0]
        deadline = time.monotonic() + 2.0
        while session.pending_audio_history_archive is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.pending_audio_history_archive is None
        assert session.history == []
        ws.send_json({"type": "session.close"})


def test_audio_cache_smoke_uses_two_turn_memory_scenario() -> None:
    from tools.smoke_duplex_qlora_audio import _smoke_scenario

    scenario, turn_count = _smoke_scenario(require_cache_hit=True)

    assert turn_count == 2
    assert [step.speaker for step in scenario.steps] == ["user", "assistant", "user", "assistant"]
    assert "nome" in scenario.steps[2].expected_text.lower()
    assert "codice" in scenario.steps[2].expected_text.lower()


def test_audio_cache_smoke_requires_direct_audio_and_observed_prefix_reuse() -> None:
    from tools.smoke_duplex_qlora_audio import _summarize_response_events

    def response_done(turn: int, timestamp_ns: int) -> dict[str, object]:
        return {
            "event_type": "response.done",
            "turn_index": turn,
            "timestamp_ns": timestamp_ns,
            "details": {
                "response_id": f"r{turn}",
                "runtime_metrics": {
                    "direct_audio": {"pcm_samples": 16000},
                    "thinker": {
                        "thinker_cache_hit": turn == 2,
                        "thinker_cache_reused_tokens": 28 if turn == 2 else 0,
                        "thinker_cache_reason": "exact_text_prefix_before_audio" if turn == 2 else "fresh",
                        "thinker_prefill_ms": 125.0,
                    },
                },
            },
        }

    events = [
        {
            "event_type": "user_speech_end",
            "turn_index": 1,
            "timestamp_ns": 900_000_000,
        },
        {
            "event_type": "response.created",
            "timestamp_ns": 1_000_000_000,
            "details": {"response_id": "r1"},
        },
        {
            "event_type": "response.audio.delta",
            "timestamp_ns": 1_125_000_000,
            "details": {"response_id": "r1"},
        },
        response_done(1, 1_250_000_000),
        {
            "event_type": "user_speech_end",
            "turn_index": 2,
            "timestamp_ns": 1_900_000_000,
        },
        {
            "event_type": "response.created",
            "timestamp_ns": 2_000_000_000,
            "details": {"response_id": "r2"},
        },
        {
            "event_type": "response.audio.delta",
            "timestamp_ns": 2_100_000_000,
            "details": {"response_id": "r2"},
        },
        response_done(2, 2_250_000_000),
        {
            "event_type": "response.output_text.delta",
            "turn_index": 2,
            "timestamp_ns": 2_200_000_000,
            "details": {"response_id": "r2", "delta": "Marco, 4829."},
        },
    ]

    summary = _summarize_response_events(events, expected_turns=2, require_cache_hit=True)

    assert summary[1]["response_text"] == "Marco, 4829."
    assert summary[1]["thinker_cache_hit"] is True
    assert summary[1]["thinker_cache_reused_tokens"] == 28
    assert summary[0]["first_output_audio_ms"] == 125.0
    assert summary[0]["response_done_ms"] == 250.0
    assert summary[1]["first_output_audio_ms"] == 100.0
    assert summary[1]["user_end_to_first_audio_ms"] == 200.0
    second_done = next(
        event for event in events
        if event["event_type"] == "response.done" and event["details"]["response_id"] == "r2"
    )
    second_done["details"]["runtime_metrics"]["thinker"]["thinker_cache_hit"] = False
    with pytest.raises(RuntimeError, match="second turn did not reuse"):
        _summarize_response_events(events, expected_turns=2, require_cache_hit=True)
