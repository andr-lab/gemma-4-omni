from __future__ import annotations

import asyncio
import base64
import json
import queue
import random
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from duplex_kernel.listener import GemmaListener
from duplex_kernel.action_conditioning import action_control_marker_text
from duplex_kernel.engine import _estimate_talker_frame_budget
from duplex_kernel.response_talk_context import ResponseTalkContextPool
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM,
    CONTROLLER_GUIDED_SAY_V1,
    format_audio_response_v2_target,
)
import duplex_kernel.server as server_module
from duplex_kernel.server import (
    _CausalUserSpeechTracker,
    _ResponseAudioAdmissionState,
    _SchedulerEventBuffer,
    _attach_native_control6_trace_diagnostic,
    _native_control6_event_annotations,
    _native_control6_selection_fields,
    _yield_admission_rejection,
    create_app,
)
from duplex_kernel.session import DuplexSession, SessionManager
from duplex_kernel.scheduler import VirtualFrameClock
from duplex_kernel.turn_policy import ControlAction, NoOpPolicy, PolicyDecision, ProvisionalFramePolicy

FRAME_SAMPLES = 1280
MAX_REALTIME_PENDING_SAMPLES = server_module.USER_SAMPLE_RATE * 30


def test_realtime_inference_error_has_terminal_response_done_event() -> None:
    event = server_module._realtime_inference_failure_done_event(
        "resp_example",
        {"generation_trace": {"terminal_reason": "decoder_error"}},
    )

    assert event == {
        "type": "response.done",
        "response_id": "resp_example",
        "status": "failed",
        "reason": "inference_error",
        "runtime_metrics": {"generation_trace": {"terminal_reason": "decoder_error"}},
    }


def test_auto_response_disabled_event_preserves_candidate_and_fallback_has_no_token() -> None:
    diagnostics = {
        "native_control6_raw_row_index": 3,
        "native_control6_raw_token_id": 256004,
        "native_control6_raw_token": "<unused92>",
        "native_control6_selected_row_index": 2,
        "native_control6_selected_token_id": 256003,
        "native_control6_selected_token": "<unused91>",
    }
    decision = SimpleNamespace(action=ControlAction.CONTINUE, diagnostics=diagnostics)

    rejected = _native_control6_event_annotations(
        decision,
        head_scored=True,
        accepted=False,
    )
    rejected_event = {
        "type": "duplex_control6.decision",
        "accepted": False,
        "rejection_reason": "auto_response_disabled",
        **rejected,
    }
    assert rejected_event["native_control6_raw_token"] == "<unused92>"
    assert rejected_event["native_control6_selected_token"] == "<unused91>"
    assert rejected_event["gate_applied_action"] is None
    assert rejected_event["gate_application_scope"] == "control_plane_acceptance_not_audible_output"
    assert rejected_event["rejection_reason"] == "auto_response_disabled"
    trace_selection = _native_control6_selection_fields(decision, head_scored=True)
    assert trace_selection["native_control6_raw_row_index"] == 3
    assert trace_selection["native_control6_raw_token_id"] == 256004
    assert trace_selection["native_control6_selected_row_index"] == 2
    assert trace_selection["native_control6_selected_token_id"] == 256003
    assert trace_selection["native_control6_token_selection_scope"] == (
        "six_projected_rows_collapsed_to_physical_actions_not_full_vocabulary_sample"
    )

    accepted = _native_control6_event_annotations(
        decision,
        head_scored=True,
        accepted=True,
    )
    assert accepted["gate_applied_action"] == "CONTINUE"
    assert accepted["gate_application_scope"] == "control_plane_acceptance_not_audible_output"

    fallback = _native_control6_event_annotations(
        SimpleNamespace(action=ControlAction.LISTEN, diagnostics={"no_input_tick": True}),
        head_scored=False,
        accepted=True,
    )
    assert fallback["gate_applied_action"] == "LISTEN"
    assert not any("raw_token" in key or "row_index" in key for key in fallback)
    assert _native_control6_selection_fields(
        SimpleNamespace(diagnostics={"native_control6_raw_token_id": 256004}),
        head_scored=False,
    ) == {}


def test_native_control6_trace_diagnostic_uses_scored_flag_and_skips_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "_NATIVE_CONTROL6_TRACE_ENABLED", True)
    arguments = {
        "native_listener": SimpleNamespace(),
        "audio_soft": torch.ones((1, 4, 3)),
        "q4_output": torch.zeros((1, 4, 3)),
        "prior_health_result": {"checked": True, "status": "healthy"},
    }
    scored = {
        "native_control6_head_scored": True,
        "native_hidden_rms": 0.0,
        "native_hidden_maxabs": 0.0,
    }
    _attach_native_control6_trace_diagnostic(scored, **arguments)
    assert scored["native_control6_output_guard_status"] == "healthy"
    assert scored["native_control6_input_rows"]["shape"] == [4, 3]

    fallback = {
        "native_control6_head_scored": False,
        "native_hidden_rms": 0.0,
        "native_hidden_maxabs": 0.0,
    }
    original = dict(fallback)
    _attach_native_control6_trace_diagnostic(fallback, **arguments)
    assert fallback == original


class _AcceleratedFrameClock(VirtualFrameClock):
    """Keep 80 ms logical frames while shortening CPU-only test wall time."""

    def wait_until(self, deadline_ns: int) -> None:
        time.sleep(0.001)
        super().wait_until(deadline_ns)


class _AcceleratedSchedulerSessionManager(SessionManager):
    def create_session(self, **kwargs):
        session = super().create_session(**kwargs)
        session.scheduler.clock = _AcceleratedFrameClock(period_ms=80)
        return session


class _Backend:
    def __init__(
        self,
        hidden_size: int = 4,
        *,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.started = started
        self.release = release

    def encode(self, _pcm, _reference, state):
        if self.started is not None:
            self.started.set()
            if self.release is None or not self.release.wait(timeout=3.0):
                raise TimeoutError("test listener worker was not released")
        return np.full((1, self.hidden_size), state.next_frame_id + 1, dtype=np.float32)


class _Tokenizer:
    def __init__(self) -> None:
        self.seen_messages: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, **_kwargs):
        self.seen_messages.append([dict(message) for message in messages])
        return " | ".join(str(message.get("content", "")) for message in messages)

    def __call__(self, _text, **_kwargs):
        return {"input_ids": np.array([[1]], dtype=np.int64)}


class _QwenTextTokenizer:
    """Character-level framed tokenizer for controller-guided live SAY tests."""

    def __init__(self):
        self.backend_tokenizer = SimpleNamespace(
            pre_tokenizer=SimpleNamespace(
                pre_tokenize_str=lambda text: [
                    (char, (index, index + 1)) for index, char in enumerate(text)
                ],
            ),
        )

    def __call__(self, text, *, return_offsets_mapping=False):
        prefix = "<|im_start|>assistant\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        assert text.startswith(prefix) and text.endswith(suffix)
        raw = text[len(prefix):-len(suffix)]
        raw_ids = [1000 + ord(char) for char in raw]
        raw_offsets = [
            (len(prefix) + index, len(prefix) + index + 1)
            for index in range(len(raw))
        ]
        ids = [0, 1, 2] + raw_ids + [20, 21, 22, 23, 24]
        offsets = [(0, 0)] * 3 + raw_offsets + [(0, 0)] * 5
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(token_id) - 1000) for token_id in ids)


class _QwenTextProcessor:
    def __init__(self):
        self.tokenizer = _QwenTextTokenizer()

    def __call__(self, *, text, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([self.tokenizer(text)["input_ids"]])}


class _Engine:
    loaded = True
    enforce_production_listener = False

    def __init__(
        self,
        *,
        hidden_size: int = 4,
        block_transcription: bool = False,
        block_listener: bool = False,
        block_talk: bool = False,
    ) -> None:
        self.tokenizer = _Tokenizer()
        self.listener_started = threading.Event()
        self.release_listener = threading.Event()
        if not block_listener:
            self.release_listener.set()
        backend = _Backend(
            hidden_size,
            started=self.listener_started if block_listener else None,
            release=self.release_listener if block_listener else None,
        )
        self.listener = GemmaListener(backend=backend)
        self.default_turn_policy_factory = NoOpPolicy
        self.transcriptions: list[np.ndarray] = []
        self.transcription_started = threading.Event()
        self.release_transcription = threading.Event()
        if not block_transcription:
            self.release_transcription.set()
        self.think_calls: list[str] = []
        self.think_max_new_tokens: list[int] = []
        self.think_min_new_tokens: list[int] = []
        self.talk_calls: list[tuple[str, object]] = []
        self.think_text_override: str | None = None
        self.think_text_chunks: list[str] | None = None
        self.talk_started = threading.Event()
        self.release_talk = threading.Event()
        if not block_talk:
            self.release_talk.set()

    def transcribe_audio(self, pcm):
        self.transcriptions.append(np.array(pcm, dtype=np.float32, copy=True))
        self.transcription_started.set()
        if not self.release_transcription.wait(timeout=3.0):
            raise TimeoutError("test ASR worker was not released")
        return {"text": f"recognized turn {len(self.transcriptions)}"}

    def think(self, _ids, *, max_new_tokens=64, min_new_tokens=0, on_text_chunk=None, on_token_id=None, **_kwargs):
        self.think_calls.append("think")
        self.think_max_new_tokens.append(int(max_new_tokens))
        self.think_min_new_tokens.append(int(min_new_tokens))
        text = self.think_text_override or f"reply {len(self.think_calls)}"
        generated_ids = [101 + 2 * (len(self.think_calls) - 1), 102 + 2 * (len(self.think_calls) - 1)]
        if on_token_id is not None:
            for token_id in generated_ids:
                on_token_id(token_id)
        if on_text_chunk is not None:
            for chunk in self.think_text_chunks or [text]:
                on_text_chunk(chunk, None, 1)
        return {
            "text": text, "hidden": np.zeros((1, 1, 4), dtype=np.float32),
            "prompt_len": 1, "gen_ids": generated_ids,
        }

    def v9_prefix(self, hidden, prompt_len):
        """Provide the engine prefix contract used by the realtime response path."""
        assert prompt_len == 1
        assert tuple(np.asarray(hidden).shape) == (1, 1, 4)
        return torch.zeros((1, 2, 2048), dtype=torch.float32)

    def talk(self, text, h_pfx, *, on_audio_chunk=None, **_kwargs):
        self.talk_calls.append((text, h_pfx))
        wav = np.full(1920, 0.1, dtype=np.float32)
        if on_audio_chunk is not None:
            on_audio_chunk(wav)
        self.talk_started.set()
        if not self.release_talk.wait(timeout=3.0):
            raise TimeoutError("test talk worker was not released")
        return wav




class _V9StreamStub:
    def __init__(
        self,
        *,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.projector = SimpleNamespace(
            audio_in=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(1)))
        )
        self.raw_dim = 4
        self.max_seq_len = 64
        self.calls = 0
        self.started = started
        self.release = release

    def new_state(self, *, device, dtype, **_kwargs):
        return SimpleNamespace(device=device, dtype=dtype, frames_processed=0)

    def step(self, _hidden, _anchor, _state, *, commit=False):
        assert commit is False
        self.calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(timeout=3.0):
            raise TimeoutError("test V9 conditioning step was not released")
        return torch.full((1, 2048), float(self.calls), dtype=torch.float32)


# The runtime records its test checkpoint so deterministic CPU tests need no model file.
class _TurnHeadRuntimeStub:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.loaded = False
        self.has_thinker_hidden = False
        self.hidden_calls: list[torch.Tensor] = []
        self.predictions: list[str] = []
        self.decisions = 0

    def load(self, checkpoint: str) -> bool:
        self._checkpoint = checkpoint
        self.loaded = checkpoint in {"stub", "always-speak"}
        return self.loaded

    def set_thinker_hidden(self, hidden) -> None:
        tensor = torch.as_tensor(hidden).detach().cpu().clone()
        self.hidden_calls.append(tensor)
        self.has_thinker_hidden = tensor.shape[-1] == 2048

    def feed_pcm(self, _pcm, *, finalize_frame=False) -> int:
        assert finalize_frame is True
        return 8

    def decide(self):
        self.decisions += 1
        if self.loaded and self._checkpoint == "always-speak":
            name = "SPEAK"
        elif self.loaded:
            name = {1: "SPEAK", 2: "BARGE_IN"}.get(self.decisions, "USER_BACKCHANNEL")
        else:
            name = "USER_BACKCHANNEL"
        labels = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]
        self.predictions.append(name)
        probs = [0.001] * len(labels)
        probs[labels.index(name)] = 0.996
        return {"pred_name": name, "probs": probs, "active": True}

    def reset(self) -> None:
        self.has_thinker_hidden = False
        self.hidden_calls.clear()


def _send_pcm(ws, frames: int = 1) -> None:
    pcm = np.full(frames * FRAME_SAMPLES, 4096, dtype="<i2").tobytes()
    ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})


def _send_pcm_level(ws, value: int) -> None:
    pcm = np.full(FRAME_SAMPLES, value, dtype="<i2").tobytes()
    ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})


def _receive_until(ws, event_type: str, *, limit: int = 30) -> list[dict[str, object]]:
    events = []
    for _ in range(limit):
        event = ws.receive_json()
        events.append(event)
        if event.get("type") == event_type:
            return events
    raise AssertionError(f"evento {event_type!r} non ricevuto; ottenuti: {events!r}")


def _receive_until_any(ws, event_types: set[str], *, limit: int = 30) -> list[dict[str, object]]:
    events = []
    for _ in range(limit):
        event = ws.receive_json()
        events.append(event)
        if event.get("type") in event_types:
            return events
    raise AssertionError(f"nessuno degli eventi {event_types!r} ricevuto; ottenuti: {events!r}")


def _receive_action_decision(ws, action: str, *, limit: int = 30) -> dict[str, object]:
    events = []
    for _ in range(limit):
        event = ws.receive_json()
        events.append(event)
        if event.get("type") == "duplex_control6.decision" and event.get("action") == action:
            return event
    raise AssertionError(f"decisione {action!r} non ricevuta; ottenuti: {events!r}")


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_session_created_records_canonical_listener_prime_cost() -> None:
    engine = _Engine()
    prime_calls: list[float] = []

    def prime_steady_state(_state) -> None:
        prime_calls.append(time.perf_counter())
        time.sleep(0.005)

    engine.listener = SimpleNamespace(prime_steady_state=prime_steady_state)
    client = TestClient(create_app(engine, SessionManager()))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert len(prime_calls) == 1
        listener_metrics = created["runtime_metrics"]["listener"]
        assert listener_metrics["canonical_prime_calls"] == 1
        assert listener_metrics["canonical_prime_succeeded"] is True
        assert listener_metrics["canonical_prime_ms"] >= 4.0
        ws.send_json({"type": "session.close"})


def test_root_route_serves_the_same_origin_realtime_ui() -> None:
    with TestClient(create_app(_Engine(), SessionManager())) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Gemma Omni" in response.text
    assert "input_audio_transcription" in response.text


def test_native_partial_asr_is_opt_in_and_emits_revisioned_snapshots(monkeypatch) -> None:
    monkeypatch.setenv(server_module.NATIVE_PARTIAL_ASR_ENV, "1")
    engine = _Engine()
    sessions = _AcceleratedSchedulerSessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["LISTEN"] * 100,
                "input_audio_transcription": True,
                "input_audio_buffer_acknowledgements": True,
            },
        })
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["input_audio_transcription_enabled"] is True

        def send_frame_and_wait_ack(frame_id: int, level: int) -> None:
            _send_pcm_level(ws, level)
            for _ in range(80):
                event = ws.receive_json()
                if (
                    event.get("type") == "input_audio_buffer.consumed"
                    and event.get("frame_id") == frame_id
                ):
                    return
                assert event.get("type") != "error", event
            raise AssertionError(f"input frame {frame_id} was not acknowledged")

        for frame_id in range(13):
            send_frame_and_wait_ack(frame_id, 4096)
        partial = _receive_until(ws, "input_audio_transcription.partial", limit=80)[-1]
        assert partial["utterance_id"]
        assert partial["revision"] == 1
        assert partial["transcript"] == "recognized turn 1"
        assert partial["first_frame_id"] == 0
        assert partial["last_frame_id"] >= 12
        assert "_transcript_source" not in partial

        for frame_id in range(13, 23):
            send_frame_and_wait_ack(frame_id, 0)
        completed = _receive_until(ws, "input_audio_transcription.completed", limit=80)[-1]
        assert completed["utterance_id"] == partial["utterance_id"]
        assert completed["revision"] == 2
        assert completed["transcript"] == "recognized turn 2"
        assert completed["first_frame_id"] == partial["first_frame_id"]
        assert completed["last_frame_id"] == partial["last_frame_id"]
        assert "_transcript_source" not in completed
        ws.send_json({"type": "session.close"})


def test_native_partial_asr_session_opt_in_is_off_unless_gpu_gate_is_enabled(monkeypatch) -> None:
    monkeypatch.delenv(server_module.NATIVE_PARTIAL_ASR_ENV, raising=False)
    client = TestClient(create_app(_Engine(), SessionManager()))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {"input_audio_transcription": True},
        })
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["input_audio_transcription_enabled"] is False
        ws.send_json({
            "type": "session.update",
            "session": {"input_audio_transcription": "true"},
        })
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == "INVALID_SESSION_UPDATE"
        ws.send_json({"type": "session.close"})


def test_native_partial_asr_keeps_latest_snapshot_while_backchannel_is_active(monkeypatch) -> None:
    monkeypatch.setenv(server_module.NATIVE_PARTIAL_ASR_ENV, "1")
    engine = _Engine(hidden_size=2048, block_talk=True)
    sessions = _AcceleratedSchedulerSessionManager()
    client = TestClient(create_app(engine, sessions))

    try:
        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({
                "type": "session.update",
                "session": {
                    "turn_policy": "scripted",
                    "scripted_actions": ["BACKCHANNEL"] + ["LISTEN"] * 100,
                    "input_audio_transcription": True,
                    "input_audio_buffer_acknowledgements": True,
                },
            })
            assert ws.receive_json()["type"] == "session.updated"

            _send_pcm_level(ws, 4096)
            first_events = _receive_until(ws, "response.created", limit=80)
            created = first_events[-1]
            assert created["action"] == "BACKCHANNEL"
            assert engine.talk_started.wait(timeout=3.0), (
                f"Thinker/Talker did not start after {created!r}; "
                f"think_calls={engine.think_calls!r}, talk_calls={engine.talk_calls!r}, "
                f"transcriptions={len(engine.transcriptions)}"
            )

            for frame_id in range(1, 13):
                _send_pcm_level(ws, 4096)
                for _ in range(80):
                    event = ws.receive_json()
                    if (
                        event.get("type") == "input_audio_buffer.consumed"
                        and event.get("frame_id") == frame_id
                    ):
                        break
                    assert event.get("type") != "error", event
                else:
                    raise AssertionError(f"input frame {frame_id} was not acknowledged")

            assert not engine.transcription_started.wait(timeout=0.05), (
                "native ASR must defer while the response owns Thinker/Talker resources"
            )
            engine.release_talk.set()
            response_events = _receive_until(ws, "response.done", limit=80)
            assert not any(event.get("type") == "input_audio_transcription.partial" for event in response_events)
            assert engine.transcription_started.wait(timeout=1.0), (
                "the latest native snapshot must resume after response.done"
            )

            partial_events = _receive_until(ws, "input_audio_transcription.partial", limit=80)
            partial = partial_events[-1]
            assert partial["utterance_id"]
            assert partial["revision"] == 1
            assert partial["first_frame_id"] == 0
            assert partial["last_frame_id"] >= 12
            assert "_transcript_source" not in partial

            for frame_id in range(13, 23):
                _send_pcm_level(ws, 0)
                for _ in range(80):
                    event = ws.receive_json()
                    if (
                        event.get("type") == "input_audio_buffer.consumed"
                        and event.get("frame_id") == frame_id
                    ):
                        break
                    assert event.get("type") != "error", event
                else:
                    raise AssertionError(f"quiet frame {frame_id} was not acknowledged")

            completed_events = _receive_until(ws, "input_audio_transcription.completed", limit=80)
            completed = completed_events[-1]
            assert completed["utterance_id"] == partial["utterance_id"]
            assert completed["revision"] == 2
            assert completed["first_frame_id"] == 0
            assert completed["last_frame_id"] == partial["last_frame_id"]
            assert "_transcript_source" not in completed
            ws.send_json({"type": "session.close"})
    finally:
        engine.release_talk.set()


def test_mode_b_continuous_silence_over_30s_keeps_ack_and_recent_voice() -> None:
    engine = _Engine()
    sessions = _AcceleratedSchedulerSessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["LISTEN"],
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]

        # 30 seconds of silence followed by two voiced frames must remain a
        # valid continuous Mode B stream; the retained API buffer is a rolling
        # window and still ends with the actual voice samples.
        def receive_frame_ack(expected_frame_id: int) -> None:
            for _ in range(80):
                event = ws.receive_json()
                assert event.get("type") != "error", event
                if (
                    event.get("type") == "input_audio_buffer.consumed"
                    and event.get("frame_id") == expected_frame_id
                ):
                    return
            raise AssertionError(f"input frame {expected_frame_id} was not acknowledged")

        for frame_id in range(377):
            _send_pcm_level(ws, 0 if frame_id < 375 else 4096)
            receive_frame_ack(frame_id)

        with session._pending_audio_lock:
            pending = session.pending_audio.copy()
        assert pending.size == MAX_REALTIME_PENDING_SAMPLES
        assert np.all(pending[-2 * FRAME_SAMPLES:] == np.float32(4096 / 32768))
        ws.send_json({"type": "session.close"})


def test_mode_a_append_larger_than_30s_still_fails_closed() -> None:
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        pcm = np.zeros(MAX_REALTIME_PENDING_SAMPLES + FRAME_SAMPLES, dtype="<i2").tobytes()
        ws.send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        })
        events = _receive_until(ws, "error")
        assert events[-1]["error"]["code"] == "AUDIO_LIMIT"
        assert sessions.all()[0].pending_audio.size == 0
        ws.send_json({"type": "session.close"})


def test_mode_b_single_append_larger_than_30s_still_fails_closed() -> None:
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["LISTEN"],
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        pcm = np.zeros(MAX_REALTIME_PENDING_SAMPLES + FRAME_SAMPLES, dtype="<i2").tobytes()
        ws.send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        })
        events = _receive_until(ws, "error")
        assert events[-1]["error"]["code"] == "AUDIO_LIMIT"
        assert sessions.all()[0].pending_audio.size == 0
        ws.send_json({"type": "session.close"})


def test_opt_in_ingress_ack_tracks_listener_consumption_with_v9_conditioning() -> None:
    engine = _Engine()
    engine.v9_stream = _V9StreamStub()
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
        session = sessions.all()[0]
        received: list[dict[str, object]] = []
        receive_error: list[BaseException] = []

        def receive_two_acks() -> None:
            try:
                received.extend([ws.receive_json(), ws.receive_json()])
            except BaseException as exc:
                receive_error.append(exc)

        reader = threading.Thread(target=receive_two_acks, name="test-ingress-ack-reader", daemon=True)
        reader.start()
        _send_pcm(ws, frames=2)
        reader.join(timeout=2.0)
        if reader.is_alive():
            ws.send_json({"type": "session.close"})
            reader.join(timeout=1.0)

        assert not reader.is_alive(), "listener did not acknowledge opt-in ingress frames"
        assert not receive_error, f"ACK reader failed: {receive_error!r}"
        assert [event["type"] for event in received] == [
            "input_audio_buffer.consumed",
            "input_audio_buffer.consumed",
        ]
        assert [event["frame_id"] for event in received] == [0, 1]
        assert _wait_for(lambda: session.v9_state.frames_processed == 2)
        assert session.v9_state.frames_processed == 2
        ws.send_json({"type": "session.close"})


def test_opt_in_conditioning_ack_is_separate_and_follows_v9_publication(monkeypatch) -> None:
    engine = _Engine()
    v9_started = threading.Event()
    release_v9 = threading.Event()
    engine.v9_stream = _V9StreamStub(started=v9_started, release=release_v9)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))
    conditioning_events: list[dict[str, object]] = []
    original_put = server_module._SchedulerEventBuffer.put_nowait

    def observe_conditioning_event(self, event):
        if event.get("type") == "input_audio_buffer.conditioning_available":
            conditioning_events.append(dict(event))
        return original_put(self, event)

    monkeypatch.setattr(server_module._SchedulerEventBuffer, "put_nowait", observe_conditioning_event)

    try:
        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({
                "type": "session.update",
                "session": {
                    "turn_policy": "none",
                    "input_audio_buffer_acknowledgements": True,
                    "input_audio_buffer_conditioning_acknowledgements": True,
                },
            })
            updated = ws.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["input_audio_buffer_conditioning_acknowledgements"] is True
            _send_pcm(ws)

            ack_result: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)

            def receive_consumed_ack() -> None:
                ack_result.put(_receive_until(ws, "input_audio_buffer.consumed")[-1])

            reader = threading.Thread(target=receive_consumed_ack, name="test-conditioning-ack-reader", daemon=True)
            reader.start()
            assert v9_started.wait(timeout=1.0), "V9 step did not start"
            try:
                ack = ack_result.get(timeout=1.0)
                assert ack["type"] == "input_audio_buffer.consumed"
                assert ack["frame_id"] == 0
                assert conditioning_events == [], "conditioning readiness preceded V9 publication"
            finally:
                release_v9.set()
            reader.join(timeout=1.0)
            assert _wait_for(lambda: bool(conditioning_events), timeout=2.0)
            event = conditioning_events[0]
            assert event["frame_id"] == 0
            assert event["listener_output_frame_id"] == 0
            assert event["conditioning_frame_available"] is True
            assert event["conditioning_frame_id"] == 0
            ws.send_json({"type": "session.close"})
    finally:
        release_v9.set()


def test_opt_in_ingress_ack_uses_input_frame_when_listener_has_no_output() -> None:
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))
    original_step = engine.listener.step
    calls = 0

    def warmup_without_output(pcm, reference, state):
        nonlocal calls
        calls += 1
        output = original_step(pcm, reference, state)
        # The first consumed frame advances the frontend but has no published
        # feature/query output yet. Its input-frame acknowledgement must still
        # be sent exactly once, including the valid frame ID zero.
        return None if calls == 1 else output

    engine.listener.step = warmup_without_output

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["LISTEN"],
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"

        acknowledgements: list[int] = []
        for expected in (0, 1):
            _send_pcm(ws)
            for _ in range(40):
                event = ws.receive_json()
                if event.get("type") == "error":
                    raise AssertionError(f"listener warmup should not fail the session: {event!r}")
                if event.get("type") == "input_audio_buffer.consumed":
                    acknowledgements.append(event.get("frame_id"))
                    if event.get("frame_id") == expected:
                        break
            else:
                raise AssertionError(f"input frame {expected} was not acknowledged")

        assert acknowledgements == [0, 1]
        ws.send_json({"type": "session.close"})


def test_shared_engine_owner_stops_only_matching_session_atomically() -> None:
    owner = server_module._EngineSynthesisOwner()
    owner.claim("session-a", "response-a")
    stop_entered = threading.Event()
    release_stop = threading.Event()
    next_owner_claimed = threading.Event()
    stop_calls: list[str] = []

    def stop_engine() -> None:
        stop_calls.append("session-a")
        stop_entered.set()
        assert release_stop.wait(timeout=2.0)

    stop_thread = threading.Thread(
        target=lambda: owner.stop_if_owned("session-a", "response-a", stop_engine),
        name="test-atomic-engine-stop",
    )
    stop_thread.start()
    assert stop_entered.wait(timeout=1.0)

    def transfer_owner() -> None:
        owner.release("session-a", "response-a")
        owner.claim("session-b", "response-b")
        next_owner_claimed.set()

    transfer_thread = threading.Thread(target=transfer_owner, name="test-engine-owner-transfer")
    transfer_thread.start()
    assert not next_owner_claimed.wait(timeout=0.05), "new GPU owner crossed an in-flight stop operation"
    release_stop.set()
    stop_thread.join(timeout=1.0)
    transfer_thread.join(timeout=1.0)

    assert not stop_thread.is_alive()
    assert not transfer_thread.is_alive()
    assert stop_calls == ["session-a"]
    assert owner.is_owned("session-b", "response-b")
    assert owner.stop_if_owned("session-a", "response-a", lambda: stop_calls.append("stale")) is False
    assert stop_calls == ["session-a"]


def test_scheduler_event_buffer_preserves_cancellation_order_across_threads() -> None:
    loop = asyncio.new_event_loop()
    events = _SchedulerEventBuffer(loop, maxsize=2)
    opened = {"type": "speaker.segment.opened", "_generation_fence": 4}
    cancelled = {"type": "response.cancelled", "_generation_fence": 5}
    try:
        producer = threading.Thread(
            target=lambda: (events.put_nowait(opened), events.put_nowait(cancelled)),
            name="test-scheduler-event-producer",
        )
        producer.start()
        producer.join(timeout=1.0)
        assert not producer.is_alive()
        assert events.qsize == events.maxsize == 2
        assert loop.run_until_complete(events.get()) == opened
        assert loop.run_until_complete(events.get()) == cancelled
        events.close()
        assert loop.run_until_complete(events.get()) is None
    finally:
        loop.close()


def test_scheduler_event_buffer_saturation_is_terminal_and_bounded() -> None:
    loop = asyncio.new_event_loop()
    events = _SchedulerEventBuffer(loop, maxsize=1)
    try:
        assert events.put_nowait({"type": "response.created"})
        assert not events.put_nowait({"type": "response.cancelled"})
        assert events.overflowed
        assert events.qsize == 1
        assert loop.run_until_complete(events.get()) is events.OVERFLOW
        assert events.qsize == 0
    finally:
        events.close()
        loop.close()


def test_audio_ingress_overload_fails_closed_without_playing_stale_frames() -> None:
    engine = _Engine(block_listener=True)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        _send_pcm(ws)
        assert engine.listener_started.wait(timeout=1.0)
        try:
            # Hold the listener on one in-flight frame, fill all three queued
            # slots, then offer one more frame. Old behavior acknowledges the
            # barrier while silently building an ever older mic queue.
            _send_pcm(ws, frames=3)
            _send_pcm(ws)
            ws.send_json({"type": "session.update", "session": {}})
            overload = ws.receive_json()
            assert overload["type"] == "error"
            assert overload["error"]["code"] == "INPUT_AUDIO_OVERLOAD"
            assert session.audio_ingress.stats()["dropped_input_frames"] == 1
            with pytest.raises(WebSocketDisconnect) as disconnected:
                ws.receive_json()
            assert disconnected.value.code == 1013
        finally:
            engine.release_listener.set()


def test_committed_audio_is_transcribed_off_scheduler_and_history_has_real_user_text() -> None:
    engine = _Engine(block_transcription=True)
    engine.v9_prefix = lambda _hidden, _prompt_len: torch.zeros((1, 1, 2048))
    thinker_started = threading.Event()
    release_thinker = threading.Event()
    original_think = engine.think

    def hold_thinker(*args, **kwargs):
        result = original_think(*args, **kwargs)
        thinker_started.set()
        if not release_thinker.wait(timeout=3.0):
            raise TimeoutError("test thinker worker was not released")
        return result

    engine.think = hold_thinker
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        _send_pcm(ws, frames=2)
        ws.send_json({"type": "input_audio_buffer.commit"})
        _receive_until(ws, "response.created")
        assert engine.transcription_started.wait(timeout=1.0)

        # ASR is a turn-level worker, not a scheduler callback: mic frames
        # continue through the listener while transcription is deliberately held.
        assert _wait_for(lambda: session.listener_state.listener_input_frames >= 2)
        _send_pcm(ws)
        assert _wait_for(lambda: session.listener_state.listener_input_frames >= 3)
        assert session.scheduler.is_running
        assert engine.think_calls == []

        engine.release_transcription.set()
        assert thinker_started.wait(timeout=1.0)
        frames_before = session.listener_state.listener_input_frames
        _send_pcm(ws)
        assert _wait_for(lambda: session.listener_state.listener_input_frames >= frames_before + 1)
        assert session.scheduler.is_running
        release_thinker.set()
        events = _receive_until(ws, "response.done")
        assert any(event.get("type") == "response.audio.delta" for event in events)
        assert len(engine.transcriptions) == 1
        assert engine.transcriptions[0].shape == (2 * FRAME_SAMPLES,)
        assert engine.think_calls == ["think"]
        assert engine.tokenizer.seen_messages[-1][-1]["content"] == "recognized turn 1"
        assert session.history[-2:] == [
            {"role": "user", "content": "recognized turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]
        assert session.listener_state.listener_input_frames >= 3
        ws.send_json({"type": "session.close"})


def test_scripted_speak_action_starts_transcription_and_response_worker() -> None:
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["SPEAK"],
                "auto_response": True,
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        _send_pcm(ws)
        first_speak = _receive_action_decision(ws, "SPEAK")
        assert first_speak["frame_id"] == 0
        assert first_speak["accepted"] is False
        assert first_speak["rejection_reason"] == "speech_endpoint_pending"
        # Continuous quiet is part of the live input timeline. The configured
        # default is ten 80 ms frames (800 ms) before SPEAK is dispatched.
        events: list[dict[str, object]] = []
        for expected_frame_id in range(1, 11):
            _send_pcm_level(ws, 0)
            ack_events = _receive_until(ws, "input_audio_buffer.consumed")
            # The response can finish before the final input-consumed ACK is
            # delivered. Keep every event consumed while pacing the input.
            events.extend(ack_events)
            assert ack_events[-1]["frame_id"] == expected_frame_id
            assert ack_events[-1]["server_listener_queue_ms"] >= 0
            assert ack_events[-1]["server_listener_step_ms"] >= 0
            assert ack_events[-1]["server_ack_queue_ms"] >= 0
        def receive_event_before(deadline: float) -> dict[str, object]:
            async def receive_one():
                return await asyncio.wait_for(
                    ws._send_rx.receive(),
                    timeout=max(0.001, deadline - time.monotonic()),
                )

            message = ws.portal.call(receive_one)
            ws._raise_on_close(message)
            return json.loads(message["text"])

        deadline = time.monotonic() + 5.0
        while not any(
            event.get("type") == "duplex_control6.decision"
            and event.get("frame_id") == 10
            and event.get("action") == "SPEAK"
            for event in events
        ):
            try:
                events.append(receive_event_before(deadline))
            except TimeoutError as exc:
                raise AssertionError(
                    f"SPEAK endpoint decision was not emitted; observed {[e.get('type') for e in events]}"
                ) from exc
        if not any(event.get("type") == "response.done" for event in events):
            deadline = time.monotonic() + 5.0
            while not any(
                event.get("type") in {"response.done", "response.failed"}
                for event in events
            ):
                try:
                    events.append(receive_event_before(deadline))
                except TimeoutError as exc:
                    raise AssertionError(
                        "response did not terminate; decisions="
                        f"{[e for e in events if e.get('type') == 'duplex_control6.decision']} "
                        f"responses={[e for e in events if str(e.get('type', '')).startswith('response.')]}"
                    ) from exc
        assert any(event.get("type") == "response.done" for event in events), events

        assert engine.transcription_started.is_set()
        assert len(engine.transcriptions) == 1
        assert engine.transcriptions[0].shape == (11 * FRAME_SAMPLES,)
        created = next(event for event in events if event.get("type") == "response.created")
        assert created["action"] == "SPEAK"
        assert created["effective_action"] == "SPEAK"
        assert created["intent_source_frame_id"] == 0
        assert created["intent_endpoint_frame_id"] == 10
        assert created["intent_source"] == "turn_policy"
        raw_endpoint_decision = next(
            event for event in events
            if event.get("type") == "duplex_control6.decision"
            and event.get("frame_id") == 10
            and event.get("action") == "SPEAK"
        )
        assert raw_endpoint_decision["accepted"] is False
        assert raw_endpoint_decision["rejection_reason"] == "deferred_speak_dispatched"
        assert any(event.get("type") == "response.audio.delta" for event in events)
        action_system = next(
            message["content"] for message in engine.tokenizer.seen_messages[-1]
            if message.get("role") == "system" and "Current turn action: SPEAK." in message.get("content", "")
        )
        assert "most recent completed user utterance" in action_system
        assert engine.tokenizer.seen_messages[-1][-1] == {
            "role": "user",
            "content": "",
        }
        assert sessions.all()[0].history[-2:] == [
            {"role": "user", "content": "recognized turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]
        assert not any(
            message.get("role") == "system" and "Current turn action:" in message.get("content", "")
            for message in sessions.all()[0].thinker_ctx.messages()
        )
        ws.send_json({"type": "session.close"})


def test_auto_response_false_preserves_pending_audio_on_policy_speak() -> None:
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["SPEAK"],
                "auto_response": False,
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]

        _send_pcm(ws)
        first_speak = _receive_action_decision(ws, "SPEAK")
        assert first_speak["frame_id"] == 0

        for expected_frame_id in range(1, 11):
            _send_pcm_level(ws, 0)
            ack_events = _receive_until(ws, "input_audio_buffer.consumed")
            assert ack_events[-1]["frame_id"] == expected_frame_id

        # Policy endpoint decided SPEAK, but because auto_response is False,
        # it must NOT drain session.pending_audio.
        with session._pending_audio_lock:
            pending_samples = session.pending_audio.size
        assert pending_samples == 11 * FRAME_SAMPLES, f"Expected 11 frames in pending_audio, got {pending_samples}"

        # Now send manual commit: it must transcribe all 11 frames and complete response.
        ws.send_json({"type": "input_audio_buffer.commit"})
        events = _receive_until(ws, "response.done")
        assert any(event.get("type") == "response.audio.delta" for event in events)
        assert len(engine.transcriptions) == 1
        assert engine.transcriptions[0].shape == (11 * FRAME_SAMPLES,)
        ws.send_json({"type": "session.close"})


@pytest.mark.parametrize(
    ("wake_action", "response_action", "cancel_during_talker", "endpoint_policy_action", "adapter_enabled", "talker_error"),
    [
        ("SPEAK", "SPEAK", False, "CONTINUE", False, False),
        ("SPEAK", "SPEAK", True, "CONTINUE", False, False),
        ("SPEAK", "BACKCHANNEL", False, "CONTINUE", False, False),
        ("SPEAK", "BACKCHANNEL", True, "CONTINUE", False, False),
        ("SPEAK", "BACKCHANNEL", None, "CONTINUE", False, False),
        ("BACKCHANNEL", "SPEAK", False, "CONTINUE", False, False),
        ("BACKCHANNEL", "SPEAK", False, "YIELD", False, False),
        ("SPEAK", "SPEAK", None, "CONTINUE", True, False),
        ("SPEAK", "SPEAK", True, "CONTINUE", True, False),
        ("SPEAK", "SPEAK", None, "CONTINUE", True, True),
    ],
)
def test_realtime_v2_talker_uses_validated_say_prefix_and_keeps_listener_ticks(
    monkeypatch: pytest.MonkeyPatch,
    wake_action: str,
    response_action: str,
    cancel_during_talker: bool | None,
    endpoint_policy_action: str,
    adapter_enabled: bool,
    talker_error: bool,
) -> None:
    processor_audio_snapshots: list[np.ndarray] = []
    cpu_archive_case = (
        wake_action == "SPEAK"
        and response_action == "SPEAK"
        and cancel_during_talker is False
        and endpoint_policy_action == "CONTINUE"
    )
    monkeypatch.delenv("DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU", raising=False)
    if cpu_archive_case:
        monkeypatch.setenv("DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU", "1")

    class _AudioProcessor:
        audio_token = "<|audio|>"

        def __call__(self, **kwargs):
            audio = kwargs.get("audio")
            if isinstance(audio, (list, tuple)) and audio:
                processor_audio_snapshots.append(np.asarray(audio[0], dtype=np.float32).copy())
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "input_features": torch.ones((1, 4, 8)),
                "input_features_mask": torch.ones((1, 4), dtype=torch.long),
            }

    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", AUDIO_RESPONSE_PROTOCOL_V2)
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "0")
    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    engine = _Engine(hidden_size=4, block_talk=True)
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.v9_stream = _V9StreamStub()
    engine.device = "cpu"
    engine.dtype = torch.float32
    if adapter_enabled:
        engine.frame_residual_adapter_enabled = True
        engine.frame_residual_adapter = SimpleNamespace(
            architecture_id="response_hidden_q1_query_film_v2",
        )
        engine.frame_residual_adapter_resolved_path = "test-frame-residual-adapter"
        engine.talker = SimpleNamespace(
            proc=_QwenTextProcessor(),
            tcfg=SimpleNamespace(hidden_size=4),
            cfg=SimpleNamespace(tts_eos_token_id=9999),
        )
        engine.response_talk_context_pool = ResponseTalkContextPool([
            SimpleNamespace(reset=lambda: None),
        ])
    engine._thinker_wrapper = SimpleNamespace(processor=_AudioProcessor())
    engine.tokenizer.audio_token = "<|audio|>"
    encoded_fragments: dict[int, str] = {}

    def encode_fragment(text, *, add_special_tokens):
        if add_special_tokens:
            return []
        action_ids = {"SPEAK": 41, "BACKCHANNEL": 42, "REPLAN": 43}
        if text in action_ids:
            return [action_ids[text]]
        token_id = 1000 + len(encoded_fragments)
        encoded_fragments[token_id] = text
        return [token_id]

    def decode_fragment(token_ids, *, skip_special_tokens):
        if skip_special_tokens and len(token_ids) == 1 and token_ids[0] in {41, 42, 43}:
            return {41: "SPEAK", 42: "BACKCHANNEL", 43: "REPLAN"}[token_ids[0]]
        if len(token_ids) == 1:
            return encoded_fragments.get(token_ids[0], "")
        return ""

    engine.tokenizer.encode = encode_fragment
    engine.tokenizer.decode = decode_fragment
    say_text = "Capito." if response_action == "BACKCHANNEL" else "Ciao, ti ascolto."
    heard_text = "" if response_action == "BACKCHANNEL" else "ho detto ciao"
    target = format_audio_response_v2_target(
        heard=heard_text,
        action=response_action,
        say=say_text,
    )
    say_start = target.index('"say":"') + len('"say":"')
    say_end = target.rfind('"')
    response_fragments = [target[:say_start], target[say_start:say_end], target[say_end:]]
    say_hidden = torch.full((1, 1, 1536), 22.0)
    engine.say_hidden_seen = None
    engine.talker_prefix_seen = None
    engine.talker_kwargs_seen = None
    engine.frame_residual_hidden_seen = None
    engine.stop_check_after_cancel = None
    engine.live_frames_seen = ()
    engine.talker_started = threading.Event()
    engine.release_talker = threading.Event()
    engine.release_talker.clear()

    def think(_ids, *, on_generated_token=None, **_kwargs):
        assert callable(on_generated_token)
        for index, (fragment, value) in enumerate(zip(response_fragments, (11.0, 22.0, 33.0), strict=True)):
            on_generated_token(index, 100 + index, fragment, torch.full((1, 1, 1536), value))
        return {
            "text": target,
            "protocol_text": target,
            "hidden": torch.full((1, 1, 1536), -99.0),
            "prompt_len": 5,
            "gen_ids": [100, 101, 102],
            "runtime_metrics": {},
        }

    def v9_prefix(hidden, prompt_len):
        assert prompt_len == 0
        engine.say_hidden_seen = hidden.detach().clone()
        return torch.full((1, 2, 2048), float(hidden[0, 0, 0]))

    def talk_from_say_prefill(text, h_pfx, **kwargs):
        assert text == say_text
        if adapter_enabled:
            assert tuple(h_pfx.shape) == (1, 0, 2048)
            engine.frame_residual_hidden_seen = kwargs.get("frame_residual_response_hidden")
            assert engine.frame_residual_hidden_seen is not None
            assert tuple(engine.frame_residual_hidden_seen.shape) == (1, len(say_text), 1536)
            assert torch.equal(
                engine.frame_residual_hidden_seen,
                torch.full((1, len(say_text), 1536), 22.0),
            )
        else:
            assert tuple(h_pfx.shape) == (1, 2, 2048)
            assert torch.equal(h_pfx, torch.full((1, 2, 2048), 22.0))
        assert "conditioning_buffer" not in kwargs
        assert "response_text_conditioning" not in kwargs
        engine.talker_prefix_seen = h_pfx.detach().clone()
        engine.talker_kwargs_seen = dict(kwargs)
        if adapter_enabled:
            assert kwargs.get("talk_context") is not None
            assert engine.response_talk_context_pool.active_count == 1
        callback = kwargs.get("on_audio_chunk")
        if callback is not None:
            callback(np.full(1920, 0.1, dtype=np.float32))
        engine.talker_started.set()
        if not engine.release_talker.wait(timeout=3.0):
            raise TimeoutError("v2 realtime talker was not released")
        if engine.cancel_during_talker is True:
            kwargs["metrics"]["cancelled"] = True
        engine.stop_check_after_cancel = kwargs["stop_check"]()
        if talker_error:
            raise RuntimeError("injected v2 frame-residual talker failure")
        return np.zeros(0, dtype=np.float32) if engine.stop_check_after_cancel else np.full(1920, 0.1, dtype=np.float32)

    engine.think = think
    engine.v9_prefix = v9_prefix
    engine.talk = talk_from_say_prefill
    engine.cancel_during_talker = cancel_during_talker
    engine._backchannel_say_phrases = ("Capito.",)
    engine.talk_stateful = lambda *_args, **_kwargs: pytest.fail(
        "v2 realtime deve usare il percorso stateless condizionato dal SAY"
    )
    archive_asr_calls: list[np.ndarray] = []
    engine.transcribe_audio = lambda pcm: archive_asr_calls.append(np.array(pcm, copy=True)) or {
        "text": "unexpected second transcription"
    }
    cpu_archive_calls: list[np.ndarray] = []
    if cpu_archive_case:
        class _ArchiveWorker:
            model_path = SimpleNamespace(is_dir=lambda: True)

            @property
            def available(self) -> bool:
                return True

            def submit(self, pcm):
                cpu_archive_calls.append(np.asarray(pcm, dtype=np.float32).copy())
                future = Future()
                future.set_result({
                    "text": "trascrizione CPU del turno reale",
                    "language": "it",
                    "duration_s": 0.16,
                    "asr_ms": 1.0,
                    "model": "faster-whisper-small",
                    "device": "cpu",
                    "compute_type": "int8",
                    "cpu_threads": 4,
                })
                return future

        monkeypatch.setattr(server_module, "CPU_AUDIO_ARCHIVE_WORKER", _ArchiveWorker())
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": (
                    [wake_action] + ["CONTINUE"] * 9 + [endpoint_policy_action] + ["CONTINUE"] * 10
                    if endpoint_policy_action == "YIELD"
                    else (
                        ["CONTINUE"] * 10 + [wake_action] + ["CONTINUE"] * 10
                        if wake_action == "SPEAK"
                        else [wake_action] + ["CONTINUE"] * 20
                    )
                ),
                "input_audio_buffer_acknowledgements": True,
                "auto_response": True,
                "instructions": "Stiamo organizzando una cena con amici.",
                "audio_history_archive_cpu": cpu_archive_case,
            },
        })
        session_updated = ws.receive_json()
        assert session_updated["type"] == "session.updated"
        if cpu_archive_case:
            assert session_updated["audio_history_archive_cpu_enabled"] is True
        observed_events: list[dict[str, object]] = []
        _send_pcm(ws)
        first_decision = None
        if wake_action == "SPEAK":
            observed_events.extend(_receive_until(ws, "input_audio_buffer.consumed"))
            for expected_frame_id in range(1, 11):
                _send_pcm_level(ws, 0)
                observed_events.extend(_receive_until(ws, "input_audio_buffer.consumed"))
            first_decision = next(
                (
                    event for event in observed_events
                    if event.get("type") == "duplex_control6.decision"
                    and event.get("action") == wake_action
                ),
                None,
            )
        else:
            for _ in range(30):
                event = ws.receive_json()
                observed_events.append(event)
                if event.get("type") == "duplex_control6.decision" and event.get("action") == wake_action:
                    first_decision = event
                    break
        if first_decision is None:
            for _ in range(30):
                event = ws.receive_json()
                observed_events.append(event)
                if event.get("type") == "duplex_control6.decision" and event.get("action") == wake_action:
                    first_decision = event
                    break
        assert first_decision is not None, f"decisione iniziale {wake_action} assente: {observed_events!r}"
        assert first_decision["frame_id"] == (10 if wake_action == "SPEAK" else 0)
        assert first_decision["accepted"] is True

        if wake_action == "BACKCHANNEL" and response_action == "SPEAK":
            observed_events.extend(_receive_until(ws, "response.action_selected", limit=60))
            session = sessions.all()[0]
            v9_state_before_deferred_done = session.v9_state
            v9_anchor_before_deferred_done = session.v9_text_anchor
            assert v9_state_before_deferred_done is not None
            observed_events.extend(_receive_until(ws, "response.done", limit=60))
            assert session.v9_state is v9_state_before_deferred_done
            assert session.v9_text_anchor is v9_anchor_before_deferred_done
            assert engine.say_hidden_seen is None, "SAY differita non deve entrare nel Talker prima dell'endpoint"
            assert not engine.talker_started.is_set()
            assert not any(event.get("type") in {"response.output_text.delta", "response.audio.delta"} for event in observed_events)
            selected = next(event for event in observed_events if event.get("type") == "response.action_selected")
            assert selected["wake_action"] == "BACKCHANNEL"
            assert selected["response_action"] == "SPEAK"
            assert selected["deferred_until_endpoint"] is True
            done = next(event for event in observed_events if event.get("type") == "response.done")
            assert done["status"] == "deferred_until_endpoint"
            assert not any(event.get("type") == "input_audio_transcription.completed" for event in observed_events)
            session = sessions.all()[0]
            assert session.pending_audio_history_archive is None
            assert session.history == []
            assert archive_asr_calls == []

            # The deferred native SPEAK must wait for this same run's fresh
            # quiet endpoint, then rebuild the full causal snapshot through it.
            assert len(processor_audio_snapshots) == 1
            for expected_frame_id in range(1, 11):
                _send_pcm_level(ws, 0)
                frame_events = _receive_until(ws, "input_audio_buffer.consumed")
                observed_events.extend(frame_events)
                assert frame_events[-1]["frame_id"] == expected_frame_id
            if endpoint_policy_action == "YIELD":
                yield_decision = _receive_action_decision(ws, "YIELD")
                observed_events.append(yield_decision)
                assert yield_decision["frame_id"] == 10
                assert yield_decision["accepted"] is False
                assert not engine.talker_started.wait(timeout=0.05), "YIELD must discard the deferred native SPEAK"
                assert len([event for event in observed_events if event.get("type") == "response.created"]) == 1
                assert len(processor_audio_snapshots) == 1
                ws.send_json({"type": "session.close"})
                return
            assert engine.talker_started.wait(timeout=2.0), "SPEAK endpoint deferred was not dispatched"
            engine.release_talker.set()
            second_done_events = _receive_until(ws, "response.done", limit=60)
            observed_events.extend(second_done_events)
            assert len(processor_audio_snapshots) == 2
            endpoint_audio = processor_audio_snapshots[-1]
            assert endpoint_audio.shape == (11 * FRAME_SAMPLES,)
            assert np.all(endpoint_audio[-FRAME_SAMPLES:] == 0.0)
            created = [event for event in observed_events if event.get("type") == "response.created"]
            assert len(created) == 2
            assert created[1]["response_id"] != done["response_id"]
            assert created[1]["intent_source_frame_id"] == 0
            assert created[1]["intent_endpoint_frame_id"] == 10
            assert not any(
                event.get("type") == "response.output_text.delta"
                and event.get("response_id") == done["response_id"]
                for event in observed_events
            )
            assert second_done_events[-1]["response_id"] == created[1]["response_id"]
            second_done = next(
                event for event in second_done_events
                if event.get("type") == "response.done"
            )
            assert second_done["runtime_metrics"]["audio_response_protocol"]["first_thinker_token_id"] == 100
            session = sessions.all()[0]
            assert [message.get("role") for message in session.history[-2:]] == ["user", "assistant"]
            assert session.history[-2]["content"] == "ho detto ciao"
            assert session.history[-1]["content"] == "Ciao, ti ascolto."
            ws.send_json({"type": "session.close"})
            return
        for expected_frame_id in range(
            11 if wake_action == "SPEAK" else 1,
            13 if wake_action == "SPEAK" else 11,
        ):
            _send_pcm_level(ws, 0)
            frame_events = _receive_until(ws, "input_audio_buffer.consumed")
            observed_events.extend(frame_events)
            consumed = frame_events[-1]
            assert consumed["frame_id"] == expected_frame_id
        assert engine.talker_started.wait(timeout=2.0)
        if adapter_enabled:
            assert engine.say_hidden_seen is None
            assert engine.frame_residual_hidden_seen is not None
            assert tuple(engine.talker_prefix_seen.shape) == (1, 0, 2048)
        else:
            assert engine.say_hidden_seen is not None
            assert engine.say_hidden_seen.shape == (1, 1, 1536)
            assert float(engine.say_hidden_seen[0, 0, 0]) == 22.0
            assert engine.talker_prefix_seen is not None
            assert tuple(engine.talker_prefix_seen.shape) == (1, 2, 2048)
            assert float(engine.talker_prefix_seen[0, 0, 0]) == 22.0

        last_listener_frame_id = max(
            int(event["frame_id"])
            for event in observed_events
            if event.get("type") == "input_audio_buffer.consumed"
        )
        for _ in range(2):
            _send_pcm_level(ws, 0)
            frame_events = _receive_until(ws, "input_audio_buffer.consumed")
            observed_events.extend(frame_events)
            consumed = frame_events[-1]
            assert int(consumed["frame_id"]) > last_listener_frame_id
            last_listener_frame_id = int(consumed["frame_id"])
        session = sessions.all()[0]
        if not adapter_enabled:
            assert _wait_for(lambda: session.conditioning_buffer.available_through >= 1)
        listener_hidden_before_cancel = (
            session.listener_hidden.detach().clone()
            if callable(getattr(session.listener_hidden, "detach", None))
            else np.array(session.listener_hidden, copy=True)
        )
        if not adapter_enabled:
            assert engine.v9_stream.calls >= 13
        assert float(listener_hidden_before_cancel[0, 0]) >= 12.0
        if adapter_enabled:
            assert torch.equal(engine.talker_prefix_seen, torch.empty((1, 0, 2048)))
        else:
            assert torch.equal(engine.talker_prefix_seen, torch.full((1, 2, 2048), 22.0))
        if cancel_during_talker is False:
            session.stop_requested = True
        engine.release_talker.set()
        terminal_event = "response.done" if cancel_during_talker is None else "response.cancelled"
        events = _receive_until_any(ws, {"response.done", "response.cancelled"})
        observed_events.extend(events)
        assert events[-1]["type"] == terminal_event
        assert engine.stop_check_after_cancel is (cancel_during_talker is False)
        assert engine.talker_kwargs_seen is not None
        if adapter_enabled:
            assert _wait_for(lambda: engine.response_talk_context_pool.active_count == 0)
        if talker_error:
            assert any(
                event.get("type") == "error"
                and event.get("error", {}).get("code") == "INFERENCE_ERROR"
                for event in events
            )
            failed = next(event for event in events if event.get("type") == "response.done")
            assert failed["status"] == "failed"
            ws.send_json({"type": "session.close"})
            return
        session = sessions.all()[0]
        assert _wait_for(lambda: session.pending_audio_history_archive is None)
        assert archive_asr_calls == []
        selected = next(event for event in observed_events if event.get("type") == "response.action_selected")
        assert selected["wake_action"] == "SPEAK"
        assert selected["response_action"] == response_action
        assert selected["deferred_until_endpoint"] is False
        audio_event_indexes = [
            index for index, event in enumerate(observed_events)
            if event.get("type") == "response.audio.delta"
        ]
        if cancel_during_talker is None:
            assert audio_event_indexes
            assert observed_events.index(selected) < audio_event_indexes[0]
            response_done = next(
                event for event in events if event.get("type") == "response.done"
            )
            assert response_done["runtime_metrics"]["audio_response_protocol"]["response_protocol"] == AUDIO_RESPONSE_PROTOCOL_V2
            assert response_done["runtime_metrics"]["audio_response_protocol"]["first_thinker_token_id"] == 100
            if adapter_enabled:
                assert response_done["runtime_metrics"]["frame_residual_adapter"]["source"] == (
                    "validated_v2_say_posttoken_hidden"
                )
        visible_text = "".join(
            str(event.get("delta", ""))
            for event in observed_events
            if event.get("type") == "response.output_text.delta"
        )
        assert '"heard"' not in visible_text and '"action"' not in visible_text
        expected_user_history = [] if response_action == "BACKCHANNEL" else ["ho detto ciao"]
        if cpu_archive_case:
            assert len(cpu_archive_calls) == 1
            assert np.array_equal(cpu_archive_calls[0], processor_audio_snapshots[0])
            expected_user_history = ["trascrizione CPU del turno reale"]
        assert [message["content"] for message in session.history if message.get("role") == "user"] == expected_user_history
        expected_assistant_history = (
            [say_text]
            if cancel_during_talker is None
            else []
        )
        assert [message["content"] for message in session.history if message.get("role") == "assistant"] == expected_assistant_history
        direct_prompt = engine.tokenizer.seen_messages[-1]
        assert "Stiamo organizzando una cena con amici." in direct_prompt[0]["content"]
        assert direct_prompt[0]["content"].endswith(AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM)
        ws.send_json({"type": "session.close"})


@pytest.mark.parametrize(
    ("action", "say_text", "expected_min_frames", "expected_frame_cap"),
    [
        ("SPEAK", "Ciao.", 48, server_module.ARIA_MAX_FRAMES),
        ("BACKCHANNEL", "Va bene grazie davvero.", 2, server_module.ARIA_BACKCHANNEL_MAX_FRAMES),
        ("MANUAL", "Ciao.", 48, server_module.ARIA_MAX_FRAMES),
        ("MANUAL", "", None, 0),
    ],
)
def test_controller_guided_vocal_actions_start_live_qwen_and_keep_plain_prefill(monkeypatch, action, say_text, expected_min_frames, expected_frame_cap) -> None:
    """Controller SAY starts its Qwen consumer during Thinker decode with one-token prefill."""
    processor_audio_snapshots: list[np.ndarray] = []

    class _AudioProcessor:
        audio_token = "<|audio|>"

        def __call__(self, **kwargs):
            processor_audio_snapshots.extend(
                np.asarray(audio, dtype=np.float32).copy()
                for audio in kwargs.get("audio", ())
            )
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "input_features": torch.ones((1, 4, 8)),
                "input_features_mask": torch.ones((1, 4), dtype=torch.long),
            }

    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", CONTROLLER_GUIDED_SAY_V1)
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "1")
    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    engine = _Engine(hidden_size=4, block_talk=False)
    engine.v9_stream = _V9StreamStub()
    engine.device = "cpu"
    engine.dtype = torch.float32
    qwen_processor = _QwenTextProcessor()
    engine.talker = SimpleNamespace(
        proc=qwen_processor,
        tcfg=SimpleNamespace(hidden_size=4),
        cfg=SimpleNamespace(tts_eos_token_id=9999),
    )
    engine.frame_residual_adapter_enabled = True
    engine.frame_residual_adapter = SimpleNamespace(
        architecture_id="response_hidden_q1_query_film_v2",
    )
    engine._thinker_wrapper = SimpleNamespace(processor=_AudioProcessor())
    engine.thinker = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=999),
        config=SimpleNamespace(eos_token_id=999),
    )
    engine.tokenizer.eos_token_id = 999
    engine.tokenizer.all_special_ids = [999]
    class _Graph:
        def reset(self):
            return None

    talk_graphs = [_Graph(), _Graph()]
    talk_context_pool = ResponseTalkContextPool(talk_graphs)
    engine.response_talk_context_pool = talk_context_pool
    say_token_pieces = (
        [(101, "Va"), (102, " bene"), (2, " grazie"), (104, " davvero.")]
        if say_text == "Va bene grazie davvero."
        else ([(101, "C"), (102, "ia"), (2, "o.")] if say_text else [])
    )
    say_hidden = torch.full((1, 3 + len(say_token_pieces), 1536), -9.0)
    if say_token_pieces:
        say_hidden[:, 3:, :] = 22.0
    engine.controller_forced_stop = None
    engine.talker_entered = threading.Event()
    engine.first_pcm_callback = threading.Event()
    engine.thinker_returned = threading.Event()
    engine.token_2_callback_seen = threading.Event()
    engine.pcm_preceded_thinker_return = False
    engine.live_frame_residual_hidden_seen = None

    def think(_ids, *, on_generated_token=None, **_kwargs):
        if say_text:
            assert callable(on_generated_token)
            for token_index, (token_id, token_text) in enumerate(say_token_pieces):
                stop = on_generated_token(
                    token_index,
                    token_id,
                    token_text,
                    torch.full((1, 1, 1536), 22.0),
                )
                if stop is not None:
                    engine.controller_forced_stop = stop
                if token_index == 2 and token_id == 2:
                    engine.token_2_callback_seen.set()
                    assert not engine.thinker_returned.is_set()
                    assert engine.first_pcm_callback.wait(timeout=1.0), (
                        "first internal PCM callback did not run while Thinker was blocked on token 2"
                    )
                    engine.pcm_preceded_thinker_return = not engine.thinker_returned.is_set()
                if stop is not None:
                    break
            assert engine.talk_started.wait(timeout=1.0), "live Qwen consumer did not start during Thinker decode"
            assert engine.talker_entered.wait(timeout=1.0), "engine.talk was not entered before Thinker returned"
            assert engine.first_pcm_callback.wait(timeout=1.0), "existing PCM callback did not run before Thinker returned"
        engine.thinker_returned.set()
        return {
            "text": say_text,
            "hidden": say_hidden,
            "hidden_prompt_len": 3,
            "gen_ids": [token_id for token_id, _ in say_token_pieces],
            "runtime_metrics": {},
            "terminal_reason": "eos",
        }

    engine.say_hidden_seen = None
    engine.v9_prefix_calls = 0

    def v9_prefix(hidden, prompt_len):
        engine.v9_prefix_calls += 1
        assert prompt_len == 0
        engine.say_hidden_seen = hidden.detach().clone()
        return torch.full((1, 2, 2048), 77.0)

    def v9_prefix_stateful(hidden, prompt_len, **_kwargs):
        return SimpleNamespace(h_pfx=v9_prefix(hidden, prompt_len))

    engine.talker_prefix_seen = None

    engine.talker_min_frames_seen = None
    engine.talker_budget_seen = None
    engine.talker_kwargs_seen = None
    engine.talker_context_seen = None
    engine.live_tail_tokens_seen = []
    engine.live_qwen_session_count = 0
    engine.talker_exited = threading.Event()
    engine._thinker_lock = threading.Lock()
    engine._talker_lock = threading.Lock()
    engine._gpu_priority_lock = threading.Lock()

    def talk(text, h_pfx, *, on_audio_chunk=None, min_frames=48, n_words=None,
             raw_text_token_ids=None, include_text_eos=True,
             live_qwen_text_stream=None, max_frames=None, **kwargs):
        engine.talker_entered.set()
        engine.live_qwen_session_count += 1
        assert not engine.thinker_returned.is_set()
        assert text
        assert tuple(h_pfx.shape) == (1, 0, 2048)
        assert raw_text_token_ids and len(raw_text_token_ids) == 1
        assert include_text_eos is False
        assert live_qwen_text_stream is not None
        engine.talker_kwargs_seen = {
            "prefill_text": text,
            "raw_text_token_ids": list(raw_text_token_ids),
            "include_text_eos": include_text_eos,
            "max_frames": max_frames,
        }
        engine.talker_min_frames_seen = min_frames
        engine.talker_budget_seen = max_frames
        engine.talker_prefix_seen = h_pfx.detach().clone()
        engine.talker_context_seen = kwargs.get("talk_context")
        assert engine.talker_context_seen is not None
        live_hidden_stream = kwargs.get("live_frame_residual_hidden_stream")
        assert live_hidden_stream is not None
        first_frame_hidden = live_hidden_stream.hidden_for_frame(0)
        engine.live_frame_residual_hidden_seen = first_frame_hidden.hidden.detach().clone()
        assert first_frame_hidden.tail_hold is False
        wav = np.full(1920, 0.1, dtype=np.float32)
        if on_audio_chunk is not None:
            on_audio_chunk(wav)
            engine.first_pcm_callback.set()
        engine.talk_started.set()
        consumed = []
        while True:
            assert not engine._thinker_lock.locked(), "Thinker priority lock held while waiting for Qwen text"
            assert not engine._talker_lock.locked(), "Talker lock held while waiting for Qwen text"
            assert not engine._gpu_priority_lock.locked(), "GPU priority lock held while waiting for Qwen text"
            token_id = live_qwen_text_stream.take_trailing_token(timeout_s=2.0)
            if token_id is None:
                break
            consumed.append(token_id)
        assert consumed.count(9999) == 1
        assert live_qwen_text_stream.codec_eos_allowed
        engine.live_tail_tokens_seen = consumed
        engine.talker_exited.set()
        return wav

    engine.think = think
    engine.v9_prefix = v9_prefix
    engine.v9_prefix_stateful = v9_prefix_stateful
    engine.talk = talk
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        seen_events: list[dict[str, object]] = []

        def receive_until_test(target: str, *, limit: int = 80):
            received = []
            for _ in range(limit):
                async def receive_with_timeout():
                    return await asyncio.wait_for(ws._send_rx.receive(), timeout=2.0)

                try:
                    message = ws.portal.call(receive_with_timeout)
                except TimeoutError as exc:
                    raise AssertionError(
                        f"{target} timed out; event sequence={seen_events[-30:]!r}"
                    ) from exc
                if message.get("type") != "websocket.send":
                    raise AssertionError(
                        f"websocket closed while waiting for {target}; "
                        f"message={message!r}; event sequence={seen_events[-30:]!r}"
                    )
                event = json.loads(message["text"])
                received.append(event)
                seen_events.append(event)
                if event.get("type") == "error":
                    raise AssertionError(f"realtime error while waiting for {target}: {event!r}")
                if event.get("type") == target:
                    return received
            raise AssertionError(f"{target} not received; last events={received[-12:]!r}")

        session_config = {"input_audio_buffer_acknowledgements": True}
        if action == "MANUAL":
            session_config["auto_response"] = False
        else:
            session_config.update({
                "auto_response": True,
                "turn_policy": "scripted",
                "scripted_actions": [action],
            })
        ws.send_json({"type": "session.update", "session": session_config})
        assert ws.receive_json()["type"] == "session.updated"
        _send_pcm(ws)
        receive_until_test("input_audio_buffer.consumed")
        if action == "SPEAK":
            for expected_frame_id in range(1, 11):
                _send_pcm_level(ws, 0)
                quiet_events = receive_until_test("input_audio_buffer.consumed")
                assert quiet_events[-1]["frame_id"] == expected_frame_id
        if action == "MANUAL":
            ws.send_json({"type": "input_audio_buffer.commit"})
        events = receive_until_test("response.done", limit=80)

        expected_action = "SPEAK" if action == "MANUAL" else action
        selected = next(event for event in events if event.get("type") == "response.action_selected")
        assert selected["wake_action"] == expected_action
        assert selected["response_action"] == expected_action
        assert selected["deferred_until_endpoint"] is False
        assert engine.controller_forced_stop == ((999, "structure") if action == "BACKCHANNEL" and say_text else None)
        assert engine.v9_prefix_calls == 0
        assert any(event.get("type") == "response.audio.delta" for event in events) == bool(say_text)
        if say_text:
            assert engine.token_2_callback_seen.is_set()
            assert engine.pcm_preceded_thinker_return is True
            assert engine.live_qwen_session_count == 1
            assert engine.say_hidden_seen is None
            assert engine.live_frame_residual_hidden_seen is not None
            assert tuple(engine.live_frame_residual_hidden_seen.shape) == (1, 1, 1536)
            assert torch.equal(engine.live_frame_residual_hidden_seen, torch.full((1, 1, 1536), 22.0))
            assert engine.talker_prefix_seen is not None
            assert tuple(engine.talker_prefix_seen.shape) == (1, 0, 2048)
            assert engine.talker_min_frames_seen == expected_min_frames
            assert engine.talker_budget_seen == expected_frame_cap
            assert engine.talker_kwargs_seen["include_text_eos"] is False
            assert len(engine.talker_kwargs_seen["raw_text_token_ids"]) == 1
            assert len(engine.live_tail_tokens_seen) >= 1
            assert engine.live_tail_tokens_seen[0] == 1000 + ord(say_text[1])
            assert engine.live_tail_tokens_seen[-1] == 9999
            response_done = next(event for event in events if event.get("type") == "response.done")
            assert response_done["runtime_metrics"]["audio_response_protocol"]["first_thinker_token_id"] == 101
            assert engine.talker_exited.is_set(), "Talker worker must exit before response.done"
            talker_metrics = response_done["runtime_metrics"]["talker"]
            assert "live_say_consumer_started_before_thinker_return" in talker_metrics, talker_metrics
            assert talker_metrics["live_say_consumer_started_before_thinker_return"] is True
            assert talker_metrics["live_say_consumer_started_at_ns"] < talker_metrics["live_say_thinker_returned_at_ns"]
            assert isinstance(response_done.get("t_first_audio_chunk_pcm_at_ns"), int)
            assert isinstance(response_done.get("response_done_at_monotonic_ns"), int)
            assert response_done["t_first_audio_chunk_pcm_at_ns"] < response_done["response_done_at_monotonic_ns"]
            if action in {"SPEAK", "BACKCHANNEL"}:
                selected = next(
                    event for event in seen_events
                    if event.get("type") == "response.action_selected"
                )
                assert isinstance(selected.get("policy_decision_at_ns"), int)
            assert sessions.all()[0].history[-1] == {"role": "assistant", "content": say_text}
            assert _wait_for(
                lambda: not any(
                    thread.name.startswith("controller-guided-say-")
                    for thread in threading.enumerate()
                ),
                timeout=2.0,
            ), "live Qwen worker was not terminal/joined at response.done"
        else:
            assert engine.say_hidden_seen is None
            assert engine.live_frame_residual_hidden_seen is None
            assert engine.talker_prefix_seen is None
            assert engine.talker_min_frames_seen is None
            assert any(event.get("status") == "no_speech" for event in events if event.get("type") == "response.done")
        assert not engine.transcriptions
        if action == "MANUAL":
            created = next(event for event in events if event.get("type") == "response.created")
            assert created["action"] == "SPEAK"
            assert created["intent_source"] == "manual_audio_commit"
            assert len(processor_audio_snapshots) == 1
            np.testing.assert_array_equal(
                processor_audio_snapshots[0],
                np.full(FRAME_SAMPLES, 4096 / 32768.0, dtype=np.float32),
            )
        assert talk_context_pool.active_count == 0
        ws.send_json({"type": "session.close"})


def test_speak_intent_survives_backchannel_until_fresh_causal_endpoint() -> None:
    engine = _Engine(hidden_size=2048, block_talk=True, block_transcription=True)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))
    observed: list[dict[str, object]] = []

    def receive_until(predicate, *, limit: int = 100):
        for _ in range(limit):
            for event in observed:
                if predicate(event):
                    return event
            observed.append(ws.receive_json())
        raise AssertionError(f"evento atteso non ricevuto; ultimi eventi={observed[-12:]!r}")

    def send_frame(frame_id: int, value: int) -> None:
        _send_pcm_level(ws, value)
        receive_until(
            lambda event: event.get("type") == "input_audio_buffer.consumed"
            and event.get("frame_id") == frame_id
        )

    try:
        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({
                "type": "session.update",
                "session": {
                    "turn_policy": "scripted",
                    "scripted_actions": ["BACKCHANNEL", "SPEAK", "LISTEN"],
                    "input_audio_buffer_acknowledgements": True,
                },
            })
            assert ws.receive_json()["type"] == "session.updated"

            send_frame(0, 4096)
            backchannel = receive_until(
                lambda event: event.get("type") == "response.created"
                and event.get("effective_action") == "BACKCHANNEL"
            )
            assert engine.talk_started.wait(timeout=1.0), (
                f"BACKCHANNEL worker did not reach talk; events={observed!r}; "
                f"think_calls={engine.think_calls!r}"
            )

            send_frame(1, 4096)
            raw_speak = receive_until(
                lambda event: event.get("type") == "duplex_control6.decision"
                and event.get("action") == "SPEAK"
                and event.get("frame_id") == 1
            )
            assert raw_speak["accepted"] is False
            if raw_speak["rejection_reason"] != "deferred_until_backchannel_done":
                engine.release_talk.set()
            assert raw_speak["rejection_reason"] == "deferred_until_backchannel_done"

            # The first endpoint is provisional. Resumed speech invalidates it
            # before a second endpoint can dispatch the original intent.
            for frame_id in range(2, 10):
                send_frame(frame_id, 0)
            send_frame(10, 8192)
            for frame_id in range(11, 21):
                send_frame(frame_id, 0)

            engine.release_talk.set()
            receive_until(
                lambda event: event.get("type") == "response.done"
                and event.get("response_id") == backchannel.get("response_id")
            )

            # A non-voiced, nonzero dispatch frame makes future-frame leakage
            # visible: the committed SPEAK input must end at endpoint 20.
            send_frame(21, 64)
            created = receive_until(
                lambda event: event.get("type") == "response.created"
                and event.get("effective_action") == "SPEAK"
            )
            assert engine.transcription_started.wait(timeout=1.0)
            assert created["action"] == "SPEAK"
            assert created["intent_source_frame_id"] == 1
            assert created["intent_endpoint_frame_id"] == 20
            assert len(engine.transcriptions) == 1
            snapshot = engine.transcriptions[0]
            # The frozen turn includes the original onset frame 0 through
            # endpoint frame 20, both inclusive.
            assert snapshot.shape == (21 * FRAME_SAMPLES,)
            assert np.all(snapshot[-FRAME_SAMPLES:] == 0.0)
            assert len([event for event in observed if event.get("type") == "response.created"]) == 2

            engine.release_transcription.set()
            receive_until(
                lambda event: event.get("type") == "response.done"
                and event.get("response_id") == created.get("response_id")
            )
            ws.send_json({"type": "session.close"})
    finally:
        engine.release_talk.set()
        engine.release_transcription.set()


def test_barge_in_during_asr_fences_transcript_and_prevents_ghost_audio() -> None:
    engine = _Engine(block_transcription=True)
    sessions = _AcceleratedSchedulerSessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            # Keep SPEAK selected until the two-frame input reaches its quiet
            # endpoint, then HARD_STOP the ASR/response generation.
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["SPEAK"] * 12 + ["HARD_STOP"],
                "input_audio_buffer_acknowledgements": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]
        for frame_id, level in enumerate([4096, 4096] + [0] * 10):
            _send_pcm_level(ws, level)
            for _ in range(80):
                event = ws.receive_json()
                if (
                    event.get("type") == "input_audio_buffer.consumed"
                    and event.get("frame_id") == frame_id
                ):
                    break
                assert event.get("type") != "error", event
            else:
                raise AssertionError(f"input frame {frame_id} was not acknowledged")
        assert engine.transcription_started.wait(timeout=2.0)
        _send_pcm_level(ws, 4096)
        for _ in range(80):
            event = ws.receive_json()
            if event.get("type") == "input_audio_buffer.consumed" and event.get("frame_id") == 12:
                break
            assert event.get("type") != "error", event
        else:
            raise AssertionError("barge-in frame was not acknowledged")
        assert _wait_for(lambda: session.session_generation_id >= 2)
        engine.release_transcription.set()
        events = _receive_until(ws, "response.cancelled")
        assert events[-1]["type"] == "response.cancelled"
        assert not any(event.get("type") == "response.audio.delta" for event in events)
        assert not engine.think_calls
        assert session.history == []
        assert session.stale_session_mutations == 0
        ws.send_json({"type": "session.update", "session": {}})
        after_cancel = _receive_until(ws, "session.updated")
        assert [event["type"] for event in after_cancel] == ["session.updated"]
        ws.send_json({"type": "session.close"})


def test_bounded_two_turn_audio_dialogue_keeps_transcripts_and_assistant_history_ordered() -> None:
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        for _ in range(2):
            _send_pcm(ws)
            ws.send_json({"type": "input_audio_buffer.commit"})
            _receive_until(ws, "response.done")

        assert len(engine.transcriptions) == 2
        assert engine.think_calls == ["think", "think"]
        assert session.history == [
            {"role": "user", "content": "recognized turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "recognized turn 2"},
            {"role": "assistant", "content": "reply 2"},
        ]
        assert [message["content"] for message in engine.tokenizer.seen_messages[-1]] == [
            "recognized turn 1",
            "reply 1",
            "recognized turn 2",
        ]
        ws.send_json({"type": "session.close"})


@pytest.mark.parametrize("trace_asr_text", [False, True])
def test_raw_asr_transcript_is_never_returned_in_runtime_metrics(monkeypatch, trace_asr_text: bool) -> None:
    if trace_asr_text:
        monkeypatch.setenv("DUPLEX_TRACE_ASR_TEXT", "1")
    else:
        monkeypatch.delenv("DUPLEX_TRACE_ASR_TEXT", raising=False)
    engine = _Engine()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        _send_pcm(ws)
        ws.send_json({"type": "input_audio_buffer.commit"})
        events = _receive_until(ws, "response.done")
        terminal = events[-1]
        assert terminal["type"] == "response.done"
        assert terminal["response_id"]
        asr_metrics = terminal["runtime_metrics"]["asr"]
        assert "input_transcript" not in asr_metrics
        ws.send_json({"type": "session.close"})


def test_backchannel_action_generates_contextual_thinker_ack_without_asr() -> None:
    engine = _Engine(hidden_size=2048)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["BACKCHANNEL"],
                "auto_response": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        _send_pcm(ws)
        events = _receive_until(ws, "response.done")

        assert engine.transcriptions == [], "BACKCHANNEL must not transcribe the unfinished input utterance"
        assert engine.think_calls == ["think"], "BACKCHANNEL must invoke Gemma Thinker"
        assert engine.think_max_new_tokens == [2]
        assert engine.think_min_new_tokens == [0]
        assert engine.talk_calls and engine.talk_calls[0][0] == "reply 1"
        assert any(event.get("type") == "response.audio.delta" for event in events)
        action_system = next(
            message["content"] for message in engine.tokenizer.seen_messages[-1]
            if message.get("role") == "system" and "Current turn action: BACKCHANNEL." in message.get("content", "")
        )
        assert "one to four words" in action_system
        assert engine.tokenizer.seen_messages[-1][-1] == {
            "role": "user",
            "content": "",
        }
        session = sessions.all()[0]
        assert session.instructions == ""
        assert session.history == [{"role": "assistant", "content": "reply 1"}]
        terminal = next(event for event in events if event.get("type") == "response.done")
        generation_trace = terminal["runtime_metrics"]["generation_trace"]
        assert generation_trace["conditioned_action"] == "BACKCHANNEL"
        assert generation_trace["thinker_token_ids"] == [101, 102]
        ws.send_json({"type": "session.close"})


def test_private_control_event_echo_split_across_text_chunks_is_rejected_before_talker() -> None:
    engine = _Engine(hidden_size=2048)
    marker = action_control_marker_text("REPLAN")
    engine.think_text_override = marker + " Va bene."
    engine.think_text_chunks = [marker[:17], marker[17:] + " Va bene."]
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": ["BACKCHANNEL"],
                "auto_response": True,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        _send_pcm(ws)
        events = _receive_until_any(ws, {"response.cancelled", "response.done"})

        errors = [event for event in events if event.get("type") == "error"]
        assert len(errors) == 1
        assert errors[0]["error"]["code"] == "PRIVATE_CONTROL_LEAK"
        assert errors[0]["response_id"] == events[-1]["response_id"]
        assert events.index(errors[0]) < len(events) - 1
        assert events[-1]["type"] == "response.cancelled"
        assert events[-1]["reason"] == "private_control_leak"
        assert not engine.talk_calls, "an echoed internal event must be rejected before Qwen Talker"
        assert not any(event.get("type") == "response.audio.delta" for event in events)
        assert not any(marker in str(event.get("delta", "")) for event in events)
        assert not sessions.all()[0].history
        trace = events[-1]["runtime_metrics"]["generation_trace"]
        assert trace["raw_text"] == marker + " Va bene."
        assert trace["outcome"] == "private_control_leak"
        ws.send_json({"type": "session.close"})


def test_replan_cancels_stream_and_returns_to_listening_without_rendering_replan() -> None:
    engine = _Engine(block_talk=True)
    release_followup_transcription = threading.Event()
    followup_transcription_started = threading.Event()

    def transcribe_first_then_block_followup(pcm) -> dict[str, str]:
        snapshot = np.array(pcm, dtype=np.float32, copy=True)
        engine.transcriptions.append(snapshot)
        transcription_index = len(engine.transcriptions)
        if transcription_index == 1:
            engine.transcription_started.set()
            return {"text": "recognized turn 1"}
        followup_transcription_started.set()
        if not release_followup_transcription.wait(timeout=3.0):
            raise TimeoutError("test follow-up ASR worker was not released")
        return {"text": f"recognized turn {transcription_index}"}

    engine.transcribe_audio = transcribe_first_then_block_followup

    def talk_with_streamable_packet(
        text,
        h_pfx,
        *,
        on_audio_chunk_with_horizon=None,
        **_kwargs,
    ):
        engine.talk_calls.append((text, h_pfx))
        wav = np.full(4 * 1920, 0.1, dtype=np.float32)
        if on_audio_chunk_with_horizon is not None:
            on_audio_chunk_with_horizon(wav, 10)
        engine.talk_started.set()
        if not engine.release_talk.wait(timeout=3.0):
            raise TimeoutError("test talk worker was not released")
        return wav

    engine.talk_stateful = talk_with_streamable_packet
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        scripted_actions = ["LISTEN"] * 10 + ["SPEAK", "REPLAN", "REPLAN"]
        scripted_actions += ["CONTINUE", "CONTINUE", "REPLAN"] + ["CONTINUE"] * 7
        ws.send_json({
            "type": "session.update",
            "session": {
                "turn_policy": "scripted",
                "scripted_actions": scripted_actions,
            },
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]
        invalidation_reasons: list[tuple[str, bool, int, int]] = []
        original_invalidate = session.invalidate_generation

        def track_invalidation(reason: str = "") -> int:
            before_closed = session._closed
            before_generation = session.session_generation_id
            result = original_invalidate(reason)
            invalidation_reasons.append((reason, before_closed, before_generation, result))
            return result

        session.invalidate_generation = track_invalidation

        # Keep the first intent pending until a real 800 ms quiet endpoint:
        # one voiced frame followed by ten 80 ms quiet frames.
        for frame_id in range(10):
            _send_pcm_level(ws, 4096 if frame_id == 0 else 0)
            listen = _receive_action_decision(ws, "LISTEN")
            assert listen["frame_id"] == frame_id
        _send_pcm_level(ws, 0)
        first_events = _receive_until_any(
            ws,
            {"response.created", "error", "response.cancelled", "response.done"},
        )
        assert first_events[-1]["type"] == "response.created", first_events
        first_response_id = first_events[-1]["response_id"]
        first_speak = next(
            event for event in first_events
            if event.get("type") == "duplex_control6.decision" and event.get("action") == "SPEAK"
        )
        assert first_speak["frame_id"] == 10
        assert first_speak["accepted"] is True
        assert engine.transcription_started.wait(timeout=1.0)
        assert engine.talk_started.wait(timeout=1.0)
        first_audio_events = _receive_until(ws, "response.audio.delta")
        assert any(event.get("response_id") == first_response_id for event in first_audio_events)

        # A voiced correction starts a new utterance and cancels the active
        # response immediately. Its ASR handoff waits for the correction's
        # own ten-frame quiet endpoint below.
        _send_pcm_level(ws, 8192)
        cancelled_events = _receive_until_any(
            ws,
            {"response.cancelled", "error", "response.done"},
        )
        assert cancelled_events[-1]["type"] == "response.cancelled", cancelled_events
        first_replan_decision = next(
            event for event in cancelled_events
            if event.get("type") == "duplex_control6.decision" and event.get("action") == "REPLAN"
        )
        assert first_replan_decision["frame_id"] == 11
        assert first_replan_decision["accepted"] is True
        cancelled = cancelled_events[-1]
        assert cancelled["response_id"] == first_response_id
        assert cancelled["reason"] == "replan"
        assert not followup_transcription_started.is_set(), "follow-up ASR waits for the correction endpoint"
        assert session.session_generation_id == 2
        assert invalidation_reasons == [("control:REPLAN", False, 1, 2)]
        assert len(engine.transcriptions) == 1
        engine.release_talk.set()

        # A repeated edge is rejected; two ordinary ticks reopen the edge,
        # but another REPLAN cannot queue while the first handoff is pending.
        _send_pcm_level(ws, 0)
        repeated_replan = _receive_action_decision(ws, "REPLAN")
        assert repeated_replan["frame_id"] == 12
        assert repeated_replan["accepted"] is False
        assert repeated_replan["rejection_reason"] == "replan_edge_latched"
        assert session.session_generation_id == 2
        assert len(engine.transcriptions) == 1

        for expected_frame_id in (13, 14):
            _send_pcm_level(ws, 0)
            non_replan = _receive_action_decision(ws, "CONTINUE")
            assert non_replan["frame_id"] == expected_frame_id
            assert non_replan["accepted"] is False
        _send_pcm_level(ws, 0)
        pending_replan = _receive_action_decision(ws, "REPLAN")
        assert pending_replan["frame_id"] == 15
        assert pending_replan["accepted"] is False
        assert pending_replan["rejection_reason"] == "replan_handoff_pending"
        assert session.session_generation_id == 2
        assert len(engine.transcriptions) == 1

        # The correction reaches its endpoint at frame 21. REPLAN is a control
        # transition and must not start another ASR/Thinker/Talker response.
        for expected_frame_id in range(16, 22):
            _send_pcm_level(ws, 0)
            continue_decision = _receive_action_decision(ws, "CONTINUE")
            assert continue_decision["frame_id"] == expected_frame_id
        assert _wait_for(lambda: not session.policy_response_pending)
        assert getattr(session.control_state, "value", session.control_state) == "LISTENING"
        assert not followup_transcription_started.is_set()
        assert len(engine.transcriptions) == 1
        assert engine.think_calls == ["think"]
        assert len(engine.talk_calls) == 1
        ws.send_json({"type": "session.close"})


def test_replan_cancels_pending_renderer_before_first_pcm() -> None:
    engine = _Engine(block_talk=True)
    engine.release_talk.clear()
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))
    release_audio = threading.Event()
    finish_talk = threading.Event()
    audio_callback_attempted = threading.Event()

    def talk_without_early_pcm(
        text,
        h_pfx,
        *,
        on_audio_chunk_with_horizon=None,
        **_kwargs,
    ):
        engine.talk_calls.append((text, h_pfx))
        engine.talk_started.set()
        if not release_audio.wait(timeout=3.0):
            raise TimeoutError("test renderer was not released")
        wav = np.full(4 * 1920, 0.1, dtype=np.float32)
        if on_audio_chunk_with_horizon is not None:
            on_audio_chunk_with_horizon(wav, 10)
        audio_callback_attempted.set()
        if not finish_talk.wait(timeout=3.0):
            raise TimeoutError("test talk worker was not finished")
        return wav

    engine.talk_stateful = talk_without_early_pcm
    actions = ["LISTEN"] * 10 + ["SPEAK"] + ["REPLAN"] * 20
    try:
        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({"type": "session.update", "session": {
                "turn_policy": "scripted",
                "scripted_actions": actions,
                "input_audio_buffer_acknowledgements": True,
            }})
            assert ws.receive_json()["type"] == "session.updated"
            for frame_id in range(10):
                _send_pcm_level(ws, 4096 if frame_id == 0 else 0)
                decision = _receive_action_decision(ws, "LISTEN")
                assert decision["frame_id"] == frame_id
            _send_pcm_level(ws, 0)
            speak = _receive_action_decision(ws, "SPEAK")
            assert speak["accepted"] is True
            created = _receive_until(ws, "response.created")[-1]
            response_id = created["response_id"]
            assert engine.talk_started.wait(timeout=1.0)

            # A fresh REPLAN must fence a response whose renderer is pending,
            # even though it has not emitted its first audio packet.
            _send_pcm_level(ws, 8192)
            replan = _receive_action_decision(ws, "REPLAN")
            assert replan["accepted"] is True
            release_audio.set()
            assert audio_callback_attempted.wait(timeout=1.0)
            finish_talk.set()
            events = _receive_until(ws, "response.cancelled")
            cancelled = [e for e in events if e.get("type") == "response.cancelled"]
            assert cancelled and cancelled[-1]["response_id"] == response_id
            assert cancelled[-1]["reason"] == "replan"
            assert not any(
                e.get("type") == "response.audio.delta" and e.get("response_id") == response_id
                for e in events
            )
            ws.send_json({"type": "session.close"})
    finally:
        release_audio.set()
        finish_talk.set()
        engine.release_talk.set()


def test_quiet_tail_replan_is_rejected_while_original_response_worker_is_live() -> None:
    engine = _Engine(block_transcription=True)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {"turn_policy": "scripted", "scripted_actions": ["SPEAK", "REPLAN"]},
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]
        _send_pcm_level(ws, 4096)
        first_events = _receive_until(ws, "response.created")
        first_response_id = first_events[-1]["response_id"]
        first_speak = next(
            event for event in first_events
            if event.get("type") == "duplex_control6.decision" and event.get("action") == "SPEAK"
        )
        assert first_speak["frame_id"] == 0
        assert first_speak["accepted"] is True
        assert engine.transcription_started.wait(timeout=1.0)
        first_generation = session.session_generation_id

        # This complete post-commit frame is below the voiced-input thresholds
        # (RMS .00061, peak .00061 after PCM normalization). A raw REPLAN
        # prediction remains visible but cannot cancel the active ASR worker.
        _send_pcm_level(ws, 20)
        assert _wait_for(lambda: session.listener_state.listener_input_frames >= 2)
        quiet_replan = _receive_action_decision(ws, "REPLAN")
        assert quiet_replan["frame_id"] == 1
        assert quiet_replan["confidence"] > 0.0
        assert quiet_replan["accepted"] is False
        assert quiet_replan["rejection_reason"] == "no_new_user_audio"
        assert session.session_generation_id == first_generation
        assert session.active_response is True
        assert not engine.release_transcription.is_set()
        assert len(engine.transcriptions) == 1

        engine.release_transcription.set()
        completion = _receive_until(ws, "response.done")
        assert not any(
            event.get("type") == "response.cancelled"
            and event.get("response_id") == first_response_id
            and event.get("reason") == "replan"
            for event in completion
        )
        ws.send_json({"type": "session.close"})


@pytest.mark.parametrize("direct_audio_candidate", [False, True])
def test_text_response_create_keeps_existing_user_content_without_unnecessary_asr(
    monkeypatch: pytest.MonkeyPatch,
    direct_audio_candidate: bool,
) -> None:
    monkeypatch.setenv(
        "DUPLEX_DIRECT_AUDIO_CANDIDATE",
        "1" if direct_audio_candidate else "0",
    )
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "1")
    engine = _Engine()
    engine.v9_prefix = lambda _hidden, _prompt_len: np.zeros((1, 1, 2048), dtype=np.float32)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "conversation.item.create",
            "item": {"role": "user", "content": "text turn"},
        })
        assert ws.receive_json()["type"] == "conversation.item.created"
        ws.send_json({"type": "response.create"})
        _receive_until(ws, "response.done")

        assert engine.transcriptions == []
        assert engine.tokenizer.seen_messages[-1][-1] == {
            "role": "user",
            "content": "text turn",
        }
        assert sessions.all()[0].history == [
            {"role": "user", "content": "text turn"},
            {"role": "assistant", "content": "reply 1"},
        ]
        ws.send_json({"type": "session.close"})


def test_audio_origin_without_pcm_snapshot_fails_closed() -> None:
    audio_intent = {"source": "native_control6"}

    assert server_module._response_uses_direct_audio_candidate(
        candidate_enabled=True,
        committed_audio=np.empty(0, dtype=np.float32),
        intent_origin=audio_intent,
    )
    with pytest.raises(RuntimeError, match="direct_audio_context_missing"):
        server_module._require_direct_audio_context(audio_intent)


def test_empty_audio_transcription_does_not_create_a_ghost_ciao_response() -> None:
    engine = _Engine()
    engine.transcribe_audio = lambda _pcm: {"text": "  "}
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        _send_pcm(ws)
        ws.send_json({"type": "input_audio_buffer.commit"})
        events = _receive_until(ws, "response.cancelled")
        assert not engine.think_calls
        assert not engine.talk_calls
        assert not any(event.get("type") == "response.audio.delta" for event in events)
        assert not sessions.all()[0].history
        ws.send_json({"type": "session.close"})


def test_manual_commit_trims_precommit_samples_from_a_straddling_policy_frame() -> None:
    from duplex_kernel.session import DuplexSession

    session = DuplexSession("partial-commit")
    frame = np.arange(FRAME_SAMPLES, dtype=np.float32)
    session.reset_policy_audio(committed_sample_count=500)

    accepted = session.append_policy_audio(0, frame)

    assert accepted is True
    committed = session.take_policy_audio()
    assert committed.shape == (FRAME_SAMPLES - 500,)
    np.testing.assert_array_equal(committed, frame[500:])


def test_provisional_policy_fails_closed_without_a_turn_model() -> None:
    from duplex_kernel.turn_policy import ControlAction, PolicyInput, ProvisionalFramePolicy

    policy = ProvisionalFramePolicy(window_frames=2)
    silent = PolicyInput("p", 0, np.zeros(FRAME_SAMPLES), False, 0)

    for frame_id in range(50):
        state = PolicyInput("p", frame_id, silent.user_pcm, False, 0)
        assert policy.step(state).action is ControlAction.LISTEN
    active = PolicyInput("p", 50, np.zeros(FRAME_SAMPLES), True, 0)
    assert policy.step(active).action is ControlAction.CONTINUE
    policy.reset()
    assert policy.step(silent).action is ControlAction.LISTEN


def test_session_update_fails_explicitly_when_turnhead_checkpoint_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("duplex_kernel.turnhead_adapter.TurnHeadRuntime", _TurnHeadRuntimeStub)
    monkeypatch.setattr(server_module, "TURNHEAD_CKPT", "missing-turn-dynamics.pt")
    engine = _Engine()
    engine.default_turn_policy_factory = None
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        # This is the former silent-fallback path: auto_response with no explicit
        # turn_policy must now reject an unavailable default TurnHead.
        ws.send_json({"type": "session.update", "session": {"auto_response": True}})
        error = ws.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == "INVALID_SESSION_UPDATE"
        assert "TurnDynamics checkpoint unavailable" in error["error"]["message"]
        assert "missing-turn-dynamics.pt" in error["error"]["message"]
        assert session.auto_response is False
        assert not isinstance(session.turn_policy, ProvisionalFramePolicy)

        # The demo's explicit selection must also fail visibly, not retain an
        # unloaded head as though session.update had succeeded.
        ws.send_json({
            "type": "session.update",
            "session": {"auto_response": True, "turn_policy": "turnhead"},
        })
        explicit_error = ws.receive_json()
        assert explicit_error["type"] == "error"
        assert "TurnDynamics checkpoint unavailable" in explicit_error["error"]["message"]
        assert session.auto_response is False
        ws.send_json({"type": "session.close"})


def test_session_update_preflights_p7_voice_before_opening_audio_gate(monkeypatch) -> None:
    engine = _Engine()
    engine.talker = SimpleNamespace(
        tcfg=SimpleNamespace(hidden_size=4),
        default_voice_conditioning=None,
    )
    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"task_type": ["Base"]},
    )
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        ws.send_json({
            "type": "session.update",
            "session": {
                "auto_response": True,
                "voice": "gemma",
                "voice_conditioning_mode": "x_vector_only",
                "input_audio_buffer_acknowledgements": True,
            },
        })
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "VOICE_CONDITIONING_ERROR"
        assert event["error"]["terminal"] is True
        assert session.auto_response is False
        assert session.voice_conditioning is None
        ws.send_json({"type": "session.close"})


def test_session_update_echoes_validated_p7_voice_mode(monkeypatch) -> None:
    engine = _Engine()
    engine.talker = SimpleNamespace(
        tcfg=SimpleNamespace(hidden_size=4),
        default_voice_conditioning=None,
    )
    expected = {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)}
    monkeypatch.setattr(server_module, "resolve_voice_conditioning", lambda *_a, **_k: expected)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "auto_response": True,
                "voice": "gemma",
                "voice_conditioning_mode": "x_vector_only",
                "input_audio_buffer_acknowledgements": True,
            },
        })
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["voice"] == "gemma"
        assert updated["voice_conditioning_mode"] == "x_vector_only"
        assert sessions.all()[0].voice_conditioning is expected
        ws.send_json({"type": "session.close"})


def test_voice_conditioning_ablation_keeps_session_unconditioned(monkeypatch) -> None:
    engine = _Engine()
    engine.ablate_session_voice_conditioning = True
    engine.talker = SimpleNamespace(
        tcfg=SimpleNamespace(hidden_size=4),
        default_voice_conditioning=None,
    )
    expected_profile = {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)}
    resolutions: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_resolution(*args, **kwargs):
        resolutions.append((args, kwargs))
        return expected_profile

    monkeypatch.setattr(server_module, "resolve_voice_conditioning", record_resolution)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "auto_response": True,
                "voice": "gemma",
                "voice_conditioning_mode": "x_vector_only",
            },
        })
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["voice_conditioning_mode"] is None
        assert sessions.all()[0].voice_conditioning is None
        assert resolutions == []
        ws.send_json({"type": "session.close"})


def test_voice_conditioning_ablation_ignores_explicit_request_voice(monkeypatch) -> None:
    engine = SimpleNamespace(ablate_session_voice_conditioning=True)
    sess = SimpleNamespace(
        voice="gemma",
        voice_style="warm",
        voice_conditioning={"mode": "x_vector_only"},
    )
    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: pytest.fail("voice resolver must be bypassed by the startup ablation"),
    )

    result = server_module._resolve_request_voice_conditioning(
        engine,
        sess,
        {"voice": "other-profile", "voice_style": "bright", "ref_audio": "reference.wav"},
    )

    assert result is None


def test_diagnostic_talker_rng_scope_reports_seed_without_mutating_global_rng() -> None:
    engine = SimpleNamespace(diagnostic_talker_seed=1234)

    def sample_once() -> tuple[float, float, float]:
        with server_module._diagnostic_talker_rng_scope(engine) as seed:
            assert seed == 1234
            return random.random(), float(np.random.random()), float(torch.rand(()))

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = [
        (random.random(), float(np.random.random()), float(torch.rand(())))
        for _ in range(2)
    ]

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    observed = [sample_once(), sample_once()]

    assert observed == expected
    assert observed[0] != observed[1]


def test_turnhead_does_not_treat_unprojected_listener_hidden_as_v9(monkeypatch) -> None:
    monkeypatch.setattr("duplex_kernel.turnhead_adapter.TurnHeadRuntime", _TurnHeadRuntimeStub)
    monkeypatch.setattr(server_module, "TURNHEAD_CKPT", "stub")
    engine = _Engine(hidden_size=2048)
    engine.default_turn_policy_factory = None
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {"auto_response": True, "turn_policy": "turnhead"},
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]
        _send_pcm(ws)

        assert _wait_for(lambda: session.listener_state.listener_input_frames >= 1)
        time.sleep(0.03)
        assert session.turn_policy.runtime.hidden_calls == []
        assert engine.transcriptions == []
        ws.send_json({"type": "session.close"})


def test_turnhead_policy_speak_barge_in_wires_v9_hidden_and_generation_fences(monkeypatch) -> None:
    monkeypatch.setattr("duplex_kernel.turnhead_adapter.TurnHeadRuntime", _TurnHeadRuntimeStub)
    monkeypatch.setattr(server_module, "TURNHEAD_CKPT", "stub")
    engine = _Engine(hidden_size=4, block_transcription=True)
    engine.default_turn_policy_factory = None
    engine.v9_stream = _V9StreamStub()
    engine.device = "cpu"
    engine.stop_generation_calls = 0
    engine.stop_generation = lambda: setattr(engine, "stop_generation_calls", engine.stop_generation_calls + 1)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        default_policy = session.turn_policy
        assert default_policy.loaded is True
        ws.send_json({
            "type": "session.update",
            "session": {"auto_response": True, "turn_policy": "turnhead"},
        })
        assert ws.receive_json()["type"] == "session.updated"
        policy = session.turn_policy
        assert policy is default_policy, "the demo update must reuse the already-loaded session head"
        policy_inputs = []
        policy_hidden_updates = []
        original_step = policy.step
        original_set_thinker_hidden = policy.runtime.set_thinker_hidden

        def record_policy_input(policy_input):
            policy_inputs.append(policy_input)
            return original_step(policy_input)

        def record_policy_hidden(hidden):
            if hidden is not None:
                policy_hidden_updates.append(torch.as_tensor(hidden).detach().cpu().clone())
            original_set_thinker_hidden(hidden)

        policy.step = record_policy_input
        policy.runtime.set_thinker_hidden = record_policy_hidden

        _send_pcm(ws)
        assert engine.transcription_started.wait(timeout=1.0)
        _send_pcm(ws)
        events = _receive_until(ws, "response.cancelled")
        engine.release_transcription.set()

        assert len(engine.transcriptions) == 1, "SPEAK deve lanciare il worker ASR una sola volta"
        assert engine.stop_generation_calls >= 1, "BARGE_IN deve interrompere il worker/speaker attivo"
        assert not engine.think_calls and not engine.talk_calls
        assert events[-1]["reason"] == "hard_stop"
        assert len(policy_inputs) >= 2
        first_input, second_input = policy_inputs[:2]
        assert first_input.speaker_active is False
        assert second_input.speaker_active is True
        assert second_input.generation_id > first_input.generation_id
        assert policy.runtime.predictions[:2] == ["SPEAK", "BARGE_IN"]
        assert len(policy_hidden_updates) >= 2
        first_hidden, second_hidden = policy_hidden_updates[:2]
        assert tuple(first_hidden.shape) == (1, 2048)
        assert tuple(second_hidden.shape) == (1, 2048)
        assert float(first_hidden[0, 0]) == 1.0
        assert float(second_hidden[0, 0]) == 2.0
        assert session.session_generation_id >= 2
        assert session.stale_session_mutations == 0
        assert engine.v9_stream.calls >= 2
        ws.send_json({"type": "session.close"})


def test_empty_asr_turn_applies_frame_cooldown_without_vad(monkeypatch) -> None:
    monkeypatch.setattr("duplex_kernel.turnhead_adapter.TurnHeadRuntime", _TurnHeadRuntimeStub)
    monkeypatch.setattr(server_module, "TURNHEAD_CKPT", "always-speak")
    engine = _Engine(hidden_size=4)
    engine.v9_stream = _V9StreamStub()

    def empty_asr(pcm):
        engine.transcriptions.append(np.array(pcm, dtype=np.float32, copy=True))
        engine.transcription_started.set()
        return {"text": "  "}

    engine.transcribe_audio = empty_asr
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {"auto_response": True, "turn_policy": "turnhead"},
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]

        silence = np.zeros(FRAME_SAMPLES, dtype="<i2").tobytes()
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode("ascii")})
        _receive_until(ws, "response.done")
        assert len(engine.transcriptions) == 1

        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode("ascii")})
        assert _wait_for(lambda: session.listener_state.listener_input_frames >= 2)
        time.sleep(0.05)
        assert len(engine.transcriptions) == 1, "empty ASR must suppress repeated immediate SPEAK/ASR loops"
        assert engine.think_calls == []
        ws.send_json({"type": "session.close"})


def test_duplex_control6_manifest_allows_optional_head_metadata(tmp_path) -> None:
    import json

    from duplex_kernel.duplex_control6 import _read_manifest

    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": 1,
            "control_head": {
                "architecture": "mlp_256_gelu_dropout_0.1",
                "training_note": "candidate metadata",
            },
        }),
        encoding="utf-8",
    )

    loaded = _read_manifest(manifest_path)

    assert loaded["control_head"]["architecture"] == "mlp_256_gelu_dropout_0.1"


def test_duplex_control6_manifest_rejects_only_explicit_abi_conflicts(tmp_path) -> None:
    import json

    from duplex_kernel.duplex_control6 import _read_manifest
    from training.duplex_qlora.schema import ACTIONS

    valid = {
        "schema_version": 1,
        "control_head": {"input_dim": 1536, "pooling": "last_audio_query"},
    }
    missing_path = tmp_path / "missing_redundant_contract.json"
    missing_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert _read_manifest(missing_path)["schema_version"] == 1

    mismatches = (
        ("input_dim", 2048),
        ("pooling", "mean_audio_queries"),
        ("actions", list(reversed(ACTIONS))),
    )
    for field, actual in mismatches:
        manifest = json.loads(json.dumps(valid))
        manifest["control_head"][field] = actual
        path = tmp_path / f"mismatch_{field}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            _read_manifest(path)


def test_duplex_control6_manifest_architecture_must_match_loaded_head(tmp_path, monkeypatch) -> None:
    import json

    import duplex_kernel.duplex_control6 as control6

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    head_path = tmp_path / "control_head.pt"
    head_path.touch()
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": 1,
            "base_model_path": str(base),
            "adapter_dir": str(adapter),
            "control_head_path": str(head_path),
            "control_head": {"architecture": "mlp_256_gelu_dropout_0.1"},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(control6, "merge_gemma_adapter_into_thinker", lambda *_args: str(base))
    monkeypatch.setattr(
        control6,
        "load_control_head",
        lambda *_args, **_kwargs: SimpleNamespace(architecture="linear"),
    )
    thinker = SimpleNamespace(model=SimpleNamespace(_gemma4_checkpoint_path=str(base)))

    with pytest.raises(ValueError, match="architettura dichiarata.*head caricata"):
        control6.load_control6_runtime(
            thinker,
            adapter_dir=adapter,
            head_path=head_path,
            manifest_path=manifest_path,
            device="cpu",
        )


def test_cancel_before_first_thinker_token_returns_cancelled_result() -> None:
    from duplex_kernel.engine import PipelineEngine

    output = SimpleNamespace(logits=torch.tensor([[[0.0, 1.0]]]))
    engine = SimpleNamespace(
        _run_thinker_prefill=lambda *_args, **_kwargs: (
            output, [torch.zeros((1, 1, 4))], None, torch.ones((1, 1), dtype=torch.long),
        ),
        thinker_prefill_chunk_tokens=16,
        tokenizer=SimpleNamespace(eos_token_id=1),
        device="cpu",
        dtype=torch.float32,
        listener=None,
    )

    result = PipelineEngine.think(engine, [1], max_new_tokens=4, stop_check=lambda: True)

    assert result["cancelled"] is True
    assert result["gen_ids"] == []
    assert result["text"] == ""


def test_control6_skips_policy_when_listener_result_belongs_to_stale_generation(monkeypatch) -> None:
    class _StrictControlPolicy:
        def __init__(self) -> None:
            self.inputs = []

        def step(self, state):
            self.inputs.append(state)
            if state.listener_audio_queries is None:
                raise RuntimeError("test policy received no current native listener queries")
            return PolicyDecision(
                action=ControlAction.LISTEN,
                frame_id=state.frame_id,
                generation_id=state.generation_id,
                listener_frame_id=state.listener_frame_id,
            )

        def reset(self) -> None:
            pass

    monkeypatch.setattr(server_module, "DuplexControl6Policy", _StrictControlPolicy)
    engine = _Engine()
    policy = _StrictControlPolicy()
    engine.default_turn_policy_factory = lambda: policy
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        listener_calls = 0

        def listener_step(_pcm, _reference, _state):
            nonlocal listener_calls
            listener_calls += 1
            frame_id = listener_calls - 1
            if listener_calls == 1:
                # Model concurrent failed-response cleanup while the current
                # native listener frame is being processed.
                session.invalidate_generation("test_worker_cleanup_race")
            return SimpleNamespace(
                frame_id=frame_id,
                listener_state_version=frame_id + 1,
                finite=True,
                gemma_hidden_t=torch.zeros((1, 4)),
                gemma_audio_queries_t=torch.ones((4, 1536)),
            )

        session.listener.step = listener_step
        _send_pcm(ws)
        assert _wait_for(lambda: session.audio_ingress.stats()["input_frames_consumed"] >= 1)
        time.sleep(0.05)

        assert session.scheduler.is_running
        assert session.scheduler.error is None
        assert policy.inputs == [], "a listener result invalidated mid-step must not reach Control6"

        _send_pcm(ws)
        decision = _receive_action_decision(ws, "LISTEN")
        assert decision["accepted"] is True
        assert len(policy.inputs) == 1
        assert policy.inputs[0].frame_id == 1
        assert tuple(policy.inputs[0].listener_audio_queries.shape) == (4, 1536)
        ws.send_json({"type": "session.close"})


def test_v2_response_bundle_defaults_to_control6_and_reports_loaded_head(monkeypatch) -> None:
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
    from training.duplex_qlora.schema import ACTIONS

    class _ControlPolicy:
        policy_id = "duplex_control6"

        def __init__(self, runtime):
            self.runtime = runtime

        def reset(self):
            pass

    runtime = {
        "bundle_id": "candidate-f-test",
        "actions": list(ACTIONS),
        "feature_contract": {"input_dim": 1540, "queries_per_tick": 4},
        "head": SimpleNamespace(architecture="native_phase_fusion_mlp", input_dim=1540),
        "control_head": "/bundle/control_head.pt",
        "base_path": "/models/gemma-test",
        "adapter_path": "/bundle/adapter",
        "base_model_id": "gemma/test",
        "feature_source": {},
        "runtime_abi": {},
    }
    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    monkeypatch.setattr(server_module, "DuplexControl6Policy", _ControlPolicy)
    engine = _Engine()
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.enable_duplex_control6 = True
    engine.duplex_response_bundle_dir = "candidate-f-test"
    engine.duplex_control6_runtime = runtime
    engine.talker = SimpleNamespace(
        tcfg=SimpleNamespace(hidden_size=4),
        default_voice_conditioning=None,
    )
    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {"mode": "x_vector_only", "speaker_embedding": torch.ones(4)},
    )
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["turn_policy"] == "duplex_control6"
        assert created["policy_id"] == "duplex_control6"
        assert created["bundle_id"] == "candidate-f-test"
        assert created["control_head_architecture"] == "native_phase_fusion_mlp"
        assert created["control_head_input_dim"] == 1540
        assert created["policy_actions"] == list(ACTIONS)
        assert created["native_action_selection"] == "wake_scoped"
        assert isinstance(sessions.all()[0].turn_policy, _ControlPolicy)
        ws.send_json({
            "type": "session.update",
            "session": {
                "auto_response": True,
                "voice": "gemma",
                "voice_conditioning_mode": "x_vector_only",
                "turn_policy": "duplex_control6",
            },
        })
        loaded = ws.receive_json()
        updated = ws.receive_json()
        assert loaded["type"] == "model_loaded"
        assert loaded["policy_id"] == "duplex_control6"
        assert loaded["control_head_architecture"] == "native_phase_fusion_mlp"
        assert loaded["control_head_input_dim"] == 1540
        assert updated["type"] == "session.updated"
        assert updated["turn_policy"] == "duplex_control6"
        assert updated["bundle_id"] == "candidate-f-test"
        assert updated["control_head_input_dim"] == 1540
        assert updated["native_action_selection"] == "wake_scoped"
        assert client.get("/health").json()["native_action_selection"] == "wake_scoped"
        ws.send_json({"type": "session.close"})


def test_v2_explicit_turnhead_selection_is_preserved(monkeypatch) -> None:
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2

    class _ControlPolicy:
        policy_id = "duplex_control6"

        def __init__(self, runtime):
            self.runtime = runtime

        def reset(self):
            pass

    class _LegacyPolicy:
        policy_id = "turnhead"
        loaded = True

        def __init__(self, **_kwargs):
            pass

        def reset(self):
            pass

    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    monkeypatch.setattr(server_module, "DuplexControl6Policy", _ControlPolicy)
    monkeypatch.setattr(server_module, "TurnHeadPolicy", _LegacyPolicy)
    engine = _Engine()
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.enable_duplex_control6 = True
    engine.duplex_response_bundle_dir = "candidate-f-test"
    engine.duplex_control6_runtime = {
        "bundle_id": "candidate-f-test",
        "actions": ["LISTEN", "CONTINUE", "SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "feature_contract": {"input_dim": 1540},
        "head": SimpleNamespace(architecture="native_phase_fusion_mlp", input_dim=1540),
        "control_head": "/bundle/control_head.pt",
    }
    client = TestClient(create_app(engine, SessionManager()))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        created = ws.receive_json()
        assert created["turn_policy"] == "duplex_control6"
        ws.send_json({"type": "session.update", "session": {"turn_policy": "turnhead"}})
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["turn_policy"] == "turnhead"
        assert updated["policy_id"] == "turnhead"
        ws.send_json({"type": "session.close"})


def test_v2_does_not_select_control6_when_bundle_head_is_not_loaded(monkeypatch) -> None:
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2

    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    engine = _Engine()
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.enable_duplex_control6 = True
    engine.duplex_response_bundle_dir = "candidate-f-not-loaded"
    engine.duplex_control6_runtime = {
        "bundle_id": "candidate-f-not-loaded",
        "head": None,
        "actions": ["LISTEN", "CONTINUE", "SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "feature_contract": {"input_dim": 1540},
    }
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["turn_policy"] == "none"
        assert not isinstance(sessions.all()[0].turn_policy, server_module.DuplexControl6Policy)
        ws.send_json({"type": "session.close"})


def test_v2_invalid_explicit_turn_policy_is_rejected_without_fallback(monkeypatch) -> None:
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2

    class _ControlPolicy:
        policy_id = "duplex_control6"

        def __init__(self, runtime):
            self.runtime = runtime

        def reset(self):
            pass

    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    monkeypatch.setattr(server_module, "DuplexControl6Policy", _ControlPolicy)
    engine = _Engine()
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.enable_duplex_control6 = True
    engine.duplex_response_bundle_dir = "candidate-f-test"
    engine.duplex_control6_runtime = {
        "bundle_id": "candidate-f-test",
        "actions": ["LISTEN", "CONTINUE", "SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "feature_contract": {"input_dim": 1540},
        "head": SimpleNamespace(architecture="native_phase_fusion_mlp", input_dim=1540),
        "control_head": "/bundle/control_head.pt",
    }
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["turn_policy"] == "duplex_control6"
        ws.send_json({"type": "session.update", "session": {"turn_policy": "mystery_policy"}})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == "INVALID_SESSION_UPDATE"
        assert "turn_policy non valida" in error["error"]["message"]
        assert isinstance(sessions.all()[0].turn_policy, _ControlPolicy)


def test_realtime_ui_explicitly_requests_duplex_control6_policy() -> None:
    from pathlib import Path

    html = (Path(__file__).parents[2] / "server" / "static" / "index.html").read_text(encoding="utf-8")
    update_start = html.index('type: "session.update"')
    update_end = html.index("});", update_start)
    assert 'turn_policy: "duplex_control6"' in html[update_start:update_end]


def test_response_worker_failure_preserves_current_listener_frame_ack(monkeypatch) -> None:
    engine = _Engine()
    response_started = threading.Event()
    fail_response = threading.Event()
    listener_frame_blocked = threading.Event()
    release_listener_frame = threading.Event()
    ack_events: list[dict[str, object]] = []
    original_put = server_module._SchedulerEventBuffer.put_nowait

    def observe_scheduler_event(self, event):
        if event.get("type") == "input_audio_buffer.consumed":
            ack_events.append(dict(event))
        return original_put(self, event)

    monkeypatch.setattr(server_module._SchedulerEventBuffer, "put_nowait", observe_scheduler_event)

    def fail_after_listener_starts(*_args, **_kwargs):
        response_started.set()
        if not fail_response.wait(timeout=2.0):
            raise TimeoutError("test response failure was not released")
        error = RuntimeError("synthetic Thinker failure")
        error.terminal_reason = "eos"
        error.terminal_token_id = 106
        error.runtime_metrics = {
            "thinker_decode_steps": 1,
            "terminal_reason": "eos",
            "terminal_token_id": 106,
        }
        raise error

    engine.think = fail_after_listener_starts
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    try:
        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({
                "type": "session.update",
                "session": {
                    "turn_policy": "scripted",
                    "scripted_actions": ["BACKCHANNEL", "LISTEN"],
                    "input_audio_buffer_acknowledgements": True,
                },
            })
            assert ws.receive_json()["type"] == "session.updated"
            session = sessions.all()[0]
            original_step = session.listener.step
            listener_calls = 0

            def block_second_listener_frame(pcm, reference, state):
                nonlocal listener_calls
                listener_calls += 1
                output = original_step(pcm, reference, state)
                if listener_calls == 2:
                    listener_frame_blocked.set()
                    if not release_listener_frame.wait(timeout=2.0):
                        raise TimeoutError("test listener frame was not released")
                return output

            session.listener.step = block_second_listener_frame
            _send_pcm(ws)
            assert response_started.wait(timeout=1.0)
            _send_pcm(ws)
            assert listener_frame_blocked.wait(timeout=1.0)

            fail_response.set()
            error_event = _receive_until(ws, "error")[-1]
            assert error_event["type"] == "error"
            release_listener_frame.set()
            assert _wait_for(lambda: any(event["frame_id"] == 1 for event in ack_events))
            assert error_event["response_id"]
            assert error_event["error"]["terminal_reason"] == "eos"
            assert error_event["error"]["terminal_token_id"] == 106
            assert error_event["runtime_metrics"]["thinker"]["terminal_token_id"] == 106
            assert session.scheduler.is_running
            assert session.scheduler.error is None
            assert session.last_listener_frame_id == 1
    finally:
        fail_response.set()
        release_listener_frame.set()
def test_listen_keeps_active_pcm_stream_and_yield_stops_talker(monkeypatch) -> None:
    monkeypatch.delenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", raising=False)
    engine = _Engine(hidden_size=2048)
    talk_started = threading.Event()
    release_talker = threading.Event()
    late_chunk_emitted = threading.Event()

    def blocking_talk(text, hidden, *, on_audio_chunk=None, **_kwargs):
        engine.talk_calls.append((text, hidden))
        wav = np.full(1920 * 8, 0.1, dtype=np.float32)
        if on_audio_chunk is not None:
            on_audio_chunk(wav)
            on_audio_chunk(wav)  # advance the causal playout horizon so PCM is actually released
        talk_started.set()
        if not release_talker.wait(timeout=3.0):
            raise TimeoutError("test backchannel talker was not released")
        if on_audio_chunk is not None and not session.stop_requested:
            on_audio_chunk(wav)
            late_chunk_emitted.set()
        return wav

    engine.talk = blocking_talk
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    try:
        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({
                "type": "session.update",
                "session": {
                    "turn_policy": "scripted",
                    "scripted_actions": ["BACKCHANNEL", "LISTEN", "YIELD"],
                },
            })
            assert ws.receive_json()["type"] == "session.updated"
            session = sessions.all()[0]
            _send_pcm(ws)
            assert talk_started.wait(timeout=1.0), "BACKCHANNEL must reach the speaker while the user is speaking"
            assert session.control_state.value == "BACKCHANNELING"
            audio_events = _receive_until(ws, "response.audio.delta")
            assert audio_events[-1]["type"] == "response.audio.delta"
            generation_before_listen = session.session_generation_id

            _send_pcm(ws)
            listen = _receive_action_decision(ws, "LISTEN")
            assert listen["accepted"] is True
            assert session.active_response is True
            assert session.session_generation_id == generation_before_listen
            assert session.control_state.value == "BACKCHANNELING"
            assert session.stop_requested is False

            # Drain the prior speech run, then provide a fresh post-watermark
            # onset so the server's causal YIELD admission gate accepts it.
            for _ in range(12):
                _send_pcm_level(ws, 0)
                _receive_action_decision(ws, "YIELD")
            yielded = None
            for _ in range(8):
                _send_pcm(ws)
                candidate = _receive_action_decision(ws, "YIELD")
                if candidate["accepted"]:
                    yielded = candidate
                    break
            assert yielded is not None, 'YIELD should stop active Talker after a fresh user onset'
            assert session.stop_requested is True
            assert session.stop_reason == "YIELD"
            assert session.session_generation_id == generation_before_listen

            release_talker.set()
            terminal = _receive_until_any(ws, {"response.done", "response.cancelled"})
            assert terminal[-1]["type"] == "response.done"
            assert not late_chunk_emitted.is_set(), "YIELD must stop future Talker PCM"
            # YIELD drains PCM accepted before the boundary; the blocked Talker callback above proves no later chunk was produced.
            assert _wait_for(lambda: not session.active_response)
            assert session.scheduler.is_running
    finally:
        release_talker.set()


def test_backchannel_requires_recent_voiced_input() -> None:
    engine = _Engine(hidden_size=2048)
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {"turn_policy": "scripted", "scripted_actions": ["BACKCHANNEL", "BACKCHANNEL"]},
        })
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]

        # Near-zero PCM is still a real input frame, but it is not evidence of
        # current speech and must not start an empty Backchannel Thinker turn.
        _send_pcm_level(ws, 0)
        quiet_backchannel = _receive_action_decision(ws, "BACKCHANNEL")
        assert quiet_backchannel["frame_id"] == 0
        assert quiet_backchannel["accepted"] is False
        assert quiet_backchannel["rejection_reason"] == "no_recent_user_audio"
        assert engine.think_calls == []

        _send_pcm_level(ws, 4096)
        voiced_backchannel = _receive_action_decision(ws, "BACKCHANNEL")
        assert voiced_backchannel["frame_id"] == 1
        assert voiced_backchannel["accepted"] is True
        engine.release_transcription.set()
        assert _wait_for(lambda: not session.active_response, timeout=1.0)
        ws.send_json({"type": "session.close"})


def test_yield_requires_emitted_audio_and_post_watermark_user_voice() -> None:
    assert _yield_admission_rejection(
        active_response=True,
        emitted_audio_frames=0,
        has_post_watermark_voice=True,
    ) == "assistant_audio_not_started"
    assert _yield_admission_rejection(
        active_response=True,
        emitted_audio_frames=2,
        has_post_watermark_voice=False,
    ) == "no_new_user_audio"
    assert _yield_admission_rejection(
        active_response=True,
        emitted_audio_frames=2,
        has_post_watermark_voice=True,
    ) is None


def test_response_audio_watermark_and_sent_frames_share_one_causality_snapshot() -> None:
    policy_lock = threading.Lock()
    admission = _ResponseAudioAdmissionState()
    tracker = _CausalUserSpeechTracker(endpoint_quiet_frames=8)
    voiced = np.full(FRAME_SAMPLES, 0.04, dtype=np.float32)

    with policy_lock:
        admission.update_input_frame_locked(40)
        assert admission.snapshot_locked(tracker) == (0, False)
        admission.mark_audio_sent_locked(2)
        assert admission.input_watermark == 40
        assert admission.snapshot_locked(tracker) == (2, False)
        # The scheduler updates speech evidence under this same lock. A tick
        # sees either the pre-send pair or the coherent sent-frame pair.
        tracker.observe(41, voiced)
        assert admission.snapshot_locked(tracker) == (2, True)
        admission.reset_locked()
        assert admission.snapshot_locked(tracker) == (0, False)


def test_causal_voice_tracker_defers_endpoint_and_distinguishes_new_onset() -> None:
    tracker = _CausalUserSpeechTracker(endpoint_quiet_frames=10)
    voiced = np.full(FRAME_SAMPLES, 0.04, dtype=np.float32)
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    assert tracker.observe(0, voiced) == (True, False)
    for frame_id in range(1, 5):
        assert tracker.observe(frame_id, silence) == (False, False)
    assert tracker.voice_run_active is True
    # A 400 ms pause followed by resumed speech belongs to the same causal
    # utterance and does not produce a second onset or SPEAK endpoint.
    assert tracker.observe(5, voiced) == (False, False)
    assert tracker.run_onset_frame_id == 0
    for frame_id in range(6, 15):
        onset, endpoint = tracker.observe(frame_id, silence)
        assert (onset, endpoint) == (False, False), "nine quiet ticks must not close an 800 ms endpoint"
    assert tracker.voice_run_active is True
    onset, endpoint = tracker.observe(15, silence)
    assert (onset, endpoint) == (False, True)
    assert tracker.speech_complete is True
    assert tracker.endpoint_frame_id == 15
    assert tracker.has_onset_after(4) is False
    assert tracker.has_onset_after(5) is False
    tracker.consume_endpoint()
    assert tracker.speech_complete is False

    assert tracker.observe(16, voiced) == (True, False)
    assert tracker.run_onset_frame_id == 16
    assert tracker.has_onset_after(15) is True


def test_deferred_native_speak_is_fenced_to_original_utterance_on_new_onset() -> None:
    deferred = {
        "source": "thinker_response_action_deferred_to_endpoint",
        "source_frame_id": 0,
        "voice_run_onset_frame_id": 0,
        "endpoint_frame_id": 10,
    }
    pending, change = server_module._advance_pending_speak_on_new_onset(deferred, 11)
    assert pending is None
    assert change == "dropped_stale_deferred_native_intent"

    # The existing listener-owned pending-SPEAK behavior may still follow a
    # resumed run; only the v2 Thinker choice is immutable to its source span.
    listener_pending = {
        **deferred,
        "source": "turn_policy",
    }
    retargeted, change = server_module._advance_pending_speak_on_new_onset(listener_pending, 11)
    assert change == "retargeted_pending_speak"
    assert retargeted == {
        **listener_pending,
        "voice_run_onset_frame_id": 11,
        "endpoint_frame_id": None,
    }


def test_default_speak_endpoint_uses_ten_80ms_ticks(monkeypatch) -> None:
    monkeypatch.delenv("DUPLEX_SPEAK_ENDPOINT_QUIET_MS", raising=False)
    assert server_module._configured_speak_endpoint_frames() == 10


def test_policy_audio_endpoint_snapshot_excludes_later_frames() -> None:
    session = DuplexSession("policy-endpoint-snapshot")
    frame0 = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)
    frame1 = np.full(FRAME_SAMPLES, 0.2, dtype=np.float32)
    frame2 = np.full(FRAME_SAMPLES, 0.3, dtype=np.float32)
    session.append_policy_audio(10, frame0)
    session.append_policy_audio(11, frame1)
    session.append_policy_audio(12, frame2)

    correction = session.take_policy_audio_through_frame(11)
    later = session.take_policy_audio()

    np.testing.assert_array_equal(correction, np.concatenate((frame0, frame1)))
    np.testing.assert_array_equal(later, frame2)
    assert _yield_admission_rejection(
        active_response=False,
        emitted_audio_frames=0,
        has_post_watermark_voice=False,
    ) is None


def test_active_direct_audio_snapshot_starts_after_consumed_onset_frame() -> None:
    session = DuplexSession("direct-snapshot-consumed-onset")
    frame = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)
    session.append_policy_audio(302, frame)
    np.testing.assert_array_equal(session.take_policy_audio_through_frame(302), frame)
    assert session.policy_audio_consumed_through_frame_id == 302
    session.reset_policy_audio(committed_sample_count=303 * FRAME_SAMPLES)
    assert session.policy_audio_consumed_through_frame_id == 302
    for frame_id in range(303, 431):
        session.append_policy_audio(frame_id, np.full(FRAME_SAMPLES, frame_id, dtype=np.float32))

    snapshot = server_module._snapshot_direct_policy_audio(
        session,
        action=ControlAction.BACKCHANNEL,
        first_frame_id=302,
        last_frame_id=430,
    )

    assert snapshot["_direct_audio_requested_first_frame_id"] == 302
    assert snapshot["_direct_audio_first_frame_id"] == 303
    assert snapshot["_direct_audio_sample_count"] == 128 * FRAME_SAMPLES
    assert snapshot["_direct_audio_prefix_frames_clipped"] == 0
    assert snapshot["_direct_audio_consumed_prefix_frames"] == 1
    np.testing.assert_array_equal(
        snapshot["_direct_audio_pcm"],
        np.concatenate([
            np.full(FRAME_SAMPLES, frame_id, dtype=np.float32)
            for frame_id in range(303, 431)
        ]),
    )


def test_policy_audio_consumption_watermark_includes_frame_zero_and_reset_discards() -> None:
    session = DuplexSession("policy-consumed-watermark-zero")
    frame = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)
    session.append_policy_audio(0, frame)
    np.testing.assert_array_equal(session.take_policy_audio(), frame)
    assert session.policy_audio_consumed_through_frame_id == 0

    session.append_policy_audio(1, frame)
    session.reset_policy_audio(committed_sample_count=2 * FRAME_SAMPLES)
    assert session.policy_audio_consumed_through_frame_id == 1


def test_active_direct_audio_snapshot_still_rejects_empty_or_noncontiguous_pcm() -> None:
    session = DuplexSession("direct-snapshot-fail-closed")
    session.append_policy_audio(302, np.full(FRAME_SAMPLES, 0.1, dtype=np.float32))
    session.take_policy_audio_through_frame(302)

    with pytest.raises(ValueError, match="snapshot PCM e span causale non allineati"):
        server_module._snapshot_direct_policy_audio(
            session,
            action=ControlAction.BACKCHANNEL,
            first_frame_id=302,
            last_frame_id=302,
        )

    for frame_id in range(303, 431):
        if frame_id != 350:
            session.append_policy_audio(frame_id, np.full(FRAME_SAMPLES, 0.1, dtype=np.float32))
    with pytest.raises(ValueError, match="snapshot PCM e span causale non allineati"):
        server_module._snapshot_direct_policy_audio(
            session,
            action=ControlAction.BACKCHANNEL,
            first_frame_id=302,
            last_frame_id=430,
        )


def test_output_queue_backpressure_waits_for_consumer_and_preserves_burst_metadata() -> None:
    output_queue = queue.Queue(maxsize=1)
    output_queue.put_nowait("older")
    result: dict[str, object] = {}

    def try_put() -> bool | None:
        try:
            output_queue.put_nowait("new")
        except queue.Full:
            return None
        return True

    worker = threading.Thread(
        target=lambda: result.setdefault(
            "write",
            server_module._put_response_queue_bounded(
                try_put,
                queue_kind="audio",
                burst_items=2,
                timeout_s=0.5,
                retry_s=0.005,
            ),
        )
    )
    worker.start()
    time.sleep(0.03)
    assert worker.is_alive(), "producer should wait while bounded output queue is full"
    assert output_queue.get(timeout=1.0) == "older"
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert output_queue.get_nowait() == "new"
    metrics = result["write"]
    assert metrics["status"] == "enqueued"
    assert metrics["queue_kind"] == "audio"
    assert metrics["burst_items"] == 2
    assert metrics["wait_ms"] > 0


def test_output_queue_backpressure_cancels_while_waiting_without_enqueuing_stale_item() -> None:
    output_queue = queue.Queue(maxsize=1)
    output_queue.put_nowait("older")
    cancelled = threading.Event()
    result: dict[str, object] = {}

    def try_put() -> bool | None:
        if cancelled.is_set():
            return False
        try:
            output_queue.put_nowait("stale")
        except queue.Full:
            return None
        return True

    worker = threading.Thread(
        target=lambda: result.setdefault(
            "write",
            server_module._put_response_queue_bounded(
                try_put,
                queue_kind="audio",
                burst_items=1,
                timeout_s=0.5,
                retry_s=0.005,
            ),
        )
    )
    worker.start()
    time.sleep(0.03)
    cancelled.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result["write"]["status"] == "cancelled"
    assert result["write"]["cancel_reason"] == "generation_fenced"
    assert output_queue.get_nowait() == "older"
    assert output_queue.empty()


def test_output_queue_backpressure_timeout_identifies_saturated_queue() -> None:
    output_queue = queue.Queue(maxsize=1)
    output_queue.put_nowait("older")

    def try_put() -> bool | None:
        try:
            output_queue.put_nowait("new")
        except queue.Full:
            return None
        return True

    with pytest.raises(server_module.ResponseOutputQueueFull) as caught:
        server_module._put_response_queue_bounded(
            try_put,
            queue_kind="text",
            burst_items=3,
            timeout_s=0.02,
            retry_s=0.003,
        )

    assert caught.value.queue_kind == "text"
    assert caught.value.burst_items == 3
    assert caught.value.wait_ms >= 20
    assert output_queue.get_nowait() == "older"


def test_session_voice_enforcement_uses_training_default_through_response_resolution(monkeypatch, tmp_path) -> None:
    engine = _Engine()
    engine.enforce_v9_training_voice = True
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"same training reference")
    training_default = {
        "mode": "x_vector_only",
        "speaker_embedding": torch.ones(4),
        "reference_wav_path": str(reference_path),
        "reference_wav_sha256": "480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c",
        "ref_code": None,
    }
    engine.talker = SimpleNamespace(
        tcfg=SimpleNamespace(hidden_size=4),
        default_voice_conditioning=training_default,
    )
    monkeypatch.setattr(
        server_module,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: pytest.fail("enforced training voice must bypass named profiles"),
    )
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({
            "type": "session.update",
            "session": {
                "auto_response": True,
                "voice": "gemma",
                "voice_conditioning_mode": "x_vector_only",
            },
        })
        updated = ws.receive_json()
        session = sessions.all()[0]

        assert updated["type"] == "session.updated"
        assert updated["voice_conditioning_mode"] == "x_vector_only"
        assert updated["conditioning_source"] == "training_default"
        assert updated["reference_wav_path"] == training_default["reference_wav_path"]
        assert updated["reference_wav_sha256"] == training_default["reference_wav_sha256"]
        assert session.voice_conditioning is not training_default
        assert session.voice_conditioning["conditioning_source"] == "training_default"
        assert session.voice_conditioning["speaker_embedding"] is training_default["speaker_embedding"]
        assert server_module._resolve_request_voice_conditioning(
            engine,
            session,
            {"voice": "gemma", "voice_style": "custom"},
        ) is session.voice_conditioning
        session.voice_conditioning = dict(session.voice_conditioning)
        session.voice_conditioning["reference_wav_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="training default V9"):
            server_module._resolve_request_voice_conditioning(
                engine,
                session,
                {"voice": "gemma"},
            )
        ws.send_json({"type": "session.close"})


def test_replan_during_stream_is_control_only_and_next_speak_is_allowed() -> None:
    engine = _Engine(block_talk=True)

    def talk_with_streamable_packet(
        text,
        h_pfx,
        *,
        on_audio_chunk_with_horizon=None,
        **_kwargs,
    ):
        engine.talk_calls.append((text, h_pfx))
        wav = np.full(4 * 1920, 0.1, dtype=np.float32)
        if on_audio_chunk_with_horizon is not None:
            on_audio_chunk_with_horizon(wav, 10)
        engine.talk_started.set()
        if not engine.release_talk.wait(timeout=3.0):
            raise TimeoutError("test talk worker was not released")
        return wav

    engine.talk_stateful = talk_with_streamable_packet
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))
    actions = ["LISTEN"] * 10 + ["SPEAK", "REPLAN"] + ["CONTINUE"] * 20 + ["SPEAK"]

    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {
            "turn_policy": "scripted",
            "scripted_actions": actions,
        }})
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]

        for frame_id in range(10):
            _send_pcm_level(ws, 4096 if frame_id == 0 else 0)
            assert _receive_action_decision(ws, "LISTEN")["frame_id"] == frame_id
        _send_pcm_level(ws, 0)
        initial_events = _receive_until_any(ws, {"response.created", "error"})
        assert initial_events[-1]["type"] == "response.created", initial_events
        first_response_id = initial_events[-1]["response_id"]
        assert engine.talk_started.wait(timeout=1.0)
        first_audio = _receive_until(ws, "response.audio.delta")
        assert any(event.get("response_id") == first_response_id for event in first_audio)

        _send_pcm_level(ws, 8192)
        cancel_events = _receive_until_any(ws, {"response.cancelled", "error"})
        assert cancel_events[-1]["type"] == "response.cancelled", cancel_events
        assert cancel_events[-1]["reason"] == "replan"
        decision = next(
            event for event in cancel_events
            if event.get("type") == "duplex_control6.decision"
            and event.get("action") == "REPLAN"
        )
        assert decision["accepted"] is True
        engine.release_talk.set()

        between_events: list[dict[str, object]] = []
        for frame_id in range(12, 22):
            _send_pcm_level(ws, 0)
            for _ in range(30):
                event = ws.receive_json()
                between_events.append(event)
                if (
                    event.get("type") == "duplex_control6.decision"
                    and event.get("frame_id") == frame_id
                ):
                    assert event["action"] == "CONTINUE"
                    break
            else:
                raise AssertionError(f"decision for frame {frame_id} not received: {between_events!r}")

        assert _wait_for(lambda: not session.policy_response_pending)
        assert not any(event.get("type") == "error" for event in between_events)
        assert not any(event.get("type") == "response.created" for event in between_events)
        assert len(engine.talk_calls) == 1, "REPLAN must never invoke the Talker"

        _send_pcm_level(ws, 8192)
        assert _receive_action_decision(ws, "CONTINUE")["frame_id"] == 22
        for frame_id in range(23, 32):
            _send_pcm_level(ws, 0)
            assert _receive_action_decision(ws, "CONTINUE")["frame_id"] == frame_id
        _send_pcm_level(ws, 0)
        next_speak_events = _receive_until_any(ws, {"response.created", "error"})
        assert not any(event.get("type") == "error" for event in next_speak_events)
        next_speak = next(
            event for event in next_speak_events
            if event.get("type") == "duplex_control6.decision"
            and event.get("action") == "SPEAK"
        )
        assert next_speak["accepted"] is True, next_speak
        assert next_speak_events[-1]["type"] == "response.created", next_speak_events
        assert _wait_for(lambda: len(engine.talk_calls) == 2)
        ws.send_json({"type": "session.close"})
