from __future__ import annotations

import base64
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from duplex_kernel.listener import (
    AssistantReferenceBuffer,
    AssistantReferenceFrame,
    GemmaListener,
    GemmaListenerState,
)
from duplex_kernel.server import create_app
from duplex_kernel.session import SessionManager


FRAME_SAMPLES = 1280


class _RecordingBackend:
    hidden_size = 4

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, AssistantReferenceFrame | None]] = []

    def encode(self, user_pcm: np.ndarray, assistant_reference: AssistantReferenceFrame | None, state: GemmaListenerState) -> np.ndarray:
        self.calls.append((user_pcm.copy(), assistant_reference))
        reference_id = -1 if assistant_reference is None else assistant_reference.frame_id
        return np.full((1, self.hidden_size), state.next_frame_id + reference_id + 2, dtype=np.float32)


def test_native_audio_tower_backend_builds_causal_interleaved_stream() -> None:
    import torch

    class _Extractor:
        def __init__(self) -> None:
            self.calls: list[list[np.ndarray]] = []

        def __call__(self, waves, **_kwargs):
            waves = [np.asarray(wave) for wave in waves]
            self.calls.append(waves)
            return {
                "input_features": [np.full((3, 128), float(index + 1), dtype=np.float32) for index in range(len(waves))],
                "input_features_mask": [np.ones(3, dtype=bool) for _ in waves],
            }

    class _AudioTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    class _GemmaModel:
        def __init__(self) -> None:
            self.audio_tower = _AudioTower()
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=1536))
            self.seen_features = None

        def get_audio_features(self, *, input_features, input_features_mask, return_dict):
            assert return_dict is True
            self.seen_features = input_features.detach().cpu()
            hidden = torch.zeros((*input_features.shape[:2], 1536), dtype=input_features.dtype)
            return SimpleNamespace(pooler_output=hidden, attention_mask=input_features_mask)

    from duplex_kernel.listener import GemmaAudioTowerBackend

    extractor = _Extractor()
    gemma = _GemmaModel()
    wrapper = SimpleNamespace(model=SimpleNamespace(model=gemma), processor=extractor)
    listener = GemmaListener(backend=GemmaAudioTowerBackend(wrapper))
    output = listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), None, GemmaListenerState())

    assert output.hidden_t.shape == (1, 1536)
    assert gemma.seen_features is not None
    assert gemma.seen_features.shape == (1, 6, 128)
    np.testing.assert_array_equal(gemma.seen_features[0, 0].float().numpy(), 1.0)
    np.testing.assert_array_equal(gemma.seen_features[0, 1].float().numpy(), 2.0)
    assert extractor.calls


def test_canonical_gemma_listener_rejects_context_above_attested_window() -> None:
    from duplex_kernel.listener import GemmaAudioTowerBackend

    backend = object.__new__(GemmaAudioTowerBackend)
    backend._lock = threading.Lock()
    backend.multimodal_model = object()
    state = GemmaListenerState(context_frames=65)
    zero = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    state.user_pcm_history.append(zero)
    state.assistant_pcm_history.append(zero)

    with pytest.raises(ValueError, match="massimo attestato"):
        backend.encode(zero, None, state)


def test_state_reset_clears_audio_feature_history_and_tails() -> None:
    state = GemmaListenerState()
    feature = np.ones((8, 128), dtype=np.float32)
    state.user_mel_history.append(feature)
    state.assistant_mel_history.append(feature)
    state.user_pcm_tail = np.ones(160, dtype=np.float32)
    state.assistant_pcm_tail = np.ones(160, dtype=np.float32)

    state.reset()

    assert not state.user_mel_history
    assert not state.assistant_mel_history
    assert state.user_pcm_tail is None
    assert state.assistant_pcm_tail is None


def test_listener_failure_restores_audio_feature_history() -> None:
    class _FailingAfterFeatureBackend:
        def encode(self, _user_pcm, _assistant_reference, state):
            state.user_mel_history.append(np.full((8, 128), 2.0, dtype=np.float32))
            state.assistant_mel_history.append(np.full((8, 128), 3.0, dtype=np.float32))
            state.user_pcm_tail = np.full(160, 4.0, dtype=np.float32)
            state.assistant_pcm_tail = np.full(160, 5.0, dtype=np.float32)
            raise RuntimeError("errore dopo feature")

    state = GemmaListenerState(context_frames=2)
    state.user_pcm_history.extend((np.full(FRAME_SAMPLES, 0.0, dtype=np.float32), np.full(FRAME_SAMPLES, 1.0, dtype=np.float32)))
    state.assistant_pcm_history.extend((np.full(FRAME_SAMPLES, 10.0, dtype=np.float32), np.full(FRAME_SAMPLES, 11.0, dtype=np.float32)))
    original_user = np.full((8, 128), 6.0, dtype=np.float32)
    original_assistant = np.full((8, 128), 7.0, dtype=np.float32)
    state.user_mel_history.append(original_user)
    state.assistant_mel_history.append(original_assistant)
    state.user_pcm_tail = np.full(160, 8.0, dtype=np.float32)
    state.assistant_pcm_tail = np.full(160, 9.0, dtype=np.float32)
    listener = GemmaListener(backend=_FailingAfterFeatureBackend())

    with pytest.raises(RuntimeError, match="errore dopo feature"):
        listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), None, state)

    np.testing.assert_array_equal(list(state.user_pcm_history), [np.full(FRAME_SAMPLES, 0.0), np.full(FRAME_SAMPLES, 1.0)])
    np.testing.assert_array_equal(list(state.assistant_pcm_history), [np.full(FRAME_SAMPLES, 10.0), np.full(FRAME_SAMPLES, 11.0)])
    np.testing.assert_array_equal(state.user_mel_history[0], original_user)
    np.testing.assert_array_equal(state.assistant_mel_history[0], original_assistant)
    np.testing.assert_array_equal(state.user_pcm_tail, np.full(160, 8.0, dtype=np.float32))
    np.testing.assert_array_equal(state.assistant_pcm_tail, np.full(160, 9.0, dtype=np.float32))


def test_listener_failure_discards_partial_listener_lm_cache() -> None:
    class _FailingBackend:
        def encode(self, _user_pcm, _assistant_reference, state):
            state.gemma_lm_cache = object()
            state.gemma_lm_shared_kv_states = {0: object()}
            state.gemma_lm_audio_frames = 7
            state.gemma_lm_context_start_frame = 3
            state.gemma_lm_text_context = (1, 2)
            raise RuntimeError("errore simulato dopo avanzamento KV")

    state = GemmaListenerState()
    listener = GemmaListener(backend=_FailingBackend())

    with pytest.raises(RuntimeError, match="errore simulato"):
        listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), None, state)

    assert state.gemma_lm_cache is None
    assert state.gemma_lm_shared_kv_states is None
    assert state.gemma_lm_audio_frames == 0
    assert state.gemma_lm_context_start_frame == 0
    assert state.gemma_lm_text_context == ()


def test_listener_nonfinite_hidden_discards_partial_listener_lm_cache() -> None:
    class _NonfiniteBackend:
        def encode(self, _user_pcm, _assistant_reference, state):
            state.gemma_lm_cache = object()
            state.gemma_lm_shared_kv_states = {0: object()}
            state.gemma_lm_audio_frames = 1
            return np.array([[np.nan]], dtype=np.float32)

    state = GemmaListenerState()
    listener = GemmaListener(backend=_NonfiniteBackend())

    with pytest.raises(RuntimeError, match="non finito"):
        listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), None, state)

    assert state.gemma_lm_cache is None
    assert state.gemma_lm_shared_kv_states is None
    assert state.gemma_lm_audio_frames == 0


def test_assistant_reference_buffer_exposes_only_one_completed_frame_behind() -> None:
    buffer = AssistantReferenceBuffer()
    frame0 = np.full(FRAME_SAMPLES, 0.25, dtype=np.float32)
    frame1 = np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)

    buffer.append(frame0, frame_id=0, emitted_at_ns=10)
    assert buffer.visible_for(listener_frame_id=0) is None
    assert buffer.visible_for(listener_frame_id=1).frame_id == 0
    assert buffer.visible_for(listener_frame_id=2) is None

    buffer.append(frame1, frame_id=1, emitted_at_ns=20)
    assert buffer.visible_for(listener_frame_id=1).frame_id == 0
    assert buffer.visible_for(listener_frame_id=2).frame_id == 1


def test_assistant_reference_buffer_resamples_emitted_24khz_frame_without_dropping_it() -> None:
    buffer = AssistantReferenceBuffer()
    source = np.linspace(-1.0, 1.0, 1920, dtype=np.float32)

    stored = buffer.append(source, frame_id=3, emitted_at_ns=30, sample_rate=24_000)

    assert stored.frame_id == 3
    assert stored.pcm.shape == (FRAME_SAMPLES,)
    assert stored.sample_rate == 16_000
    np.testing.assert_allclose(stored.pcm[[0, -1]], source[[0, -1]], atol=1e-6)


def test_emitted_reference_ids_follow_listener_timeline_after_midstream_start() -> None:
    session = SessionManager().create_session()

    assert session.reserve_assistant_reference_frames(2) == 0
    session.mark_listener_frame(10)
    assert session.reserve_assistant_reference_frames(1) == 10
    session.mark_listener_frame(11)
    assert session.reserve_assistant_reference_frames(1) == 11


def test_listener_accounts_every_frame_and_increments_version_after_success() -> None:
    backend = _RecordingBackend()
    listener = GemmaListener(backend=backend)
    state = GemmaListenerState()

    outputs = [listener.step(np.full(FRAME_SAMPLES, i, dtype=np.float32), None, state) for i in range(5)]

    assert len(backend.calls) == 5
    assert [output.frame_id for output in outputs] == list(range(5))
    assert [output.listener_state_version for output in outputs] == list(range(1, 6))
    assert [output.audio_timestamp for output in outputs] == [80_000_000 * (i + 1) for i in range(5)]
    assert all(output.finite and output.hidden_t.size for output in outputs)
    assert state.total_samples == 5 * FRAME_SAMPLES
    assert state.listener_input_frames == 5
    assert state.dropped_listener_frames == 0


def test_listener_accepts_only_causal_reference_frame_for_current_tick() -> None:
    listener = GemmaListener(backend=_RecordingBackend())
    state = GemmaListenerState()
    reference = AssistantReferenceFrame(
        frame_id=0,
        pcm=np.zeros(FRAME_SAMPLES, dtype=np.float32),
        emitted_at_ns=80_000_000,
        sample_rate=16_000,
    )

    with pytest.raises(ValueError, match="future|causale"):
        listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), reference, state)
    assert state.future_reference_leaks == 1
    assert state.listener_input_frames == 0

    listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), None, state)
    output = listener.step(np.zeros(FRAME_SAMPLES, dtype=np.float32), reference, state)
    assert output.reference_frame_id == 0
    assert state.listener_input_frames == 2


def test_realtime_scheduler_drives_listener_while_response_is_active() -> None:
    backend = _RecordingBackend()
    listener = GemmaListener(backend=backend)

    class _Tokenizer:
        def apply_chat_template(self, messages, **_):
            return messages[-1]["content"]

        def __call__(self, text, **_):
            return {"input_ids": np.array([[1]])}

    class _Engine:
        loaded = True
        tokenizer = _Tokenizer()

        def transcribe_audio(self, _pcm):
            raise AssertionError("P3 realtime non deve usare ASR")

    engine = _Engine()
    engine.listener = listener
    sessions = SessionManager()
    client = TestClient(create_app(engine, sessions))
    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        session.active_response = True
        pcm = np.zeros(3 * FRAME_SAMPLES, dtype="<i2").tobytes()
        ws.send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        })
        deadline = time.monotonic() + 1.0
        while session.listener_state.listener_input_frames < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.listener_state.listener_input_frames == 3
        assert session.listener_state.total_samples == 3 * FRAME_SAMPLES
        assert len(backend.calls) == 3
        assert session.scheduler.is_running
        ws.send_json({"type": "session.close"})


def test_p3_gate_harness_smoke_runs_full_accounting_without_model_download() -> None:
    from pathlib import Path
    from tools.kernel_gates.p3_checks import run_checks

    trace: list[dict[str, object]] = []
    metrics, thresholds = run_checks(
        Path(__file__).resolve().parents[2],
        listener=GemmaListener(backend=_RecordingBackend()),
        state=GemmaListenerState(context_frames=2),
        seed=1234,
        trace=trace,
    )

    assert metrics["listener_input_frames"] == 10_000
    assert metrics["samples_received"] == metrics["samples_accounted_for"]
    assert metrics["replay_seconds"] >= thresholds["replay_seconds_min"]
    assert metrics["listener_coverage_while_speaking"] == 1.0
    assert metrics["future_reference_leaks"] == 0
    assert metrics["wrong_reference_frames"] == 0
    assert len([event for event in trace if event["event"] == "p3_listener_frame"]) == 10_000


def test_listener_rejects_non_frame_or_nonfinite_pcm() -> None:
    listener = GemmaListener(backend=_RecordingBackend())
    state = GemmaListenerState()

    with pytest.raises(ValueError, match="1280"):
        listener.step(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32), None, state)
    with pytest.raises(ValueError, match="finite"):
        listener.step(np.full(FRAME_SAMPLES, np.nan, dtype=np.float32), None, state)
    assert state.listener_input_frames == 0
