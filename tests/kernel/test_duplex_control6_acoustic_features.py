from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from duplex_kernel.acoustic_phase_features import acoustic_phase_features
from duplex_kernel.duplex_control6 import (
    ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT,
    DuplexControl6Policy,
)
from duplex_kernel.turn_policy import PolicyInput
from training.duplex_qlora.control import ControlHead
from training.duplex_qlora.schema import ACTIONS


class _RecordingHead(ControlHead):
    input_dim = 1540

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.marker = nn.Parameter(torch.zeros(()))
        self.seen: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.seen = features.detach().clone()
        return torch.zeros((features.shape[0], len(ACTIONS)), device=features.device)


def test_acoustic_head_receives_native_hidden_plus_causal_pcm_features() -> None:
    head = _RecordingHead()
    policy = DuplexControl6Policy({
        "adapter_path": "adapter",
        "control_head": "head.pt",
        "actions": list(ACTIONS),
        "feature_contract": ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT,
        "head": head,
    })
    user_frames = [np.full(1280, value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.4, 0.5)]
    user_history = np.concatenate(user_frames)
    assistant_previous = np.full(1280, 0.25, dtype=np.float32)

    decision = policy.step(PolicyInput(
        session_id="acoustic-policy-test",
        frame_id=4,
        user_pcm=user_frames[-1],
        user_pcm_history=user_history,
        assistant_reference_pcm=assistant_previous,
        speaker_active=False,
        generation_id=1,
        listener_audio_queries=torch.ones((1, 4, 1536)),
        listener_frame_id=4,
    ))

    assert head.seen is not None
    assert head.seen.shape == (1, 1540)
    assert torch.equal(head.seen[0, :1536], torch.ones(1536))
    expected = acoustic_phase_features(user_history, assistant_previous)
    assert torch.allclose(head.seen[0, 1536:], torch.from_numpy(expected))
    assert head.seen[0, 1536].item() == pytest.approx(float(np.log10(0.5)), abs=1e-6)
    assert decision.source.endswith("+causal_acoustic_phase")
    assert decision.diagnostics["feature_contract_input_dim"] == 1540
    assert decision.diagnostics["acoustic_phase_features_used"] is True


def test_acoustic_head_accepts_prior_query_only_for_explicit_silence_skip() -> None:
    head = _RecordingHead()
    policy = DuplexControl6Policy({
        "adapter_path": "adapter",
        "control_head": "head.pt",
        "actions": list(ACTIONS),
        "feature_contract": ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT,
        "head": head,
    })
    query = torch.full((1, 4, 1536), 0.25)
    state = PolicyInput(
        session_id="silence-skip-policy-test",
        frame_id=7,
        user_pcm=np.zeros(1280, dtype=np.float32),
        user_pcm_history=np.zeros(5 * 1280, dtype=np.float32),
        assistant_reference_pcm=np.zeros(1280, dtype=np.float32),
        speaker_active=False,
        generation_id=1,
        listener_audio_queries=query,
        listener_frame_id=5,
        listener_query_reused=True,
    )

    decision = policy.step(state)
    assert decision.frame_id == 7
    assert decision.listener_frame_id == 5
    assert decision.diagnostics["listener_query_reused"] is True
    assert decision.diagnostics["listener_query_age_frames"] == 2

    with pytest.raises(RuntimeError, match="silence-skip"):
        policy.step(PolicyInput(
            **{
                **state.__dict__,
                "listener_query_reused": False,
            }
        ))


def test_realtime_scheduler_policy_uses_current_native_tick_and_causal_pcm_snapshot() -> None:
    import base64
    import time
    from fastapi.testclient import TestClient

    from duplex_kernel.duplex_control6 import ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT
    from duplex_kernel.listener import GemmaListener, ListenerBackendOutput
    from duplex_kernel.server import create_app
    from duplex_kernel.session import SessionManager
    from duplex_kernel.turn_policy import NoOpPolicy

    class _SchedulerHead(_RecordingHead):
        def __init__(self) -> None:
            super().__init__()
            self.seen_rows: list[torch.Tensor] = []

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.seen = features.detach().clone()
            self.seen_rows.append(self.seen)
            return torch.zeros((features.shape[0], len(ACTIONS)), device=features.device)

    class _Backend:
        hidden_size = 1536

        def encode(self, user_pcm, _assistant_reference, _state):
            value = float(np.asarray(user_pcm, dtype=np.float32)[0])
            return ListenerBackendOutput(
                gemma_hidden_t=torch.zeros((1, 1536)),
                gemma_audio_queries_t=torch.full((1, 4, 1536), value),
            )

    head = _SchedulerHead()
    runtime = {
        "adapter_path": "/fixture/adapter",
        "control_head": "/fixture/head.pt",
        "actions": list(ACTIONS),
        "feature_contract": ACOUSTIC_PHASE_CONTROL_FEATURE_CONTRACT,
        "head": head,
        "base_path": "/fixture/base",
        "base_model_id": "fixture/gemma",
        "feature_source": {"base_model_id": "fixture/gemma", "adapter_dir": "adapter"},
        "runtime_abi": {"hidden_dim": 1536, "query_index": 3, "classes": list(ACTIONS)},
        "bundle_id": "fixture-phase-head",
    }

    class _Engine:
        loaded = True
        enforce_production_listener = False
        enable_duplex_control6 = True
        duplex_control6_runtime = runtime
        listener = GemmaListener(backend=_Backend())
        default_turn_policy_factory = NoOpPolicy

    sessions = SessionManager()
    client = TestClient(create_app(_Engine(), sessions))
    user_frames = [np.full(1280, value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    observed_user_frames: list[np.ndarray] = []
    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {"turn_policy": "duplex_control6"}})
        loaded = ws.receive_json()
        if loaded["type"] != "model_loaded":
            loaded = ws.receive_json()
        assert loaded["type"] == "model_loaded"
        session = sessions.all()[0]
        consumed = 0
        for frame_index, frame in enumerate(user_frames):
            pcm16 = np.rint(frame * 32768.0).clip(-32768, 32767).astype("<i2")
            observed_user_frames.append(pcm16.astype(np.float32) / 32768.0)
            ws.send_json({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16.tobytes()).decode("ascii"),
            })
            deadline = time.monotonic() + 2.0
            while session.audio_ingress.stats()["input_frames_consumed"] < frame_index + 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert session.audio_ingress.stats()["input_frames_consumed"] == frame_index + 1
            consumed += 1
            if frame_index == 4:
                session.assistant_reference_buffer.append(
                    np.full(1280, 0.25, dtype=np.float32), frame_id=4
                )
                session.assistant_reference_buffer.append(
                    np.full(1280, 0.75, dtype=np.float32), frame_id=5
                )
        ws.send_json({"type": "session.close"})

    assert consumed == 6
    assert len(head.seen_rows) == 6
    expected_user_history = np.concatenate(observed_user_frames[-5:])
    expected = acoustic_phase_features(
        expected_user_history,
        np.full(1280, 0.25, dtype=np.float32),
    )
    assert head.seen_rows[-1].shape == (1, 1540)
    assert torch.allclose(head.seen_rows[-1][0, 1536:], torch.from_numpy(expected))
