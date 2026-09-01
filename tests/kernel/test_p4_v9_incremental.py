from __future__ import annotations

import pytest
import torch

from duplex_kernel.listener import GemmaListener
from duplex_kernel.models.v9 import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    IncrementalV9Projector,
)
from duplex_kernel.server import create_app
from duplex_kernel.session import SessionManager
from tools.kernel_gates.p4_checks import _run_equivalence, _run_future_prefix, _stack_cache


def _tiny_stream() -> tuple[DeepCausalTransformerProjectorV9, CausalConv1D, IncrementalV9Projector]:
    torch.manual_seed(17)
    projector = DeepCausalTransformerProjectorV9(
        raw_dim=8,
        text_dim=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
    ).eval()
    causal_conv = CausalConv1D(hidden_size=8, kernel_size=3).eval()
    return projector, causal_conv, IncrementalV9Projector(projector, causal_conv)


def test_incremental_v9_matches_causal_batch_forward_in_fp32() -> None:
    projector, causal_conv, stream = _tiny_stream()
    raw = torch.randn(2, 7, 8)
    anchor = torch.randn(2, 8)

    batch = projector(causal_conv(raw), anchor)
    state = stream.new_state(batch_size=2, device="cpu", dtype=torch.float32, max_seq_len=16)
    incremental = torch.stack([stream.step(raw[:, index], anchor, state) for index in range(raw.shape[1])], dim=1)

    torch.testing.assert_close(incremental, batch, rtol=2e-5, atol=2e-5)
    assert state.frames_processed == raw.shape[1]
    assert state.rope_position == raw.shape[1]


def test_incremental_v9_prefix_is_independent_of_future_frames() -> None:
    _, _, stream = _tiny_stream()
    prefix = torch.randn(1, 3, 8)
    future_a = torch.randn(1, 4, 8)
    future_b = torch.randn(1, 4, 8)
    anchor = torch.randn(1, 8)

    def run(sequence: torch.Tensor) -> torch.Tensor:
        state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=16)
        return torch.stack([stream.step(sequence[:, i], anchor, state) for i in range(sequence.shape[1])], dim=1)

    output_a = run(torch.cat((prefix, future_a), dim=1))
    output_b = run(torch.cat((prefix, future_b), dim=1))
    torch.testing.assert_close(output_a[:, :3], output_b[:, :3], rtol=0.0, atol=1e-6)


def test_incremental_v9_requires_a_fixed_text_anchor_and_supports_reset() -> None:
    _, _, stream = _tiny_stream()
    state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=4)
    raw = torch.zeros(1, 8)
    anchor = torch.ones(1, 8)

    stream.step(raw, anchor, state)
    with pytest.raises(ValueError, match="anchor"):
        stream.step(raw, anchor + 1.0, state)
    assert state.frames_processed == 1

    state.reset()
    assert state.frames_processed == 0
    assert state.rope_position == 0
    stream.step(raw, anchor, state)
    assert state.frames_processed == 1


def test_p4_gate_helpers_verify_equivalence_and_future_prefix() -> None:
    _, _, stream = _tiny_stream()
    raw = torch.randn(3, 5, 8)
    anchors = torch.randn(3, 8)
    equivalence_trace = []
    future_trace = []
    equivalence = _run_equivalence(stream, raw, anchors, trace=equivalence_trace)
    future = _run_future_prefix(stream, raw, anchors, trace=future_trace)
    assert equivalence["sequences"] == 3
    assert equivalence["frames"] == 15
    assert equivalence["mae"] <= 1e-6
    assert equivalence["cosine_min"] >= 0.999999
    assert future["sequences"] == 3
    assert future["max_difference"] == 0.0
    assert len([event for event in equivalence_trace if event["event"] == "p4_equivalence_frame"]) == 15


def test_session_v9_state_is_reset_and_released_with_session_lifecycle() -> None:
    _, _, stream = _tiny_stream()
    sessions = SessionManager()
    session = sessions.create_session()
    session.v9_state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=4)
    session.v9_text_anchor = torch.ones(1, 8)
    stream.step(torch.zeros(1, 8), session.v9_text_anchor, session.v9_state)
    session.reset_turn()
    assert session.v9_state.frames_processed == 0
    assert session.v9_text_anchor is None
    assert hasattr(session, "stale_session_mutations")
    sessions.close(session.session_id)
    assert session.v9_state is None


def test_realtime_scheduler_routes_listener_hidden_through_incremental_v9() -> None:
    class _Backend:
        def encode(self, user_pcm, assistant_reference, state):
            del user_pcm, assistant_reference, state
            return torch.ones(1, 8)

    projector, causal_conv, stream = _tiny_stream()
    listener = GemmaListener(backend=_Backend())

    class _Tokenizer:
        def apply_chat_template(self, messages, **_):
            return messages[-1]["content"]

        def __call__(self, text, **_):
            return {"input_ids": torch.tensor([[1]])}

    class _Engine:
        loaded = True
        device = "cpu"
        dtype = torch.float32
        tokenizer = _Tokenizer()
        v9_stream = stream

    engine = _Engine()
    engine.listener = listener
    sessions = SessionManager()
    from fastapi.testclient import TestClient
    import base64
    import time

    with TestClient(create_app(engine, sessions)).websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        session = sessions.all()[0]
        pcm = torch.zeros(1280, dtype=torch.int16).numpy().tobytes()
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        deadline = time.monotonic() + 1.0
        while (
            session.listener_state.listener_input_frames < 1
            or session.v9_state is None
            or session.v9_state.frames_processed < 1
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.listener_state.listener_input_frames == 1
        assert session.v9_state is not None
        assert session.v9_state.frames_processed == 1
        assert tuple(session.listener_hidden.shape) == (1, 16)
        torch.testing.assert_close(session.v9_text_anchor, torch.ones(1, 8))
        ws.send_json({"type": "session.close"})


def test_p4_gate_rejects_post_causal_conv_cache(tmp_path) -> None:
    cache = {
        "va_raw": [torch.zeros(1, 1536) for _ in range(100)],
        "va_htx": [torch.zeros(1536) for _ in range(100)],
        "va_c": [torch.zeros(1, dtype=torch.long) for _ in range(100)],
        "v9_raw_stage": "post_causal_conv",
    }
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    with pytest.raises(ValueError, match="pre-causal_conv|doppia"):
        _stack_cache(path)
    cache["v9_raw_stage"] = "pre_causal_conv"
    torch.save(cache, path)
    raw, anchor, codes, stage = _stack_cache(path)
    assert raw.shape == (100, 1, 1536)
    assert anchor.shape == (100, 1536)
    assert codes.shape == (100, 1)
    assert stage == "pre_causal_conv"


def test_incremental_v9_rejects_bad_shapes_and_sequence_overrun() -> None:
    _, _, stream = _tiny_stream()
    state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=1)
    with pytest.raises(ValueError, match="1536|raw"):
        stream.step(torch.zeros(1, 7), torch.zeros(1, 8), state)
    stream.step(torch.zeros(1, 8), torch.zeros(1, 8), state)
    with pytest.raises(OverflowError, match="max_seq_len"):
        stream.step(torch.zeros(1, 8), torch.zeros(1, 8), state)
