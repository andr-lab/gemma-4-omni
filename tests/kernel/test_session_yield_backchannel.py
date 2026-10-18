from __future__ import annotations

import numpy as np
import pytest
import torch

from duplex_kernel.session import (
    DuplexKVCacheProxy,
    DuplexSession,
    DuplexSpeakerProxy,
    GenerationToken,
    SessionManager,
    SpokenTokenPacket,
    TurnLocalSpeechFIFO,
)


class DummyGemmaEngine:
    def __init__(self) -> None:
        self.embedded_tokens_calls: list[tuple[int, ...]] = []
        self.forward_steps: list[dict[str, Any]] = []
        self.backchannel_calls: list[dict[str, Any]] = []

    def embed_tokens(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        self.embedded_tokens_calls.append(tuple(token_ids))
        return torch.zeros((1, len(token_ids), 1536), dtype=torch.float32)

    def forward_embeds_step(
        self,
        *,
        session_id: Any,
        inputs_embeds: torch.Tensor,
        cache_position: int,
    ) -> None:
        self.forward_steps.append({
            "session_id": session_id,
            "shape": tuple(inputs_embeds.shape),
            "cache_position": cache_position,
        })

    def generate_ephemeral_backchannel(
        self,
        *,
        session_id: Any,
        start_kv_pos: int,
        max_tokens: int = 2,
    ) -> tuple[list[int], torch.Tensor]:
        self.backchannel_calls.append({
            "session_id": session_id,
            "start_kv_pos": start_kv_pos,
            "max_tokens": max_tokens,
        })
        tokens = [9991, 9992][:max_tokens]
        hiddens = torch.ones((1, len(tokens), 1536), dtype=torch.float32)
        return tokens, hiddens


class MockSpeaker:
    def __init__(self) -> None:
        self.aborted = False
        self.enqueued_bc: list[tuple[Any, Any]] = []

    def abort_playback(self) -> None:
        self.aborted = True

    def enqueue_backchannel_tts(self, bc_tokens: Any, bc_hiddens: Any) -> None:
        self.enqueued_bc.append((bc_tokens, bc_hiddens))


def test_turn_local_speech_fifo_clear_unspoken_on_bargein() -> None:
    fifo = TurnLocalSpeechFIFO(capacity=4)
    fifo.begin_turn(generation_id=1)

    packet = SpokenTokenPacket(
        token_id=42,
        text_piece="ciao",
        gemma_hidden_1536=torch.zeros((1, 1536), dtype=torch.float32),
        exclusive_kv_cursor=15,
    )
    fifo.put(packet, qwen_token_ids=[101, 102], generation_id=1)

    token = fifo.take_qwen_token(generation_id=1)
    assert token is not None
    assert fifo.ack_qwen_token(generation_id=1) is None

    token2 = fifo.take_qwen_token(generation_id=1)
    assert token2 is not None
    cursor = fifo.ack_qwen_token(generation_id=1)
    assert cursor == 15
    assert fifo.last_vocalized_kv_position == 15

    # clear_unspoken_on_bargein returns the last vocalized KV position
    rollback_pos = fifo.clear_unspoken_on_bargein()
    assert rollback_pos == 15


def test_duplex_session_assistant_speaking_property() -> None:
    session = DuplexSession("sess-speak")
    assert session.assistant_speaking is False
    session.assistant_speaking = True
    assert session.active_response is True
    session.active_response = False
    assert session.assistant_speaking is False


def test_handle_bargein_yield_end_to_end() -> None:
    engine = DummyGemmaEngine()
    manager = SessionManager(gemma_engine=engine)
    session = manager.create_session()
    session_id = session.session_id

    speaker = MockSpeaker()
    session.speaker = speaker
    session.assistant_speaking = True

    # Setup KV cache position and FIFO
    session.kv_cache.current_position = 25
    session.speech_fifo.begin_turn(generation_id=0)
    packet = SpokenTokenPacket(
        token_id=10,
        text_piece="Milano",
        gemma_hidden_1536=torch.zeros((1, 1536), dtype=torch.float32),
        exclusive_kv_cursor=18,
    )
    session.speech_fifo.put(packet, qwen_token_ids=[201], generation_id=0)
    tok = session.speech_fifo.take_qwen_token(generation_id=0)
    assert tok is not None
    session.speech_fifo.ack_qwen_token(generation_id=0)

    # Put 2 prebuffer embeddings (Q=2 frames)
    prebuffer_frame1 = torch.randn(1, 2, 1536)
    prebuffer_frame2 = torch.randn(1, 2, 1536)
    session.bargein_prebuffer_embeds = [prebuffer_frame1, prebuffer_frame2]

    # Execute YIELD
    manager.handle_bargein_yield(session_id)

    # 2. Speaker aborted
    assert speaker.aborted is True

    # 3. Interrupted turn close embeds forwarded at 19, then prebuffer at 19 + 6 = 25
    assert len(engine.embedded_tokens_calls) == 1
    assert engine.embedded_tokens_calls[0] == manager.interrupted_model_to_user_audio_token_ids
    assert len(engine.forward_steps) == 2
    assert engine.forward_steps[0]["cache_position"] == 19
    assert engine.forward_steps[1]["cache_position"] == 19 + len(manager.interrupted_model_to_user_audio_token_ids)
    assert engine.forward_steps[1]["shape"] == (1, 4, 1536)

    # 4. Final current_position advanced to 19 + 6 + 4 = 29
    expected_final_pos = 19 + len(manager.interrupted_model_to_user_audio_token_ids) + 4
    assert session.kv_cache.current_position == expected_final_pos

    # 5. Prebuffer cleared and speaking state updated
    assert len(session.bargein_prebuffer_embeds) == 0
    assert session.assistant_speaking is False
    assert session.user_speech_active is True


def test_handle_ephemeral_backchannel_preserves_audio_kv_pos() -> None:
    engine = DummyGemmaEngine()
    manager = SessionManager(gemma_engine=engine)
    session = manager.create_session()
    session_id = session.session_id

    speaker = MockSpeaker()
    session.speaker = speaker

    # User audio ongoing at KV position 50
    session.kv_cache.current_position = 50

    bc_tokens, bc_hiddens = manager.handle_ephemeral_backchannel(session_id)

    assert bc_tokens == [9991, 9992]
    assert len(speaker.enqueued_bc) == 1
    assert speaker.enqueued_bc[0][0] == [9991, 9992]

    # Crucial property: KV cache position was restored to saved position (50)
    assert session.kv_cache.current_position == 50
    assert len(engine.backchannel_calls) == 1
    assert engine.backchannel_calls[0]["start_kv_pos"] == 50


def test_session_manager_int_and_str_session_ids() -> None:
    manager = SessionManager()
    session = DuplexSession("123")
    manager.sessions[123] = session

    # Should resolve with both int and str
    speaker = MockSpeaker()
    session.speaker = speaker
    session.assistant_speaking = True

    manager.handle_bargein_yield(123)
    assert speaker.aborted is True

    speaker.aborted = False
    session.assistant_speaking = True
    manager.handle_bargein_yield("123")
    assert speaker.aborted is True
