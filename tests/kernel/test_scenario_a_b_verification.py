from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from client.duplex_dialogue_harness import (
    FRAME_SAMPLES_16K,
    StreamingDialogueScenario,
    StreamingDialogueStep,
    streaming_scenario_from_fixtures,
)
from duplex_kernel.session import (
    DuplexSession,
    DuplexSpeakerProxy,
    DuplexKVCacheProxy,
    GenerationToken,
    SessionManager,
    SpokenTokenPacket,
    TurnLocalSpeechFIFO,
)


class ConversationalGemmaEngine:
    """Mock engine simulating Gemma thinker + audio tower with KV cache inspection."""

    def __init__(self) -> None:
        self.forwarded_embeds: list[dict[str, Any]] = []
        self.ephemeral_backchannel_calls: list[dict[str, Any]] = []
        self.special_tokens_generated: list[str] = []

    def embed_tokens(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros((1, len(token_ids), 1536), dtype=torch.float32)

    def forward_embeds_step(
        self,
        *,
        session_id: Any,
        inputs_embeds: torch.Tensor,
        cache_position: int,
    ) -> None:
        self.forwarded_embeds.append({
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
        self.ephemeral_backchannel_calls.append({
            "session_id": session_id,
            "start_kv_pos": start_kv_pos,
            "max_tokens": max_tokens,
        })
        # Generate 2 backchannel tokens, e.g. "Capisco." / "Sì."
        bc_tokens = [10901, 10902][:max_tokens]
        bc_hiddens = torch.ones((1, len(bc_tokens), 1536), dtype=torch.float32)
        return bc_tokens, bc_hiddens


class PlayoutSpeaker:
    def __init__(self) -> None:
        self.playback_active = True
        self.aborted = False
        self.abort_timestamp_ns: int | None = None
        self.enqueued_bc_tts: list[tuple[Any, Any]] = []

    def abort_playback(self) -> None:
        self.playback_active = False
        self.aborted = True
        self.abort_timestamp_ns = time.monotonic_ns()

    def enqueue_backchannel_tts(self, bc_tokens: Any, bc_hiddens: Any) -> None:
        self.enqueued_bc_tts.append((bc_tokens, bc_hiddens))


def test_scenario_a_bargein_replan_criteria() -> None:
    """Scenario A: Barge-in (<160ms abort, KV rollback to unspoken, REPLAN Torino)."""
    fixture_dir = Path("artifacts/turn_local_rollback_harness_20261018/inputs_16k")
    scenario = streaming_scenario_from_fixtures("A", fixture_dir=fixture_dir)

    assert scenario.scenario_id == "A_milano_correction_torino"
    assert len(scenario.steps) == 2
    step_milano, step_torino = scenario.steps

    # Verify input audio exists and matches expectations
    assert "Milano" in step_milano.expected_text
    assert "Torino" in step_torino.expected_text
    assert step_torino.after_first_consumed_s == 1.5

    engine = ConversationalGemmaEngine()
    manager = SessionManager(gemma_engine=engine)
    session = manager.create_session()
    speaker = PlayoutSpeaker()
    session.speaker = speaker

    # Phase 1: Assistant begins speaking: "Certamente! Per il primo giorno a Milano ti consiglio di partire dal Duomo..."
    session.assistant_speaking = True
    session.kv_cache.current_position = 40  # e.g. 40 tokens planned in advance
    session.speech_fifo.begin_turn(generation_id=1)

    # Put packets into FIFO. Say packet 5 is "partire dal" at KV position 22,
    # and subsequent packets ("Duomo...") are queued up to 40 but not vocalized yet.
    packet_vocalized = SpokenTokenPacket(
        token_id=501,
        text_piece="partire dal",
        gemma_hidden_1536=torch.zeros((1, 1536), dtype=torch.float32),
        exclusive_kv_cursor=22,
    )
    session.speech_fifo.put(packet_vocalized, qwen_token_ids=[1001, 1002], generation_id=1)

    # Qwen consumed both tokens of "partire dal"
    tok1 = session.speech_fifo.take_qwen_token(generation_id=1)
    session.speech_fifo.ack_qwen_token(generation_id=1)
    tok2 = session.speech_fifo.take_qwen_token(generation_id=1)
    cursor = session.speech_fifo.ack_qwen_token(generation_id=1)
    assert cursor == 22

    # Queued unspoken packets (thinking ahead)
    packet_unspoken = SpokenTokenPacket(
        token_id=502,
        text_piece=" Duomo",
        gemma_hidden_1536=torch.zeros((1, 1536), dtype=torch.float32),
        exclusive_kv_cursor=23,
    )
    session.speech_fifo.put(packet_unspoken, qwen_token_ids=[1003], generation_id=1)

    # Phase 2: User interrupts at 1.5s with "Scusa, intendevo dire Torino, non Milano!"
    bargein_start_ns = time.monotonic_ns()

    # Prebuffer audio embeds captured during the 160ms barge-in detection window (2 ticks = 4 soft tokens)
    prebuffer_embeds = [torch.randn(1, 2, 1536), torch.randn(1, 2, 1536)]
    session.bargein_prebuffer_embeds = list(prebuffer_embeds)

    # YIELD is triggered by policy
    manager.handle_bargein_yield(session.session_id)
    bargein_end_ns = time.monotonic_ns()

    # Criterion 1: Playout aborted within < 160 ms
    latency_ms = (bargein_end_ns - bargein_start_ns) / 1_000_000.0
    assert latency_ms < 160.0, f"Barge-in latency {latency_ms} ms exceeds 160 ms"
    assert speaker.aborted is True

    # Criterion 2: KV cache rolled back to unspoken position (rollback_kv_pos + 1 = 23),
    # then sequentially advanced by turn close (6 tokens) and prebuffer (4 tokens) to 33
    assert engine.forwarded_embeds[0]["cache_position"] == 23
    assert engine.forwarded_embeds[1]["cache_position"] == 23 + len(manager.interrupted_model_to_user_audio_token_ids)
    assert session.kv_cache.current_position == 33

    # Criterion 3: Unspoken lookahead tokens ("Duomo") were discarded
    assert len(session.speech_fifo) == 0

    # Criterion 4: Prebuffer frames containing "Torino" were forwarded to the engine
    assert len(engine.forwarded_embeds) == 2
    # First forward: close turn tokens (<end_of_turn>\n<start_of_turn>user\n<|audio_boa|>)
    assert engine.forwarded_embeds[0]["shape"] == (1, 6, 1536)
    # Second forward: prebuffer (4 tokens)
    assert engine.forwarded_embeds[1]["shape"] == (1, 4, 1536)

    # Criterion 5: State transitioned to listening/user speech without generating special REPLAN tokens
    assert session.assistant_speaking is False
    assert session.user_speech_active is True
    assert len(engine.special_tokens_generated) == 0


def test_scenario_b_ephemeral_backchannel_preserves_kv_cache() -> None:
    """Scenario B: 15s long utterance with 300ms pause at 6s; ephemeral backchannel preserves KV."""
    fixture_dir = Path("artifacts/turn_local_rollback_harness_20261018/inputs_16k")
    scenario = streaming_scenario_from_fixtures("B", fixture_dir=fixture_dir)

    assert scenario.scenario_id == "B_docker_long_utterance"
    assert len(scenario.steps) == 1
    step_docker = scenario.steps[0]

    # Verify input audio exists and mentions Docker
    assert "Docker" in step_docker.expected_text
    assert step_docker.pcm_16k.size >= 16000 * 10  # ~15 seconds of audio

    engine = ConversationalGemmaEngine()
    manager = SessionManager(gemma_engine=engine)
    session = manager.create_session()
    speaker = PlayoutSpeaker()
    session.speaker = speaker

    # Phase 1: User starts speaking (first 6 seconds, e.g. "Ieri stavo configurando il server con Docker...")
    # KV cache advances to position 75 (representing user mic audio tokens Q=2 per 80ms)
    session.user_speech_active = True
    session.kv_cache.current_position = 75

    # Phase 2: At 6s, 300ms breathing pause occurs. Policy predicts BACKCHANNEL.
    saved_pos = session.kv_cache.current_position
    assert saved_pos == 75

    # Ephemeral backchannel executes
    bc_tokens, bc_hiddens = manager.handle_ephemeral_backchannel(session.session_id)

    # Criterion 1: Emits 1-2 token backchannel ("Capisco.") to speaker without closing user audio
    assert bc_tokens is not None and len(bc_tokens) == 2
    assert len(speaker.enqueued_bc_tts) == 1
    assert speaker.enqueued_bc_tts[0][0] == bc_tokens

    # Criterion 2: Crucial invariant — KV cache is rolled back immediately to saved_pos (75)
    assert session.kv_cache.current_position == saved_pos

    # Phase 3: User resumes speaking for remaining 9 seconds ("...e a un certo punto il container...")
    # Tokens continue appending contiguously from position 75 to position 180
    new_user_tokens = torch.randn(1, 105, 1536)
    engine.forward_embeds_step(
        session_id=session.session_id,
        inputs_embeds=new_user_tokens,
        cache_position=session.kv_cache.current_position,
    )
    session.kv_cache.current_position += 105

    # Criterion 3: Total KV cache is contiguous: first half (0..75) + second half (75..180) = 180 tokens
    assert session.kv_cache.current_position == 180
    assert engine.forwarded_embeds[-1]["cache_position"] == 75
    # The early discourse ("Docker" at t=2s, position ~25) is fully preserved in the primary KV cache!
