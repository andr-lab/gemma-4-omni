from __future__ import annotations

import pytest
import torch

from duplex_kernel.session import SpokenTokenPacket, TurnLocalSpeechFIFO


def _packet(token_id: int, text: str, cursor: int) -> SpokenTokenPacket:
    return SpokenTokenPacket(
        token_id=token_id,
        text_piece=text,
        gemma_hidden_1536=torch.zeros((1, 1536)),
        exclusive_kv_cursor=cursor,
    )


def test_fifo_reopens_only_after_closed_segment_drains_and_keeps_cumulative_ack():
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=17)

    first = _packet(10, "Ciao ", 101)
    fifo.put(first, qwen_token_ids=(20,), generation_id=17)
    assert fifo.take_qwen_token(initial=True, generation_id=17, timeout_s=0).packet is first
    fifo.ack_qwen_token(generation_id=17)
    fifo.close()
    assert fifo.take_qwen_token(generation_id=17, timeout_s=0) is None

    fifo.begin_segment(generation_id=17)
    second = _packet(11, "mondo.", 102)
    fifo.put(second, qwen_token_ids=(21,), generation_id=17)
    assert fifo.take_qwen_token(initial=True, generation_id=17, timeout_s=0).packet is second
    fifo.ack_qwen_token(generation_id=17)

    assert fifo.last_vocalized_kv_position == 102
    assert fifo.vocalized_text == "Ciao mondo."


@pytest.mark.parametrize("pending", ["queued", "inflight"])
def test_fifo_does_not_reopen_before_previous_segment_drains(pending: str):
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=18)
    packet = _packet(12, "stop", 201)
    fifo.put(packet, qwen_token_ids=(22,), generation_id=18)
    if pending == "inflight":
        fifo.take_qwen_token(initial=True, generation_id=18, timeout_s=0)
    fifo.close()

    with pytest.raises(RuntimeError, match="drained"):
        fifo.begin_segment(generation_id=18)
