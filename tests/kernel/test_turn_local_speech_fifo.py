from __future__ import annotations

import asyncio
import threading

import pytest
import torch

from duplex_kernel.live_say import TurnLocalQwenSayStream
from duplex_kernel.session import (
    DuplexSession,
    GemmaTokenTickPacer,
    SpokenTokenPacket,
    TurnLocalSpeechFIFO,
)


def _packet(token_id: int, text: str, exclusive_kv_cursor: int) -> SpokenTokenPacket:
    hidden = torch.full((1, 1536), float(token_id), dtype=torch.float32)
    return SpokenTokenPacket(
        token_id=token_id,
        text_piece=text,
        gemma_hidden_1536=hidden,
        exclusive_kv_cursor=exclusive_kv_cursor,
    )


def test_fifo_holds_packet_until_all_qwen_ids_are_consumed_and_returns_exact_hidden():
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=7)
    packet = _packet(22, "Aspetta!", 412)

    fifo.put(packet, qwen_token_ids=(31, 32), generation_id=7)
    first = fifo.take_qwen_token(initial=True, generation_id=7, timeout_s=0)

    assert first.token_id == 31
    assert first.packet is packet
    assert first.packet.gemma_hidden_1536 is packet.gemma_hidden_1536
    assert fifo.last_vocalized_kv_position is None
    assert len(fifo) == 1

    fifo.ack_qwen_token(generation_id=7)
    second = fifo.take_qwen_token(generation_id=7, timeout_s=0)
    assert second.token_id == 32
    assert second.packet is packet
    assert fifo.last_vocalized_kv_position is None
    fifo.ack_qwen_token(generation_id=7)
    assert fifo.last_vocalized_kv_position == 412
    assert fifo.vocalized_text == "Aspetta!"
    assert len(fifo) == 0


def test_fifo_clear_returns_last_consumed_cursor_and_discards_pending_turn_packets():
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=3)
    fifo.put(_packet(1, "Ciao", 99), qwen_token_ids=(10,), generation_id=3)
    fifo.take_qwen_token(initial=True, generation_id=3, timeout_s=0)
    fifo.ack_qwen_token(generation_id=3)
    fifo.put(_packet(2, " dopo", 100), qwen_token_ids=(11, 12), generation_id=3)
    fifo.take_qwen_token(generation_id=3, timeout_s=0)
    fifo.ack_qwen_token(generation_id=3)

    assert fifo.clear() == 99
    assert len(fifo) == 0
    assert fifo.last_vocalized_kv_position == 99


def test_gemma_tick_pacer_limits_two_tokens_per_rolling_80ms_tick_without_catchup():
    now = [0.0]
    sleeps: list[float] = []

    async def sleep(duration: float) -> None:
        sleeps.append(duration)
        now[0] += duration

    pacer = GemmaTokenTickPacer(
        tick_ms=80,
        max_tokens_per_tick=2,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    async def exercise() -> None:
        await pacer.acquire()
        now[0] += 0.012
        await pacer.acquire()
        assert not sleeps

        await pacer.acquire()
        assert sleeps == [pytest.approx(0.068)]
        assert now[0] == pytest.approx(0.08)

        # A delayed callback starts in the current window and cannot borrow
        # unused budget from earlier windows to emit a burst.
        now[0] += 0.24
        await pacer.acquire()
        await pacer.acquire()
        before = now[0]
        await pacer.acquire()
        assert now[0] == pytest.approx(before + 0.08)

    asyncio.run(exercise())


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) + 1 for char in text]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id - 1) for token_id in ids)


def test_incremental_qwen_stream_uses_exact_packet_hidden_and_marks_tail_hold():
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=8)
    consumed: list[tuple[int, str]] = []
    stream = TurnLocalQwenSayStream(
        qwen_tokenizer=_CharacterTokenizer(),
        text_eos_token_id=900,
        fifo=fifo,
        generation_id=8,
        on_packet_consumed=lambda cursor, text: consumed.append((cursor, text)),
    )
    first_packet = _packet(1, "As", 20)
    second_packet = _packet(2, "p", 21)
    stream.feed_packet(first_packet)

    initial = stream.take_initial_token(timeout_s=0)
    assert initial.token_id == ord("A") + 1
    assert torch.equal(initial.hidden, first_packet.gemma_hidden_1536.unsqueeze(1))
    assert fifo.last_vocalized_kv_position is None
    stream.ack_initial_token()

    stream.feed_packet(second_packet)
    second_id = stream.take_trailing_token(timeout_s=0)
    assert second_id == ord("s") + 1
    assert torch.equal(stream.hidden_for_frame(0).hidden, first_packet.gemma_hidden_1536.unsqueeze(1))
    assert fifo.last_vocalized_kv_position is None
    stream.ack_trailing_token()
    assert fifo.last_vocalized_kv_position == 20
    assert consumed == [(20, "As")]

    third_id = stream.take_trailing_token(timeout_s=0)
    assert third_id == ord("p") + 1
    assert torch.equal(stream.hidden_for_frame(1).hidden, second_packet.gemma_hidden_1536.unsqueeze(1))
    stream.ack_trailing_token()
    assert fifo.last_vocalized_kv_position == 21
    assert consumed == [(20, "As"), (21, "Asp")]

    stream.finish(final_text="Asp")
    assert stream.take_trailing_token(timeout_s=0) == 900
    assert stream.codec_eos_allowed is False
    stream.ack_trailing_token()
    assert stream.hidden_for_frame(2).tail_hold is True
    assert stream.codec_eos_allowed is True


def test_fifo_timeout_does_not_consume_initial_slot_and_clear_discards_unacked_token():
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=9)
    with pytest.raises(TimeoutError):
        fifo.take_qwen_token(initial=True, generation_id=9, timeout_s=0)

    packet = _packet(5, "x", 42)
    fifo.put(packet, qwen_token_ids=(70,), generation_id=9)
    fifo.take_qwen_token(initial=True, generation_id=9, timeout_s=0)
    assert fifo.last_vocalized_kv_position is None
    assert fifo.clear() is None
    assert len(fifo) == 0


def test_session_generation_invalidation_preserves_only_acknowledged_spoken_prefix_snapshot():
    session = DuplexSession("fifo-invalidation")
    token = session.begin_response("response")
    packet = _packet(8, "Aspetta!", 58)
    session.speech_fifo.put(packet, qwen_token_ids=(70,), generation_id=token.session_generation_id)
    session.speech_fifo.take_qwen_token(initial=True, generation_id=token.session_generation_id, timeout_s=0)
    session.speech_fifo.ack_qwen_token(generation_id=token.session_generation_id)
    pending = _packet(9, " no", 59)
    session.speech_fifo.put(pending, qwen_token_ids=(71, 72), generation_id=token.session_generation_id)

    session.invalidate_generation("bargein")

    assert session.speech_fifo.last_vocalized_kv_position == 58
    assert session.speech_fifo.vocalized_text == "Aspetta!"
    assert len(session.speech_fifo) == 0


def test_cancel_freezes_atomic_ack_snapshot_before_delayed_packet_callback():
    from duplex_kernel.server import (
        _record_turn_local_vocalized_prefix,
        _turn_local_vocalized_prefix_snapshot,
    )

    session = DuplexSession("fifo-ack-notify-race")
    generation = session.begin_response("response").session_generation_id
    branch = {
        "session_generation_id": generation,
        "spoken_prefix_text": "",
        "spoken_prefix_kv_position": None,
        "spoken_prefix_lock": threading.Lock(),
    }
    callback_entered = threading.Event()
    callback_release = threading.Event()
    callback_errors: list[BaseException] = []

    class OnePacketTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [700] if text == "Aspetta!" else []

        def decode(
            self,
            ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "Aspetta!" if ids == [700] else ""

    def delayed_callback(cursor: int, text: str) -> None:
        callback_entered.set()
        if not callback_release.wait(2.0):
            callback_errors.append(TimeoutError("delayed speech snapshot callback timed out"))
            return
        _record_turn_local_vocalized_prefix(branch, generation, cursor, text)

    stream = TurnLocalQwenSayStream(
        qwen_tokenizer=OnePacketTokenizer(),
        text_eos_token_id=900,
        fifo=session.speech_fifo,
        generation_id=generation,
        on_packet_consumed=delayed_callback,
    )
    stream.feed_packet(_packet(12, "Aspetta!", 123))
    stream.take_initial_token(timeout_s=0)
    ack_thread = threading.Thread(target=stream.ack_initial_token, daemon=True)
    ack_thread.start()
    try:
        assert callback_entered.wait(1.0)
        session.invalidate_generation("bargein")
        # Repeated control notifications after the first fence have no new
        # FIFO generation to snapshot and must leave the original frozen entry intact.
        session.invalidate_generation("duplicate_bargein")

        # The archive and cache finalizer both read the frozen session snapshot;
        # neither depends on the callback that is still paused after FIFO ACK.
        frozen_cursor, frozen_text = session.freeze_speech_snapshot(generation_id=generation)
        _record_turn_local_vocalized_prefix(branch, generation, frozen_cursor, frozen_text)
        archive_cursor, archive_text = _turn_local_vocalized_prefix_snapshot(branch)
        finalizer_cursor, finalizer_text = session.freeze_speech_snapshot(
            generation_id=generation
        )
        assert (archive_cursor, archive_text) == (123, "Aspetta!")
        assert (finalizer_cursor, finalizer_text) == (archive_cursor, archive_text)
        with pytest.raises(RuntimeError, match="snapshot is frozen"):
            session.speech_fifo.ack_qwen_token(generation_id=generation)
    finally:
        callback_release.set()
        ack_thread.join(timeout=1.0)

    assert not ack_thread.is_alive()
    assert callback_errors == []
    assert _turn_local_vocalized_prefix_snapshot(branch) == (123, "Aspetta!")


def test_fifo_stop_check_runs_outside_condition_and_generation_invalidation_unblocks_producer():
    fifo = TurnLocalSpeechFIFO(capacity=1)
    fifo.begin_turn(generation_id=4)
    fifo.put(_packet(1, "a", 1), qwen_token_ids=(65,), generation_id=4)
    lock_was_free = threading.Event()

    def stop_check() -> bool:
        def acquire_fifo_lock() -> None:
            with fifo._condition:
                lock_was_free.set()

        lock_probe = threading.Thread(target=acquire_fifo_lock, daemon=True)
        lock_probe.start()
        return lock_was_free.wait(0.2)

    with pytest.raises(RuntimeError, match="producer cancelled"):
        fifo.put(
            _packet(2, "b", 2),
            qwen_token_ids=(66,),
            generation_id=4,
            stop_check=stop_check,
        )
    assert lock_was_free.is_set()


def test_live_qwen_waiting_consumer_propagates_failure_instead_of_sending_text_eos():
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=10)
    stream = TurnLocalQwenSayStream(
        qwen_tokenizer=_CharacterTokenizer(),
        text_eos_token_id=900,
        fifo=fifo,
        generation_id=10,
    )
    started = threading.Event()
    outcome: dict[str, object] = {}

    def consume() -> None:
        started.set()
        try:
            outcome["value"] = stream.take_trailing_token(timeout_s=1.0)
        except BaseException as exc:
            outcome["error"] = exc

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert started.wait(0.2)
    stream.fail(RuntimeError("synthetic qwen producer failure"))
    consumer.join(timeout=1.0)

    assert not consumer.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
    assert str(outcome["error"]) == "synthetic qwen producer failure"
    assert "value" not in outcome


def test_closed_tts_unit_feed_runs_concurrently_with_capacity_four_fifo_consumer():
    fifo = TurnLocalSpeechFIFO(capacity=4)
    fifo.begin_turn(generation_id=11)
    stream = TurnLocalQwenSayStream(
        qwen_tokenizer=_CharacterTokenizer(),
        text_eos_token_id=900,
        fifo=fifo,
        generation_id=11,
    )
    texts = tuple("abcdef")
    packets = tuple(_packet(index + 1, text, 100 + index) for index, text in enumerate(texts))
    producer_error: list[BaseException] = []
    producer_done = threading.Event()

    def produce_closed_unit() -> None:
        try:
            for packet in packets:
                stream.feed_packet(packet)
            stream.finish(final_text="".join(texts))
        except BaseException as exc:
            producer_error.append(exc)
        finally:
            producer_done.set()

    producer = threading.Thread(target=produce_closed_unit, daemon=True)
    producer.start()
    initial = stream.take_initial_token(timeout_s=1.0)
    assert initial.token_id == ord("a") + 1
    stream.ack_initial_token()

    while not stream.codec_eos_allowed:
        try:
            token_id = stream.take_trailing_token(timeout_s=0.25)
        except TimeoutError:
            if producer_error:
                raise producer_error[0]
            continue
        if token_id is not None:
            stream.ack_trailing_token()

    producer.join(timeout=1.0)
    assert producer_done.is_set()
    assert not producer.is_alive()
    assert producer_error == []
    assert fifo.peak_depth <= fifo.capacity
    assert fifo.last_vocalized_kv_position == 105
