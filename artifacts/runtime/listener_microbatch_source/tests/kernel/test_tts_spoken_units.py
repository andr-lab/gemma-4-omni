from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.session import SpokenTokenPacket
from duplex_kernel.tts_units import SpokenTextUnitAssembler


def _packet(token_id: int, text_piece: str, cursor: int) -> SpokenTokenPacket:
    return SpokenTokenPacket(
        token_id=token_id,
        text_piece=text_piece,
        gemma_hidden_1536=np.zeros((1, 1536), dtype=np.float32),
        exclusive_kv_cursor=cursor,
    )


def test_spoken_units_keep_original_packets_and_carry_control_source_ids() -> None:
    assembler = SpokenTextUnitAssembler(max_words=2)
    assert assembler.push_source(100, 10, "") == []
    ciao = _packet(101, "Ciao", 11)
    mondo = _packet(103, " mondo", 13)
    punctuation = _packet(104, "!", 14)

    assert assembler.push(ciao) == []
    assert assembler.push_source(102, 12, "") == []
    assert assembler.push(mondo) == []
    units = assembler.push(punctuation)

    assert len(units) == 1
    unit = units[0]
    assert unit.text == "Ciao mondo!"
    assert [source.token_id for source in unit.source_tokens] == [100, 101, 102, 103, 104]
    assert [source.exclusive_kv_cursor for source in unit.source_tokens] == [10, 11, 12, 13, 14]
    assert unit.token_ids == (100, 101, 102, 103, 104)
    assert all(
        actual is expected
        for actual, expected in zip(unit.token_packets, (ciao, mondo, punctuation))
    )
    assert unit.source_tokens[0].speech_packet is None
    assert unit.source_tokens[2].speech_packet is None
    with pytest.raises((AttributeError, TypeError)):
        unit.token_packets[0].token_id = 999
    assert unit.exclusive_kv_cursor == 14
    assert assembler.finish() == []


def test_spoken_units_use_soft_word_limit_and_preserve_incremental_unicode_text() -> None:
    assembler = SpokenTextUnitAssembler(max_words=2)
    pieces = [
        _packet(1, "L’", 1),
        _packet(2, "itinerario", 2),
        _packet(3, " per", 3),
        _packet(4, " Torino.", 4),
        _packet(5, " Domani", 5),
    ]
    emitted = []
    for packet in pieces:
        emitted.extend(assembler.push(packet))
    emitted.extend(assembler.finish())

    assert [unit.text for unit in emitted] == ["L’itinerario per Torino.", " Domani"]
    assert "".join(unit.text for unit in emitted) == "".join(packet.text_piece for packet in pieces)
    assert [unit.token_ids for unit in emitted] == [(1, 2, 3, 4), (5,)]
    assert [unit.exclusive_kv_cursor for unit in emitted] == [4, 5]
    assert assembler.finish() == []


def test_one_source_packet_is_never_split_and_trailing_control_ids_are_not_a_unit() -> None:
    assembler = SpokenTextUnitAssembler(max_words=2)
    multiword = _packet(1, "Uno, due e tre ", 1)

    units = assembler.push(multiword)

    assert len(units) == 1
    assert units[0].text == "Uno, due e tre "
    assert units[0].token_packets[0] is multiword
    assert assembler.push_source(2, 2, "") == []
    assert assembler.finish() == []


def test_finish_drops_empty_source_ids_after_last_visible_say_token() -> None:
    assembler = SpokenTextUnitAssembler(max_words=2)
    visible = _packet(10, "Ciao", 10)

    assert assembler.push(visible) == []
    assert assembler.push_source(11, 11, "") == []
    units = assembler.finish()

    assert len(units) == 1
    assert units[0].text == "Ciao"
    assert units[0].token_packets == (visible,)
    assert units[0].token_ids == (10,)
    assert units[0].exclusive_kv_cursor == 10


def test_empty_source_ids_after_closed_unit_are_carried_to_next_visible_unit_only() -> None:
    assembler = SpokenTextUnitAssembler(max_words=2)
    first = [
        _packet(1, "Uno", 1),
        _packet(3, " due.", 3),
    ]
    assert assembler.push(first[0]) == []
    assert assembler.push_source(2, 2, "") == []
    first_unit = assembler.push(first[1])
    assert len(first_unit) == 1
    assert first_unit[0].token_ids == (1, 2, 3)

    assert assembler.push_source(4, 4, "") == []
    last_visible = _packet(5, " Poi", 5)
    assert assembler.push(last_visible) == []
    last_unit = assembler.finish()

    assert len(last_unit) == 1
    assert last_unit[0].token_ids == (4, 5)
    assert last_unit[0].exclusive_kv_cursor == 5


def test_spoken_unit_assembler_rejects_source_cursor_drift_and_packet_mismatch() -> None:
    assembler = SpokenTextUnitAssembler(max_words=2)
    assembler.push_source(1, 3, "")
    with pytest.raises(ValueError, match="cursor"):
        assembler.push_source(2, 3, "")

    other = SpokenTokenPacket(
        token_id=10,
        text_piece="ciao",
        gemma_hidden_1536=np.zeros((1, 1536), dtype=np.float32),
        exclusive_kv_cursor=4,
    )
    with pytest.raises(ValueError, match="corrisponde"):
        assembler.push_source(11, 5, "ciao", speech_packet=other)
