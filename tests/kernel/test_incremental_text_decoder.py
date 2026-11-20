from __future__ import annotations

from pathlib import Path

import pytest

from duplex_kernel.incremental_text_decoder import IncrementalTextDecoder


def _local_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(
        str(Path(__file__).resolve().parents[2] / "data" / "tokenizer.json")
    )


def test_local_bytefallback_emoji_keeps_zero_width_source_tokens_and_emits_once() -> None:
    tokenizer = _local_tokenizer()
    assert tokenizer.__class__.__module__.startswith("tokenizers")
    assert [tokenizer.id_to_token(token_id) for token_id in (478, 397, 409, 406)] == [
        "<0xF0>", "<0x9F>", "<0xAB>", "<0xA8>",
    ]
    decoder = IncrementalTextDecoder(tokenizer)

    deltas = [
        decoder.append(token_id, source_sequence=index, source_token_id=100 + index)
        for index, token_id in enumerate((478, 397, 409, 406))
    ]

    assert deltas == ["", "", "", "🫨"]
    assert "".join(deltas) == tokenizer.decode([478, 397, 409, 406], skip_special_tokens=True)
    assert [source.source_sequence for source in decoder.source_tokens] == [0, 1, 2, 3]
    assert [source.source_token_id for source in decoder.source_tokens] == [100, 101, 102, 103]
    assert decoder.last_text_source_group == ((0, 100), (1, 101), (2, 102), (3, 103))


def test_candidate_preview_isolated_and_final_flush_matches_incomplete_byte_decode() -> None:
    tokenizer = _local_tokenizer()
    decoder = IncrementalTextDecoder(tokenizer)
    for sequence, token_id in enumerate((478, 397, 409)):
        assert decoder.append(token_id, source_sequence=sequence) == ""

    before = (decoder.token_ids, decoder.source_tokens, decoder.text)
    assert decoder.preview(406) == "🫨"
    assert decoder.preview(406) == "🫨"
    assert (decoder.token_ids, decoder.source_tokens, decoder.text) == before

    expected = tokenizer.decode([478, 397, 409], skip_special_tokens=True)
    assert decoder.finish() == expected
    assert decoder.text == expected == "���"
    assert [source.source_sequence for source in decoder.source_tokens] == [0, 1, 2]
    assert decoder.last_text_source_group == ((0, 478), (1, 397), (2, 409))


def test_fallback_tokenizer_keeps_prefix_delta_and_clone_state() -> None:
    class ToyTokenizer:
        def decode(self, token_ids, *, skip_special_tokens=True):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    decoder = IncrementalTextDecoder(ToyTokenizer())
    assert decoder.append(4, source_sequence=0) == "<4>"
    assert decoder.preview(5) == "<5>"
    assert decoder.token_ids == (4,)
    fork = decoder.clone()
    assert fork.append(5, source_sequence=1) == "<5>"
    assert fork.finish() == ""
    assert decoder.token_ids == (4,)
    assert decoder.finish() == ""


def test_canonical_final_decode_cannot_rewrite_an_emitted_prefix() -> None:
    class UnstableTokenizer:
        calls = 0

        def decode(self, token_ids, *, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            self.calls += 1
            return "temporary" if self.calls == 1 else "changed"

    decoder = IncrementalTextDecoder(UnstableTokenizer())
    assert decoder.append(1) == "temporary"
    with pytest.raises(RuntimeError, match="changed text that was already emitted"):
        decoder.finish()
