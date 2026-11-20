from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from duplex_kernel.backchannel_guard import validate_backchannel_phrase
from duplex_kernel.server_response_builder import (
    _backchannel_budget_without_eos,
    _publish_backchannel_tokens_after_eos,
)
from duplex_kernel.server_response_runtime import (
    _decode_backchannel_source_text,
    _response_generation_max_new_tokens,
)


def test_private_and_controller_backchannel_limits_are_six_tokens() -> None:
    assert _response_generation_max_new_tokens(
        response_v2_enabled=False,
        private_backchannel_active=True,
        backchannel_action=True,
        requested_max_tokens=64,
    ) == 6
    assert _response_generation_max_new_tokens(
        response_v2_enabled=False,
        private_backchannel_active=False,
        backchannel_action=True,
        requested_max_tokens=64,
    ) == 6


def test_real_tokenizer_reconstructs_valid_phrase_before_natural_eos_within_budget() -> None:
    tokenizer = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "Capito.": 1, "[EOS]": 2},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(["[EOS]"])

    eos_id = tokenizer.token_to_id("[EOS]")
    assert eos_id is not None
    generated_ids = tokenizer.encode("Capito.").ids + [eos_id]
    reconstructed = tokenizer.decode(generated_ids, skip_special_tokens=True)

    assert len(generated_ids) <= 6
    assert reconstructed == "Capito."
    assert _decode_backchannel_source_text(
        {"gen_ids": generated_ids, "protocol_text": reconstructed}, tokenizer
    ) == reconstructed
    assert _decode_backchannel_source_text(
        {"gen_ids": generated_ids, "protocol_text": "changed"}, tokenizer
    ) is None
    assert validate_backchannel_phrase(reconstructed) == (True, "ok", 1)
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL",
        private=True,
        controller_guided=True,
        terminal_reason="eos",
        cancelled=False,
    ) is None


@pytest.mark.parametrize("terminal_reason", ["length", "budget_exhausted", "structure"])
def test_non_natural_termination_omits_the_entire_buffered_batch(terminal_reason: str) -> None:
    hidden = object()
    buffered = ((0, 17, "Capito.", hidden),)
    published: list[tuple[tuple[int, int, str, object], ...]] = []

    assert _backchannel_budget_without_eos(
        "BACKCHANNEL",
        private=True,
        controller_guided=True,
        terminal_reason=terminal_reason,
        cancelled=False,
        buffered_tokens=buffered,
    ) == "backchannel_without_eos"
    assert not _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason=terminal_reason,
        publish_batch=published.append,
    )
    assert published == []


def test_structural_callback_forced_eos_is_not_natural_backchannel_eos() -> None:
    assert _backchannel_budget_without_eos(
        "BACKCHANNEL",
        private=False,
        controller_guided=True,
        terminal_reason="eos",
        forced_structure_overrides=1,
        cancelled=False,
        buffered_tokens=((0, 17, "Capito.", object()),),
    ) == "backchannel_without_eos"


def test_prepublication_cancellation_emits_no_text_or_pcm_and_keeps_canonical_kv() -> None:
    canonical_kv = np.arange(12, dtype=np.int64)
    original_kv = canonical_kv.copy()
    private_fork = canonical_kv.copy()
    buffered = ((0, 31, "Capito.", object()),)
    live_say_tokens: list[int] = []
    public_text: list[str] = []
    pcm_chunks: list[np.ndarray] = []

    def publish_batch(packets: tuple[tuple[int, int, str, object], ...]) -> None:
        for _index, token_id, text, _hidden in packets:
            live_say_tokens.append(token_id)
            public_text.append(text)
            pcm_chunks.append(np.ones(1, dtype=np.float32))

    assert not _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason="eos",
        can_publish=lambda: False,
        publish_batch=publish_batch,
    )

    assert live_say_tokens == []
    assert public_text == []
    assert pcm_chunks == []
    np.testing.assert_array_equal(canonical_kv, original_kv)
    assert private_fork is not canonical_kv


def test_natural_stop_publishes_original_ids_and_hidden_in_one_batch() -> None:
    hidden = object()
    buffered = (
        (0, 41, "Va ", hidden),
        (1, 42, "bene.", hidden),
    )
    published: list[tuple[tuple[int, int, str, object], ...]] = []

    assert _publish_backchannel_tokens_after_eos(
        buffered,
        terminal_reason="stop",
        publish_batch=published.append,
    )

    assert published == [buffered]
    assert published[0][0][1] == 41
    assert published[0][0][3] is hidden
    assert published[0][1][1] == 42
    assert published[0][1][3] is hidden
