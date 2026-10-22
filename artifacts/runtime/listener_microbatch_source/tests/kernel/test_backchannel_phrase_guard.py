from __future__ import annotations

from duplex_kernel.backchannel_guard import (
    BackchannelAttemptLatch,
    peer_audio_event_key_for_frame,
    validate_backchannel_phrase,
)


def test_backchannel_guard_accepts_complete_short_acknowledgements() -> None:
    assert validate_backchannel_phrase("Sì") == (True, "ok", 1)
    assert validate_backchannel_phrase("Va bene, sì.") == (True, "ok", 3)


def test_backchannel_guard_rejects_fifth_lexical_word_and_open_clause() -> None:
    assert validate_backchannel_phrase("Esplorò l’anatomia umana con") == (
        False,
        "too_many_words",
        5,
    )
    assert validate_backchannel_phrase("D'accordo, con") == (
        False,
        "incomplete_clause",
        3,
    )


def test_backchannel_guard_accepts_server_max_words_keyword_contract() -> None:
    assert validate_backchannel_phrase("Va bene, sì.", max_words=4) == (True, "ok", 3)
    assert validate_backchannel_phrase("Sì, va bene, certo, grazie.", max_words=4) == (
        False,
        "too_many_words",
        5,
    )


def test_backchannel_guard_never_rewrites_source_text() -> None:
    original = "  Sì, certo.  "
    validate_backchannel_phrase(original)
    assert original == "  Sì, certo.  "


def test_backchannel_latch_allows_one_attempt_per_real_peer_event() -> None:
    latch = BackchannelAttemptLatch()
    event_key = peer_audio_event_key_for_frame(None, 12, new_voice_onset=True)
    next_frame_key = peer_audio_event_key_for_frame(event_key, 24, new_voice_onset=False)
    next_onset = peer_audio_event_key_for_frame(event_key, 31, new_voice_onset=True)

    assert not latch.has_attempted(None)
    assert event_key == ("onset", 12)
    assert next_frame_key == event_key  # endpoint/tail frames retain the current onset scope
    assert latch.mark_attempted(event_key)
    assert latch.has_attempted(event_key)
    # Response completion, including empty SAY, does not re-arm the event.
    assert not latch.mark_attempted(next_frame_key)
    assert next_onset == ("onset", 31)
    assert latch.mark_attempted(next_onset)
