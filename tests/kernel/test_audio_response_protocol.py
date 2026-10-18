from __future__ import annotations

import pytest
import importlib.util
import sys
from pathlib import Path
import torch
from duplex_kernel.audio_response_conditioning import build_controller_guided_say_v1_messages

_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "duplex_kernel" / "audio_response_protocol.py"
_SPEC = importlib.util.spec_from_file_location("audio_response_protocol", _PROTOCOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROTOCOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROTOCOL
_SPEC.loader.exec_module(_PROTOCOL)

AUDIO_RESPONSE_PROTOCOL_V2 = _PROTOCOL.AUDIO_RESPONSE_PROTOCOL_V2
AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM = _PROTOCOL.AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM
AudioResponseProtocolError = _PROTOCOL.AudioResponseProtocolError
AudioResponseV2ActionConstraint = _PROTOCOL.AudioResponseV2ActionConstraint
AudioResponseV2Collector = _PROTOCOL.AudioResponseV2Collector
AudioResponseV2Parser = _PROTOCOL.AudioResponseV2Parser
format_audio_response_v2_target = _PROTOCOL.format_audio_response_v2_target
normalize_audio_response_protocol = _PROTOCOL.normalize_audio_response_protocol
finish_audio_response_v2_with_say_hidden = _PROTOCOL.finish_audio_response_v2_with_say_hidden
audio_response_actions_for_wake = _PROTOCOL.audio_response_actions_for_wake
audio_response_disposition_for_actions = _PROTOCOL.audio_response_disposition_for_actions
controller_guided_response_action = _PROTOCOL.controller_guided_response_action
slice_controller_guided_say_hidden = _PROTOCOL.slice_controller_guided_say_hidden
controller_backchannel_candidate_allowed = _PROTOCOL.controller_backchannel_candidate_allowed
controller_backchannel_should_stop = _PROTOCOL.controller_backchannel_should_stop
audio_response_v2_say_token_char_spans = _PROTOCOL.audio_response_v2_say_token_char_spans


def test_v2_target_uses_compact_ordered_json_and_unicode() -> None:
    target = format_audio_response_v2_target(
        heard="Perché?",
        action="SPEAK",
        say='È semplice: "ciao".',
    )

    assert target == '{"heard":"Perché?","action":"SPEAK","say":"È semplice: \\"ciao\\"."}'


def test_v2_parser_withholds_private_fields_and_decodes_escaped_say_incrementally() -> None:
    parser = AudioResponseV2Parser(expected_action="SPEAK")

    assert parser.feed('{"heard":"ho detto \\"ciao\\"","action":"SPEAK","say":"') == ""
    assert parser.heard == 'ho detto "ciao"'
    assert parser.action == "SPEAK"
    assert parser.feed("Perché ") == "Perché "
    assert parser.feed('\\"no\\"?\\n') == '"no"?\n'
    assert parser.feed('"}') == ""
    assert parser.finish() == {
        "heard": 'ho detto "ciao"',
        "action": "SPEAK",
        "say": 'Perché "no"?\n',
    }


def test_v2_parser_accepts_json_whitespace_between_ordered_fields() -> None:
    parser = AudioResponseV2Parser(expected_action="SPEAK")

    assert parser.feed(' \n{ "heard" : "ciao" , "action" : "SPEAK" , "say" : "Salve." }\t') == "Salve."
    assert parser.finish() == {"heard": "ciao", "action": "SPEAK", "say": "Salve."}


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        ('{"action":"SPEAK","heard":"x","say":"y"}', "expected_heard_key"),
        ('{"heard":"x","action":"YIELD","say":"y"}', "unsupported_action"),
        ('{"heard":"x","action":"REPLAN","say":"y"} trailing', "trailing_content"),
    ],
)
def test_v2_parser_fails_closed_with_a_traceable_reason(target: str, reason: str) -> None:
    parser = AudioResponseV2Parser()
    with pytest.raises(AudioResponseProtocolError) as caught:
        parser.feed(target)

    assert caught.value.reason == reason
    assert parser.failure_reason == reason


def test_v2_parser_enforces_action_heard_semantics_before_say_begins() -> None:
    parser = AudioResponseV2Parser(expected_action="BACKCHANNEL")

    with pytest.raises(AudioResponseProtocolError, match="expected_action_mismatch"):
        parser.feed('{"heard":"current utterance","action":"SPEAK","say":"')
    assert parser.failure_reason == "expected_action_mismatch"

    backchannel = AudioResponseV2Parser(expected_action="BACKCHANNEL")
    with pytest.raises(AudioResponseProtocolError, match="backchannel_heard_must_be_empty"):
        backchannel.feed('{"heard":"future words","action":"BACKCHANNEL","say":"')
    assert backchannel.failure_reason == "backchannel_heard_must_be_empty"


def test_wake_action_allows_native_response_choice_without_forcing_class() -> None:
    allowed = audio_response_actions_for_wake("BACKCHANNEL")
    assert allowed == ("SPEAK", "BACKCHANNEL")

    speak = AudioResponseV2Parser(allowed_actions=allowed)
    assert speak.feed('{"heard":"testo completo","action":"SPEAK","say":"Rispondo."}') == "Rispondo."
    assert speak.finish() == {
        "heard": "testo completo", "action": "SPEAK", "say": "Rispondo."
    }

    backchannel = AudioResponseV2Parser(allowed_actions=allowed)
    backchannel.feed('{"heard":"partial while user speaks","action":"BACKCHANNEL","say":"Si."}')
    assert backchannel.finish() == {"heard": "", "action": "BACKCHANNEL", "say": "Si."}

    # The selected BACKCHANNEL invalidates this partial utterance transcript;
    # it must never be returned as a completed user turn.
    partial_before_action = AudioResponseV2Parser(allowed_actions=allowed)
    partial_before_action.feed('{"heard":"partial while user speaks","action":"BACKCHANNEL"')
    assert partial_before_action.heard == ""

    disallowed = AudioResponseV2Parser(allowed_actions=allowed)
    with pytest.raises(AudioResponseProtocolError, match="action_not_allowed_for_wake"):
        disallowed.feed('{"heard":"testo","action":"REPLAN","say":"Aggiorno."}')


def test_union_action_constraint_leaves_choice_native_and_exposes_only_allowed_tokens() -> None:
    class _Tokenizer:
        def __init__(self):
            self._ids = {}
            self._pieces = {}

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            token_id = self._ids.setdefault(text, 100 + len(self._ids))
            self._pieces[token_id] = text
            return [token_id]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self._pieces[int(token_id)] for token_id in token_ids)

    allowed = audio_response_actions_for_wake("SPEAK")
    collector = AudioResponseV2Collector(allowed_actions=allowed)
    collector.feed_token(
        token_index=0,
        token_id=1,
        text='{"heard":"parole","action":"',
        hidden=object(),
    )
    tokenizer = _Tokenizer()
    constraint = AudioResponseV2ActionConstraint(
        expected_action=None,
        allowed_actions=allowed,
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )

    assert constraint.next_token_id(collector) is None
    candidates = constraint.action_value_candidate_token_ids(collector)
    decoded = {tokenizer.decode([token_id], skip_special_tokens=True) for token_id in candidates}
    assert decoded == {"SPEAK", "BACKCHANNEL"}
    assert collector.is_valid_next_text("SPEAK", terminal=False)
    assert collector.is_valid_next_text("BACKCHANNEL", terminal=False)
    assert not collector.is_valid_next_text("REPLAN", terminal=False)


def test_union_native_backchannel_action_closure_discards_partial_heard() -> None:
    class _Tokenizer:
        def __init__(self):
            self._ids = {}
            self._pieces = {}

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            token_id = self._ids.setdefault(text, 300 + len(self._ids))
            self._pieces[token_id] = text
            return [token_id]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self._pieces[int(token_id)] for token_id in token_ids)

    allowed = audio_response_actions_for_wake("BACKCHANNEL")
    collector = AudioResponseV2Collector(allowed_actions=allowed)
    prefix = '{"heard":"partial user words","action":"BACKCHANNEL'
    collector.feed_token(token_index=0, token_id=1, text=prefix, hidden=object())
    constraint = AudioResponseV2ActionConstraint(
        expected_action=None,
        allowed_actions=allowed,
        tokenizer=_Tokenizer(),
        enforce_json_structure=True,
    )

    quote_candidates = constraint.action_value_candidate_token_ids(collector)
    assert quote_candidates
    assert collector.is_valid_next_text('"', terminal=False)
    collector.feed_token(token_index=1, token_id=2, text='"', hidden=object())
    tail = ',"say":"Capito."}'
    collector.feed_token(token_index=2, token_id=3, text=tail, hidden=object())
    parsed = collector.finish(expected_text=prefix + '"' + tail)

    assert parsed == {"heard": "", "action": "BACKCHANNEL", "say": "Capito."}
    assert [token.token_id for token in collector.say_tokens] == [3]
    assert parsed["heard"] == ""


def test_replan_wake_allows_only_replan_or_speak() -> None:
    assert audio_response_actions_for_wake("REPLAN") == ("REPLAN", "SPEAK")
    with pytest.raises(ValueError, match="unsupported audio response action"):
        audio_response_actions_for_wake("YIELD")


@pytest.mark.parametrize(
    ("wake", "response", "expected"),
    [
        ("BACKCHANNEL", "SPEAK", "defer_until_endpoint"),
        ("BACKCHANNEL", "BACKCHANNEL", "backchannel_ack"),
        ("SPEAK", "BACKCHANNEL", "backchannel_ack"),
        ("SPEAK", "SPEAK", "archive_completed_user_turn"),
        ("REPLAN", "SPEAK", "archive_completed_user_turn"),
        ("REPLAN", "REPLAN", "archive_completed_user_turn"),
    ],
)
def test_native_response_action_uses_wake_specific_lifecycle(wake, response, expected) -> None:
    assert audio_response_disposition_for_actions(wake, response) == expected


def test_disallowed_response_action_fails_closed_before_lifecycle() -> None:
    with pytest.raises(AudioResponseProtocolError, match="response_action_not_allowed_for_wake"):
        audio_response_disposition_for_actions("REPLAN", "BACKCHANNEL")


def test_v2_parser_requires_complete_heard_for_completed_user_turns() -> None:
    parser = AudioResponseV2Parser(expected_action="REPLAN")

    with pytest.raises(AudioResponseProtocolError, match="completed_turn_heard_empty"):
        parser.feed('{"heard":"","action":"REPLAN","say":"')


def test_v2_parser_cancellation_clears_unpublished_private_content() -> None:
    parser = AudioResponseV2Parser(expected_action="SPEAK")
    parser.feed('{"heard":"private transcript","action":"SPEAK","say":"draft')

    parser.cancel()

    assert parser.cancelled
    assert parser.heard is None
    assert parser.action is None
    assert parser.pending_say == ""
    with pytest.raises(AudioResponseProtocolError, match="cancelled"):
        parser.feed(" later")


def test_v2_collector_holds_text_and_say_hidden_until_full_json_validation() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    hidden = [object(), object(), object(), object()]
    fragments = [
        '{"heard":"private transcript"',
        ',"action":"SPEAK","say"',
        ':"Ciao',
        '"}',
    ]

    for index, fragment in enumerate(fragments):
        assert collector.feed_token(
            token_index=index,
            token_id=100 + index,
            text=fragment,
            hidden=hidden[index],
        ) == ("Ciao" if index == 2 else "")
        assert not collector.publishable
        assert collector.publishable_text == ""
        assert collector.say_tokens == ()

    parsed = collector.finish(expected_text="".join(fragments))

    assert parsed == {"heard": "private transcript", "action": "SPEAK", "say": "Ciao"}
    assert collector.publishable_text == "Ciao"
    assert [token.token_index for token in collector.say_tokens] == [2]
    assert collector.say_tokens[0].hidden is hidden[2]
    assert collector.say_tokens[0].char_span == (0, 4)


def test_v2_say_hidden_finalizer_selects_only_tokens_overlapping_say_body() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    fragments = [
        '{"heard":"private","action":"SPEAK","say":"',
        "Ciao",
        "!",
        '"}',
    ]
    hidden = [
        torch.full((1, 1, 2), -1.0),
        torch.full((1, 1, 2), 10.0),
        torch.full((1, 1, 2), 11.0),
        torch.full((1, 1, 2), -2.0),
    ]
    for index, (fragment, token_hidden) in enumerate(zip(fragments, hidden, strict=True)):
        collector.feed_token(
            token_index=index,
            token_id=100 + index,
            text=fragment,
            hidden=token_hidden,
        )

    parsed, say_hidden = finish_audio_response_v2_with_say_hidden(
        collector,
        expected_text="".join(fragments),
    )

    assert parsed == {"heard": "private", "action": "SPEAK", "say": "Ciao!"}
    assert say_hidden.shape == (1, 2, 2)
    assert say_hidden[0, :, 0].tolist() == [10.0, 11.0]


def test_v2_say_spans_keep_json_boundary_token_and_map_split_escapes_causally() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    fragments = [
        '{"heard":"private","action":"SPEAK","say":"Ciao \\u',
        "00",
        "E9 e \\uD83D",
        "\\uDE00\"}",
    ]
    for index, fragment in enumerate(fragments):
        collector.feed_token(
            token_index=index,
            token_id=200 + index,
            text=fragment,
            hidden=torch.full((1, 1, 1), float(index + 1)),
        )
    parsed, hidden = finish_audio_response_v2_with_say_hidden(
        collector, expected_text="".join(fragments)
    )

    assert parsed["say"] == "Ciao é e 😀"
    assert hidden.shape == (1, 4, 1)
    assert [token.char_span for token in collector.say_tokens] == [
        (0, 5), (5, 5), (5, 9), (9, 10)
    ]
    assert audio_response_v2_say_token_char_spans(
        collector.say_tokens, parsed["say"]
    ) == [(0, 5), (5, 6), (5, 9), (9, 10)]


def test_v2_say_spans_preserve_decomposed_unicode_source_offsets() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    fragments = [
        '{"heard":"private","action":"SPEAK","say":"e',
        "\u0301",
        '"}',
    ]
    for index, fragment in enumerate(fragments):
        collector.feed_token(
            token_index=index,
            token_id=300 + index,
            text=fragment,
            hidden=torch.full((1, 1, 1), float(index + 1)),
        )
    parsed = collector.finish(expected_text="".join(fragments))

    assert parsed["say"] == "e\u0301"
    assert audio_response_v2_say_token_char_spans(
        collector.say_tokens, parsed["say"]
    ) == [(0, 1), (1, 2)]


def test_v2_empty_say_has_no_adapter_spans_and_collector_rejects_it() -> None:
    assert audio_response_v2_say_token_char_spans((), "") == []
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    with pytest.raises(AudioResponseProtocolError, match="say_empty"):
        collector.feed_token(
            token_index=0,
            token_id=400,
            text='{"heard":"private","action":"SPEAK","say":""}',
            hidden=torch.ones((1, 1, 1)),
        )


def test_v2_candidate_preview_rejects_compound_bad_key_without_mutating_parser() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.parser.feed('{"heard":"Marco"')
    before = (collector.parser.phase, collector.parser.heard, collector.parser.failure_reason)

    assert not collector.is_valid_next_text(',"nope"', terminal=False)
    assert collector.is_valid_next_text(',"action":"', terminal=False)
    assert (collector.parser.phase, collector.parser.heard, collector.parser.failure_reason) == before


def test_v2_candidate_preview_requires_complete_json_for_eos_and_bounds_backchannel_words() -> None:
    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    collector.parser.feed(
        '{"heard":"","action":"BACKCHANNEL","say":"Sì, certo, capito, volentieri '
    )

    assert not collector.is_valid_next_text(" per", terminal=False, max_say_words=4)
    assert collector.is_valid_next_text('"}', terminal=False, max_say_words=4)
    assert not collector.is_valid_next_text("", terminal=True, max_say_words=4)


def test_v2_candidate_preview_preserves_split_unicode_escape_and_accepts_eos_only_after_closure() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.parser.feed('{"heard":"ok","action":"SPEAK","say":"')

    assert collector.is_valid_next_text(r"\uD83D", terminal=False)
    collector.parser.feed(r"\uD83D")
    assert collector.parser.pending_say == ""
    assert not collector.is_valid_next_text("x", terminal=False)
    assert collector.is_valid_next_text(r"\uDE00", terminal=False)
    collector.parser.feed(r"\uDE00")
    assert collector.parser.pending_say == "😀"
    assert collector.is_valid_next_text('\"}', terminal=False)
    collector.parser.feed('\"}')
    assert collector.is_valid_next_text("", terminal=True)


def test_backchannel_constraint_closes_after_four_words_at_boundary() -> None:
    class _QuoteTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            assert text == '"'
            return [77]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return '"' if list(token_ids) == [77] else ""

    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    collector.parser.feed(
        '{"heard":"","action":"BACKCHANNEL","say":"Sì, certo, capito, volentieri '
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=_QuoteTokenizer(),
        enforce_json_structure=True,
    )

    assert constraint.next_token_id(collector) == 77
    assert constraint.bounded_output
    assert constraint.forced_structure_token_count == 1


def test_backchannel_fallback_closes_only_when_rejected_token_starts_a_fifth_word() -> None:
    class _QuoteTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            assert text == '"'
            return [77]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return '"' if list(token_ids) == [77] else ""

    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    collector.parser.feed(
        '{"heard":"","action":"BACKCHANNEL","say":"uno due tre quattro'
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=_QuoteTokenizer(),
        enforce_json_structure=True,
    )
    rejected = [(91, " cinque", False)]

    assert collector.would_exceed_say_word_limit(" cinque", max_say_words=4)
    assert constraint.bounded_say_closure_token_id(collector, rejected) == 77
    assert constraint.bounded_output


@pytest.mark.parametrize(
    ("say_suffix", "rejected_fragment"),
    [("rmente", "rmente cinque"), (r"\uD83D", r"\uDE00 cinque")],
    ids=["unfinished-word-piece", "unfinished-unicode-scalar"],
)
def test_backchannel_fallback_does_not_close_inside_unfinished_content(
    say_suffix: str,
    rejected_fragment: str,
) -> None:
    class _QuoteTokenizer:
        def encode(self, _text, *, add_special_tokens):
            raise AssertionError("unsafe closures must not request a quote token")

        def decode(self, _token_ids, *, skip_special_tokens):
            raise AssertionError("unsafe closures must not request a quote token")

    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    collector.parser.feed(
        '{"heard":"","action":"BACKCHANNEL","say":"uno due tre quattro'
        + say_suffix
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=_QuoteTokenizer(),
        enforce_json_structure=True,
    )

    assert constraint.bounded_say_closure_token_id(
        collector,
        [(91, rejected_fragment, False)],
    ) is None
    assert not constraint.bounded_output


def test_v2_collector_spans_cover_raw_say_body_across_token_boundaries_and_json_escapes() -> None:
    say = 'Ciao "Mondo", perché sì.'
    target = format_audio_response_v2_target(heard="ok", action="SPEAK", say=say)
    body_start = target.index('"say":"') + len('"say":"')
    body_end = len(target) - 2
    body = target[body_start:body_end]
    escaped_quote = body.index('\\"')
    fragments = [
        target[:body_start + escaped_quote],
        "\\",
        '"',
        target[body_start + escaped_quote + 2:],
    ]
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    for index, fragment in enumerate(fragments):
        collector.feed_token(token_index=index, token_id=100 + index, text=fragment, hidden=object())

    assert collector.finish(expected_text=target) == {"heard": "ok", "action": "SPEAK", "say": say}
    cursor = 0
    for token in collector.say_tokens:
        start, end = token.char_span
        assert start == cursor
        assert end >= start
        cursor = end
    assert cursor == len(say)


def test_v2_collector_discards_all_private_material_on_cancel_or_bad_closure() -> None:
    collector = AudioResponseV2Collector(expected_action="REPLAN")
    collector.feed_token(
        token_index=0,
        token_id=1,
        text='{"heard":"correction","action":"REPLAN","say":"Updated',
        hidden=object(),
    )
    with pytest.raises(AudioResponseProtocolError) as caught:
        collector.finish(expected_text='{"heard":"correction","action":"REPLAN","say":"Updated')
    assert caught.value.reason == "incomplete_json"
    assert not collector.publishable
    assert collector.publishable_text == ""
    assert collector.say_tokens == ()


def test_action_constraint_forces_only_existing_ids_after_heard_and_roundtrips_field_value() -> None:
    class _Tokenizer:
        pieces = {41: "BACK", 42: "CHANNEL"}

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            assert text == "BACKCHANNEL"
            return [41, 42]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self.pieces[int(token_id)] for token_id in token_ids)

    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=_Tokenizer(),
    )
    parts = [
        '{"heard":"","action":"',
        "BACK",
        "CHANNEL",
        '","say":"Capisco."}',
    ]
    forced_ids = []
    for index, part in enumerate(parts):
        collector.feed_token(token_index=index, token_id=50 + index, text=part, hidden=object())
        forced = constraint.next_token_id(collector)
        if forced is not None:
            forced_ids.append(forced)

    assert forced_ids == [41, 42]
    assert constraint.forced_token_count == 2
    assert collector.finish(expected_text="".join(parts)) == {
        "heard": "", "action": "BACKCHANNEL", "say": "Capisco."
    }


def test_action_constraint_finishes_only_the_selected_action_prefix() -> None:
    class _Tokenizer:
        pieces = {41: "PEAK", 42: '"'}

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"PEAK": [41], '"': [42]}[text]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self.pieces[int(token_id)] for token_id in token_ids)

    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.feed_token(
        token_index=0,
        token_id=10,
        text='{"heard":"ok","action":"S',
        hidden=object(),
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="SPEAK", tokenizer=_Tokenizer(), enforce_json_structure=True,
    )

    assert not collector.action_value_pending
    assert constraint.next_token_id(collector) == 41
    assert constraint.last_forced_kind == "action"
    assert constraint.forced_token_count == 1
    collector.feed_token(token_index=1, token_id=41, text="PEAK", hidden=object())
    assert constraint.next_token_id(collector) == 42
    assert constraint.last_forced_kind == "structure"


def test_action_constraint_rejects_a_different_action_prefix_before_cache_advance() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.feed_token(
        token_index=0,
        token_id=10,
        text='{"heard":"ok","action":"',
        hidden=object(),
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="SPEAK", tokenizer=object(), enforce_json_structure=True,
    )
    budget = {
        "max_new_tokens": 128,
        "heard_token_cap": 48,
        "action_token_cap": 8,
        "say_token_cap": 48,
    }

    assert not constraint.candidate_within_generation_budget(
        collector,
        "BACKCHANNEL",
        terminal=False,
        generated_tokens=20,
        budget=budget,
    )


def test_action_constraint_fails_closed_when_existing_tokenization_does_not_roundtrip() -> None:
    class _Tokenizer:
        def encode(self, _text, *, add_special_tokens):
            assert add_special_tokens is False
            return [41]

        def decode(self, _token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "different"

    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.feed_token(
        token_index=0,
        token_id=1,
        text='{"heard":"ok","action":"',
        hidden=object(),
    )
    constraint = AudioResponseV2ActionConstraint(expected_action="SPEAK", tokenizer=_Tokenizer())

    with pytest.raises(AudioResponseProtocolError) as caught:
        constraint.next_token_id(collector)

    assert caught.value.reason == "expected_action_tokenization_not_roundtrippable"

    collector.cancel()
    assert collector.parser.heard is None
    assert collector.parser.action is None
    assert collector.publishable_text == ""
    assert collector.say_tokens == ()


def test_v2_grammar_forces_private_backchannel_structure_without_forcing_say() -> None:
    class _Tokenizer:
        pieces = {
            1: "{",
            2: '\"heard\":\"\"',
            3: ',\"action\":\"',
            4: "BACKCHANNEL",
            5: '\"',
            6: ',\"say\":\"',
            7: "Capito.",
            8: "}",
        }
        by_text = {text: token_id for token_id, text in pieces.items()}

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [self.by_text[text]]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self.pieces[int(token_id)] for token_id in token_ids)

    tokenizer = _Tokenizer()
    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )
    forced_ids = [constraint.initial_token_id()]
    forced_kinds: list[str] = ["structure"]
    sequence = [
        (1, "{"),
        (2, '\"heard\":\"\"'),
        (3, ',\"action\":\"'),
        (4, "BACKCHANNEL"),
        (5, '\"'),
        (6, ',\"say\":\"'),
        (7, "Capito."),
    ]
    next_token = forced_ids[0]
    emitted: list[tuple[int, str]] = []
    for index, (token_id, fragment) in enumerate(sequence):
        if next_token is not None:
            assert next_token == token_id
        emitted.append((token_id, fragment))
        collector.feed_token(token_index=index, token_id=token_id, text=fragment, hidden=object())
        next_token = constraint.next_token_id(collector)
        if next_token is not None:
            forced_ids.append(next_token)
            forced_kinds.append(constraint.last_forced_kind)
    collector.feed_token(token_index=7, token_id=5, text='\"', hidden=object())
    next_token = constraint.next_token_id(collector)
    assert next_token == 8
    forced_kinds.append(constraint.last_forced_kind)
    emitted.extend(((5, '\"'), (8, "}")))
    collector.feed_token(token_index=8, token_id=8, text="}", hidden=object())

    assert collector.finish(expected_text='{"heard":"","action":"BACKCHANNEL","say":"Capito."}') == {
        "heard": "", "action": "BACKCHANNEL", "say": "Capito."
    }
    assert [fragment for _, fragment in emitted] == [tokenizer.pieces[token_id] for token_id, _ in emitted]
    assert forced_kinds == ["structure", "structure", "structure", "action", "structure", "structure", "structure"]
    assert constraint.forced_action_token_count == 1
    assert constraint.forced_structure_token_count == 6
    assert [token.hidden is not None for token in collector.say_tokens] == [True]


def test_v2_grammar_resumes_after_a_real_gemma_bpe_action_key_boundary() -> None:
    class _GemmaBpeBoundaryTokenizer:
        # Verified against the loaded Transformers 5.5 Gemma tokenizer. The
        # first delimiter token closes HEARD and opens the ACTION key; the
        # following token is a legal partial key, not a fresh grammar prefix.
        pieces = {
            4337: '","',
            236746: "a",
            2098: ',"',
            2064: "action",
            12375: '\":\"',
            647: "ction",
        }
        encodings = {
            ',"action":"': [2098, 2064, 12375],
            'action":"': [2064, 12375],
            'ction":"': [647, 12375],
        }

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return self.encodings[text]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self.pieces[int(token_id)] for token_id in token_ids)

    tokenizer = _GemmaBpeBoundaryTokenizer()
    collector = AudioResponseV2Collector(allowed_actions=("SPEAK", "BACKCHANNEL"))
    constraint = AudioResponseV2ActionConstraint(
        expected_action=None,
        allowed_actions=("SPEAK", "BACKCHANNEL"),
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )

    # The escaped quote is part of HEARD. Token 4337 then consumes the actual
    # closing quote, comma, and opening quote of the ACTION key in one piece.
    collector.feed_token(
        token_index=0,
        token_id=100,
        text=r'{"heard":"he said \"ciao',
        hidden=object(),
    )
    assert collector.parser.phase == "heard_value"
    assert constraint.next_token_id(collector) is None
    collector.feed_token(token_index=1, token_id=4337, text='","', hidden=object())
    assert collector.parser.phase == "action_prefix"

    # The first structural proposal may be replaced by a legal top candidate
    # (the model's `a` token). The next proposal must resume at `ction`, not
    # replay stale IDs for the full `,"action":"` sequence.
    constraint.next_token_id(collector)
    assert collector.is_valid_next_text("a", terminal=False)
    collector.feed_token(token_index=2, token_id=236746, text="a", hidden=object())
    assert constraint.next_token_id(collector) == 647
    assert collector.is_valid_next_text("ction", terminal=False)
    collector.feed_token(token_index=3, token_id=647, text="ction", hidden=object())
    assert constraint.next_token_id(collector) == 12375


def test_v2_system_prompt_is_generic_and_marks_only_say_as_user_facing() -> None:
    assert AUDIO_RESPONSE_PROTOCOL_V2 == "heard_action_say_v2"
    assert '{"heard":"...","action":"SPEAK|BACKCHANNEL|REPLAN","say":"..."}' in AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM
    assert "solo" in AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM.lower()
    assert all(action in AUDIO_RESPONSE_PROTOCOL_V2_SYSTEM for action in ("SPEAK", "BACKCHANNEL", "REPLAN"))


def test_optional_bundle_protocol_keeps_v1_as_the_default_and_rejects_unknown_ids() -> None:
    assert normalize_audio_response_protocol(None) is None
    assert normalize_audio_response_protocol("") is None
    assert normalize_audio_response_protocol(AUDIO_RESPONSE_PROTOCOL_V2) == AUDIO_RESPONSE_PROTOCOL_V2
    with pytest.raises(ValueError, match="unsupported audio response protocol"):
        normalize_audio_response_protocol("future_protocol")


def test_controller_guided_say_uses_the_controller_action_without_reclassification() -> None:
    assert controller_guided_response_action("SPEAK") == "SPEAK"
    assert controller_guided_response_action("BACKCHANNEL") == "BACKCHANNEL"
    with pytest.raises(AudioResponseProtocolError, match="controller_action_not_renderable"):
        controller_guided_response_action("LISTEN")


def test_controller_guided_say_prompt_keeps_audio_history_and_action_fixed() -> None:
    messages = build_controller_guided_say_v1_messages(
        language="Italian",
        history_messages=[{"role": "assistant", "content": "Piacere di conoscerti."}],
        audio_token="<|audio|>",
        session_instructions="Parliamo di Leonardo.",
        action="BACKCHANNEL",
    )

    assert messages[1:] == [
        {"role": "assistant", "content": "Piacere di conoscerti."},
        {"role": "user", "content": "<|audio|>"},
    ]
    system = messages[0]["content"]
    assert "BACKCHANNEL" in system
    assert "1–4 parole" in system
    assert "heard_action_say_v2" not in system
    assert '"heard"' not in system and '"action"' not in system


def test_controller_guided_speak_prompt_keeps_action_fixed() -> None:
    messages = build_controller_guided_say_v1_messages(
        language="Italian",
        history_messages=[],
        audio_token="<|audio|>",
        session_instructions=None,
        action="SPEAK",
    )

    system = messages[0]["content"]
    assert "SPEAK" in system
    assert "non riclassificare" in system.lower()
    assert "genera soltanto il testo da pronunciare" in system.lower()
    assert "1–4 parole" not in system


def test_controller_guided_say_hidden_contains_only_generated_text_tokens() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)

    say_hidden = slice_controller_guided_say_hidden(
        hidden,
        hidden_prompt_len=3,
        generated_token_ids=[41, 99, 42],
        excluded_token_ids={99},
    )

    assert torch.equal(say_hidden, hidden[:, [3, 5], :])
    with pytest.raises(AudioResponseProtocolError, match="hidden_token_alignment"):
        slice_controller_guided_say_hidden(
            hidden,
            hidden_prompt_len=3,
            generated_token_ids=[41],
        )


def test_controller_backchannel_stops_on_punctuation_or_fourth_word_boundary() -> None:
    assert controller_backchannel_should_stop("Perfetto.")
    assert controller_backchannel_should_stop("Sì molto versatile davvero ")
    assert not controller_backchannel_should_stop("Sì molto versatile")


def test_controller_backchannel_candidate_filter_keeps_generated_text_within_four_words() -> None:
    assert controller_backchannel_candidate_allowed("Sì", ",", terminal=False)
    assert controller_backchannel_candidate_allowed(
        "Sì molto versatile dav", "vero", terminal=False
    )
    assert controller_backchannel_candidate_allowed("Va bene davvero", "!", terminal=False)
    assert controller_backchannel_candidate_allowed("Va bene davvero!", "", terminal=True)
    assert not controller_backchannel_candidate_allowed("Sì,", " molto", terminal=False)
    assert not controller_backchannel_candidate_allowed(
        "Sì molto versatile davvero", " notevole", terminal=False
    )
    assert not controller_backchannel_candidate_allowed("", "", terminal=True)
