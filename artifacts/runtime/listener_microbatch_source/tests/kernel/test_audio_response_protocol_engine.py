from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import duplex_kernel.engine as engine_module
from duplex_kernel.audio_response_protocol import (
    AudioResponseProtocolError,
    AudioResponseV2ActionConstraint,
    AudioResponseV2Collector,
    finish_audio_response_v2_with_say_hidden,
)
from duplex_kernel.engine import PipelineEngine


class _AudioThinker:
    def __init__(self) -> None:
        self.model = self
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.config = SimpleNamespace(eos_token_id=3)

    def get_placeholder_mask(self, *, input_ids):
        audio = input_ids == 99
        empty = torch.zeros_like(audio)
        return empty, empty, audio

    def __call__(self, *, input_ids, **_kwargs):
        logits = torch.zeros((1, input_ids.shape[1], 5), dtype=torch.float32)
        preferred = getattr(self, "preferred_token_id", 2)
        logits[..., preferred] = 10.0
        logits[..., 2 if preferred != 2 else 4] = 9.0
        hidden = torch.ones((1, input_ids.shape[1], 4), dtype=torch.float32)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=object(),
        )


class _Tokenizer:
    eos_token_id = 3
    eot_id = None

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        del skip_special_tokens
        vocabulary = {2: "{", 3: "", 4: "Capito"}
        return "".join(vocabulary[int(token_id)] for token_id in token_ids)


def test_protocol_text_uses_exact_special_skipping_decode_not_cleaned_display_text() -> None:
    raw_text = '{"heard":"Capito","action":"SPEAK","say":"a  b"}<unused93><eos>'
    protocol_text = '{"heard":"Capito","action":"SPEAK","say":"a  b"}'

    class _DualDecodeTokenizer:
        @staticmethod
        def decode(_token_ids, *, skip_special_tokens):
            return protocol_text if skip_special_tokens else raw_text

    display_text, decoded_protocol_text = engine_module._decode_generated_response_texts(
        _DualDecodeTokenizer(), [1, 2, 3]
    )
    assert decoded_protocol_text == protocol_text
    assert display_text == '{"heard":"Capito","action":"SPEAK","say":"a b"}'

    rejected = AudioResponseV2Collector()
    rejected.feed_token(
        token_index=0,
        token_id=1,
        text=protocol_text,
        hidden=torch.ones((1, 1, 4)),
    )
    with pytest.raises(AudioResponseProtocolError, match="token_decode_mismatch") as rejected_error:
        rejected.finish(expected_text=display_text)
    assert rejected_error.value.expected_text_repr == repr(display_text)
    assert rejected_error.value.collector_text_repr == repr(protocol_text)
    assert rejected_error.value.generated_token_ids == [1]

    accepted = AudioResponseV2Collector()
    accepted.feed_token(
        token_index=0,
        token_id=1,
        text=protocol_text,
        hidden=torch.ones((1, 1, 4)),
    )
    parsed = accepted.finish(expected_text=decoded_protocol_text)
    assert parsed["say"] == "a  b"

    envelope_space = AudioResponseV2Collector()
    envelope_space.feed_token(
        token_index=0,
        token_id=1,
        text=protocol_text,
        hidden=torch.ones((1, 1, 4)),
    )
    assert envelope_space.finish(expected_text=protocol_text + " ")["say"] == "a  b"


class _CompoundBoundaryThinker(_AudioThinker):
    """Tiny actual-token loop with an invalid compound JSON-boundary token."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[int] = []

    def __call__(self, *, input_ids, **kwargs):
        if input_ids.shape[-1] == 1:
            current = int(input_ids[0, 0])
            self.inputs.append(current)
        else:
            current = -1
        logits = torch.full((1, input_ids.shape[1], 10), -100.0, dtype=torch.float32)
        preferred = {
            -1: 4,   # replaced by the forced JSON opener
            2: 6,    # begin heard value
            6: 7,    # close heard value
            7: 8,    # invalid compound token wins unless checked before cache use
            9: 3,
        }.get(current, 3)
        logits[..., preferred] = 10.0
        if current == 7:
            logits[..., 9] = 9.0  # legal ordered action-key prefix is runner-up
        hidden = torch.ones((1, input_ids.shape[1], 4), dtype=torch.float32)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=object(),
        )


class _CompoundBoundaryTokenizer(_Tokenizer):
    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        vocabulary = {
            2: "{",
            3: "",
            4: "unused",
            6: '"heard":"Marco',
            7: '"',
            8: ',"nope"',
            9: ',"action":"',
        }
        return "".join(vocabulary[int(token_id)] for token_id in token_ids)


class _BoundedBackchannelThinker(_AudioThinker):
    """Native-token fixture whose top 64 candidates all start a fifth word."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[int] = []

    def __call__(self, *, input_ids, **_kwargs):
        current = int(input_ids[0, 0]) if input_ids.shape[-1] == 1 else -1
        if current >= 0:
            self.inputs.append(current)
        logits = torch.full((1, input_ids.shape[1], 128), -100.0, dtype=torch.float32)
        preferred = {
            -1: 2,
            2: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
            10: 11,
            11: 12,
            12: 13,
            13: 14,
            14: 15,
            15: 16,
            16: 20,
            91: 3,
        }.get(current, 3)
        logits[..., preferred] = 10.0
        if current == 16:
            for offset, token_id in enumerate(range(20, 84)):
                logits[..., token_id] = 9.0 - (offset / 100.0)
            logits[..., 8] = 1.0
        hidden = torch.ones((1, input_ids.shape[1], 4), dtype=torch.float32)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=object(),
        )


class _BoundedBackchannelTokenizer(_Tokenizer):
    eos_token_id = 3

    _vocabulary = {
        2: "{",
        3: "",
        5: '"heard":""',
        6: ',"action":"',
        7: "BACKCHANNEL",
        8: '"',
        9: ',"say":"',
        10: "uno",
        11: " ",
        12: "due",
        13: " ",
        14: "tre",
        15: " ",
        16: "quattro",
        91: "}",
    }

    @classmethod
    def decode(cls, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(
            " cinque" if 20 <= int(token_id) <= 83 else cls._vocabulary[int(token_id)]
            for token_id in token_ids
        )

    @classmethod
    def encode(cls, text, *, add_special_tokens):
        assert add_special_tokens is False
        mapping = {
            "{": [2],
            '"heard":""': [5],
            ',"action":"': [6],
            "BACKCHANNEL": [7],
            '"': [8],
            ',"say":"': [9],
            "}": [91],
        }
        return mapping[text]


class _BackchannelLexiconThinker(_AudioThinker):
    """Prefer an unapproved SAY prefix while a legal phrase is runner-up."""

    def __init__(self) -> None:
        super().__init__()
        self.forwarded: list[int] = []
        self._preferred = [2, 5, 6, 7, 8, 9, 10, 8, 12, 3]

    def __call__(self, *, input_ids, **_kwargs):
        if input_ids.shape[-1] == 1:
            self.forwarded.append(int(input_ids[0, 0]))
        call_index = len(self.forwarded)
        preferred = self._preferred[call_index] if call_index < len(self._preferred) else 3
        logits = torch.full((1, input_ids.shape[1], 16), -100.0, dtype=torch.float32)
        logits[..., preferred] = 10.0
        if call_index == 6:
            logits[..., 11] = 9.0  # trained "Sì." phrase is available below the invalid argmax
        hidden = torch.ones((1, input_ids.shape[1], 4), dtype=torch.float32)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=object())


class _BackchannelLexiconTokenizer(_Tokenizer):
    _vocabulary = {
        2: "{",
        3: "",
        5: '"heard":""',
        6: ',"action":"',
        7: "BACKCHANNEL",
        8: '"',
        9: ',"say":"',
        10: "Buongiorno",
        11: "Sì.",
        12: "}",
    }

    @classmethod
    def decode(cls, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(cls._vocabulary[int(token_id)] for token_id in token_ids)

    @classmethod
    def encode(cls, text, *, add_special_tokens):
        assert add_special_tokens is False
        mapping = {
            "{": [2],
            '"heard":""': [5],
            ',"action":"': [6],
            "BACKCHANNEL": [7],
            '"': [8],
            ',"say":"': [9],
            "}": [12],
        }
        return mapping[text]


class _BackchannelPoolThinker(_AudioThinker):
    """All high-ranked body tokens are illegal; a legal phrase is lower-ranked."""

    def __init__(self) -> None:
        super().__init__()
        self.forwarded: list[int] = []

    def __call__(self, *, input_ids, **_kwargs):
        if input_ids.shape[-1] == 1:
            current = int(input_ids[0, 0])
            self.forwarded.append(current)
        else:
            current = -1
        logits = torch.full((1, input_ids.shape[1], 128), -100.0, dtype=torch.float32)
        preferred = {
            -1: 2,
            2: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 20,
            100: 8,
            12: 3,
        }.get(current, 3)
        logits[..., preferred] = 12.0
        if current == 9:
            for offset, token_id in enumerate(range(20, 84)):
                logits[..., token_id] = 10.0 - offset / 100.0
            logits[..., 100] = 2.0
            logits[..., 101] = 1.0
        hidden = torch.ones((1, input_ids.shape[1], 4), dtype=torch.float32)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=object())


class _BackchannelPoolTokenizer(_Tokenizer):
    eos_token_id = 3
    _vocabulary = {
        2: "{",
        3: "",
        5: '"heard":""',
        6: ',"action":"',
        7: "BACKCHANNEL",
        8: '"',
        9: ',"say":"',
        12: "}",
        **{token_id: " Buongiorno" for token_id in range(20, 84)},
        100: "Sì.",
        101: " Sì.",
        102: "Va",
        103: " bene",
        104: ".",
        105: " Va",
    }

    @classmethod
    def decode(cls, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(cls._vocabulary[int(token_id)] for token_id in token_ids)

    @classmethod
    def encode(cls, text, *, add_special_tokens):
        assert add_special_tokens is False
        mapping = {
            "{": [2],
            '"heard":""': [5],
            ',"action":"': [6],
            "BACKCHANNEL": [7],
            '"': [8],
            ',"say":"': [9],
            "}": [12],
            "Sì.": [100],
            " Sì.": [101],
            "Va bene.": [102, 103, 104],
            " Va bene.": [105, 103, 104],
        }
        return mapping[text]


def _engine() -> PipelineEngine:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.thinker = _AudioThinker()
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    engine._compiled_thinker = None
    engine.thinker_static_cache_max_len = 16
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = _Tokenizer()
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    engine._backchannel_token_trie = {
        (): frozenset({4}),
        (4,): frozenset({3}),
    }
    engine._backchannel_phrase_count = 1
    return engine


@pytest.mark.parametrize(
    ("disable_thinker_cadence_pacing", "expected_frame_counter"),
    [(False, 3), (True, 0)],
)
def test_v2_thinker_skips_conversational_frame_pacing_but_legacy_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
    disable_thinker_cadence_pacing: bool,
    expected_frame_counter: int,
) -> None:
    engine = _engine()
    backend = SimpleNamespace(
        _waiting_for_lm=threading.Event(),
        _frame_counter=0,
        _active_streaming=True,
    )
    engine.listener = SimpleNamespace(backend=backend)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if seconds == 0.001:
            backend._frame_counter += 1

    monkeypatch.setattr(engine_module.time, "sleep", fake_sleep)
    engine.think(
        [1],
        max_new_tokens=1,
        temperature=0.0,
        disable_thinker_cadence_pacing=disable_thinker_cadence_pacing,
    )

    assert backend._frame_counter == expected_frame_counter
    assert sleeps.count(0.001) == expected_frame_counter


def _audio_input() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "input_features": torch.ones((1, 1, 8), dtype=torch.float32),
        "input_features_mask": torch.ones((1, 1), dtype=torch.long),
    }


def test_multimodal_backchannel_does_not_apply_legacy_phrase_trie_to_json_prefix() -> None:
    engine = _engine()
    generated: list[int] = []

    result = engine.think(
        [1, 99],
        max_new_tokens=1,
        temperature=0.0,
        action="BACKCHANNEL",
        input_audio=_audio_input(),
        on_token_id=generated.append,
    )

    assert result["gen_ids"] == [2]
    assert generated == [2]
    assert result["text"] == "{"


def test_audio_response_constraint_overrides_next_token_after_native_prefix_and_logs_its_likelihood() -> None:
    engine = _engine()
    observed: list[tuple[int, int, str]] = []

    def constrain_after_json_opener(index, token_id, text, _hidden):
        observed.append((index, token_id, text))
        return 4 if index == 0 else None

    result = engine.think(
        [1, 99],
        max_new_tokens=2,
        temperature=0.0,
        action="BACKCHANNEL",
        input_audio=_audio_input(),
        on_generated_token=constrain_after_json_opener,
    )

    assert observed == [(0, 2, "{"), (1, 4, "Capito")]
    assert result["gen_ids"] == [2, 4]
    assert result["runtime_metrics"]["forced_action_token_count"] == 1
    assert result["runtime_metrics"]["forced_action_override_count"] == 1
    assert result["runtime_metrics"]["native_action_logprob"] == pytest.approx(
        float(torch.log_softmax(torch.tensor([0.0, 0.0, 10.0, 0.0, 9.0]), dim=-1)[4])
    )


def test_action_logit_probe_captures_raw_argmax_before_dfa_override() -> None:
    engine = _engine()
    class _ProbeTokenizer(_Tokenizer):
        @staticmethod
        def encode(text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"SPEAK": [0], "BACKCHANNEL": [4], "REPLAN": [1]}[text]

    engine.tokenizer = _ProbeTokenizer()
    probes: list[dict[str, object]] = []

    def constrain_after_json_opener(index, token_id, text, _hidden):
        return (4, "action") if index == 0 else None

    engine.think(
        [1, 99],
        max_new_tokens=2,
        temperature=0.0,
        input_audio=_audio_input(),
        on_generated_token=constrain_after_json_opener,
        on_action_logits_probe=probes.append,
    )

    assert len(probes) == 1
    assert probes[0] == {
        "raw_argmax_token_id": 2,
        "sampled_token_id": 2,
        "forced_token_id": 4,
        "action_label_token_ids": {
            "SPEAK": [0],
            "BACKCHANNEL": [4],
            "REPLAN": [1],
        },
        "action_label_first_token_logits": {
            "SPEAK": 0.0,
            "BACKCHANNEL": 9.0,
            "REPLAN": 0.0,
        },
        "raw_argmax_action_candidates": [],
        "best_first_logit_action_candidates": ["BACKCHANNEL"],
    }


def test_audio_response_start_constraint_overrides_first_token_as_structure() -> None:
    engine = _engine()
    engine.thinker.preferred_token_id = 4

    result = engine.think(
        [1, 99],
        max_new_tokens=1,
        temperature=0.0,
        action="BACKCHANNEL",
        input_audio=_audio_input(),
        on_generation_start=lambda: 2,
    )

    assert result["gen_ids"] == [2]
    assert result["text"] == "{"
    assert result["runtime_metrics"]["forced_action_token_count"] == 0
    assert result["runtime_metrics"]["forced_structure_token_count"] == 1
    assert result["runtime_metrics"]["forced_structure_override_count"] == 1


def test_legacy_text_backchannel_still_uses_phrase_trie() -> None:
    engine = _engine()

    result = engine.think(
        [1],
        max_new_tokens=1,
        temperature=0.0,
        action="BACKCHANNEL",
    )

    assert result["gen_ids"] == [4]
    assert result["text"] == "Capito"


def test_generated_token_callback_gets_same_forward_hidden_and_never_eos() -> None:
    engine = _engine()

    class _ForwardSentinelThinker(_AudioThinker):
        def __init__(self) -> None:
            super().__init__()
            self.forwarded: list[int] = []

        def __call__(self, *, input_ids, **_kwargs):
            ids = [int(value) for value in input_ids.reshape(-1).tolist()]
            self.forwarded.extend(ids)
            logits = torch.zeros((1, input_ids.shape[1], 5), dtype=torch.float32)
            if ids == [1]:
                logits[..., 4] = 10.0
            else:
                logits[..., 3] = 10.0
            hidden = torch.full((1, input_ids.shape[1], 4), float(ids[-1]))
            return SimpleNamespace(
                logits=logits,
                hidden_states=(hidden,),
                past_key_values=object(),
            )

    thinker = _ForwardSentinelThinker()
    engine.thinker = thinker
    observed: list[tuple[int, float, tuple[int, ...]]] = []

    def callback(_index: int, token_id: int, _text: str, hidden: torch.Tensor):
        observed.append((token_id, float(hidden[0, 0, 0]), tuple(thinker.forwarded)))

    result = engine.think(
        [1],
        max_new_tokens=3,
        temperature=0.0,
        on_generated_token=callback,
    )

    assert result["gen_ids"] == [4]
    assert result["terminal_reason"] == "eos"
    assert observed == [(4, 4.0, (1, 4))]


def test_compound_token_with_bad_action_key_is_rejected_before_cache_forward() -> None:
    engine = _engine()
    engine.thinker = _CompoundBoundaryThinker()
    engine.tokenizer = _CompoundBoundaryTokenizer()
    collector = AudioResponseV2Collector(expected_action="SPEAK")

    def collect(index, token_id, token_text, hidden):
        collector.feed_token(
            token_index=index,
            token_id=token_id,
            text=token_text,
            hidden=hidden,
        )

    result = engine.think(
        [1, 99],
        max_new_tokens=4,
        temperature=0.0,
        input_audio=_audio_input(),
        on_generation_start=lambda: 2,
        on_generated_token=collect,
        on_candidate_token=lambda _token_id, text, terminal: collector.is_valid_next_text(
            text,
            terminal=terminal,
        ),
    )

    assert result["gen_ids"] == [2, 6, 7, 9]
    assert engine.thinker.inputs == [2, 6, 7, 9]
    assert collector.parser.phase == "action_value"


def test_backchannel_closure_fallback_is_selected_before_thinker_cache_forward() -> None:
    engine = _engine()
    engine.thinker = _BoundedBackchannelThinker()
    engine.tokenizer = _BoundedBackchannelTokenizer()
    collector = AudioResponseV2Collector(expected_action="BACKCHANNEL")
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=engine.tokenizer,
        enforce_json_structure=True,
    )

    def collect(index, token_id, token_text, hidden):
        collector.feed_token(
            token_index=index,
            token_id=token_id,
            text=token_text,
            hidden=hidden,
        )
        forced = constraint.next_token_id(collector)
        if forced is None:
            return None
        return forced, "structure"

    result = engine.think(
        [1, 99],
        max_new_tokens=24,
        temperature=0.0,
        input_audio=_audio_input(),
        on_generation_start=constraint.initial_token_id,
        on_generated_token=collect,
        on_candidate_token=lambda _token_id, text, terminal: collector.is_valid_next_text(
            text,
            terminal=terminal,
            max_say_words=4,
        ),
        on_candidate_fallback=lambda rejected: constraint.bounded_say_closure_token_id(
            collector,
            rejected,
            max_say_words=4,
        ),
    )

    parsed = collector.finish(expected_text=result["text"])
    assert parsed == {"heard": "", "action": "BACKCHANNEL", "say": "uno due tre quattro"}
    assert 8 in result["gen_ids"]
    assert 91 in result["gen_ids"]
    assert not any(20 <= token_id <= 83 for token_id in engine.thinker.inputs)
    assert constraint.bounded_output is True
    assert result["runtime_metrics"]["audio_response_protocol_v2_bounded_closures"] == 1
    assert [token.token_id for token in collector.say_tokens] == [10, 11, 12, 13, 14, 15, 16]


def test_backchannel_say_lexicon_replaces_echo_candidate_before_thinker_cache_forward() -> None:
    engine = _engine()
    engine.thinker = _BackchannelLexiconThinker()
    engine.tokenizer = _BackchannelLexiconTokenizer()
    collector = AudioResponseV2Collector(
        expected_action="BACKCHANNEL",
        backchannel_say_phrases=("Sì.", "Va bene."),
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=engine.tokenizer,
        enforce_json_structure=True,
    )

    def collect(index, token_id, token_text, hidden):
        collector.feed_token(
            token_index=index,
            token_id=token_id,
            text=token_text,
            hidden=hidden,
        )
        forced = constraint.next_token_id(collector)
        if forced is None:
            return None
        return forced, "structure"

    result = engine.think(
        [1, 99],
        max_new_tokens=16,
        temperature=0.0,
        input_audio=_audio_input(),
        on_generation_start=constraint.initial_token_id,
        on_generated_token=collect,
        on_candidate_token=lambda _token_id, text, terminal: collector.is_valid_next_text(
            text,
            terminal=terminal,
            max_say_words=4,
        ),
        on_candidate_fallback=lambda rejected: constraint.bounded_say_closure_token_id(
            collector,
            rejected,
            max_say_words=4,
        ),
    )

    parsed = collector.finish(expected_text=result["text"])
    assert parsed == {"heard": "", "action": "BACKCHANNEL", "say": "Sì."}
    assert result["gen_ids"] == [2, 5, 6, 7, 8, 9, 11, 8, 12]
    assert engine.thinker.forwarded == [2, 5, 6, 7, 8, 9, 11, 8, 12]
    assert 10 not in engine.thinker.forwarded
    assert result["runtime_metrics"]["audio_response_protocol_v2_predecode_replacements"] == 1
    assert collector.backchannel_say_lexicon_rejections >= 1


def test_backchannel_say_lexicon_allows_only_train_phrase_prefixes_and_complete_phrase_closure() -> None:
    phrases = ("Sì.", "Va bene.")
    tokenizer = _BackchannelLexiconTokenizer()
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )

    va = AudioResponseV2Collector(
        expected_action="BACKCHANNEL",
        backchannel_say_phrases=phrases,
    )
    va.parser.feed('{"heard":"","action":"BACKCHANNEL","say":"Va')
    assert not va.is_valid_next_text('"', terminal=False)
    assert va.is_valid_next_text(" bene.", terminal=False)
    assert constraint.bounded_say_closure_token_id(
        va, [(10, " untrained", False)], max_say_words=4,
    ) is None

    complete = AudioResponseV2Collector(
        expected_action="BACKCHANNEL",
        backchannel_say_phrases=phrases,
    )
    complete.parser.feed('{"heard":"","action":"BACKCHANNEL","say":"Sì.')
    assert complete.backchannel_say_is_complete_phrase
    assert constraint.bounded_say_closure_token_id(
        complete, [(10, " untrained", False)], max_say_words=4,
    ) == 8

    speak = AudioResponseV2Collector(
        expected_action="SPEAK",
        backchannel_say_phrases=phrases,
    )
    speak.parser.feed('{"heard":"trascrizione","action":"SPEAK","say":"')
    assert speak.is_valid_next_text("Buongiorno", terminal=False)


def test_backchannel_fallback_uses_native_logit_over_tokenizer_lexicon_pool() -> None:
    engine = _engine()
    engine.thinker = _BackchannelPoolThinker()
    engine.tokenizer = _BackchannelPoolTokenizer()
    phrases = ("Sì.", "Va bene.")
    collector = AudioResponseV2Collector(
        expected_action="BACKCHANNEL",
        backchannel_say_phrases=phrases,
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=engine.tokenizer,
        enforce_json_structure=True,
    )

    def collect(index, token_id, token_text, hidden):
        collector.feed_token(
            token_index=index,
            token_id=token_id,
            text=token_text,
            hidden=hidden,
        )
        forced = constraint.next_token_id(collector)
        return None if forced is None else (forced, "structure")

    result = engine.think(
        [1, 99],
        max_new_tokens=16,
        temperature=0.0,
        input_audio=_audio_input(),
        on_generation_start=constraint.initial_token_id,
        on_generated_token=collect,
        on_candidate_token=lambda _token_id, text, terminal: collector.is_valid_next_text(
            text,
            terminal=terminal,
            max_say_words=4,
        ),
        on_candidate_fallback=lambda rejected: (
            constraint.bounded_say_closure_token_id(
                collector,
                rejected,
                max_say_words=4,
            )
            or constraint.backchannel_say_candidate_token_ids(collector)
        ),
    )

    assert collector.finish(expected_text=result["text"]) == {
        "heard": "",
        "action": "BACKCHANNEL",
        "say": "Sì.",
    }
    assert result["gen_ids"] == [2, 5, 6, 7, 8, 9, 100, 8, 12]
    assert engine.thinker.forwarded[-3:] == [100, 8, 12]
    assert not any(20 <= token_id <= 83 for token_id in engine.thinker.forwarded)


def test_real_gemma_tokenizer_backchannel_pool_can_escape_invalid_top64() -> None:
    from transformers import AutoTokenizer

    tokenizer_path = Path("/workspace/.hf_cache/models--google--gemma-4-E2B-it/full")
    if not tokenizer_path.is_dir():
        pytest.skip("the cached Gemma tokenizer is available only in the remote CPU image")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    collector = AudioResponseV2Collector(
        expected_action="BACKCHANNEL",
        backchannel_say_phrases=("Sì.", "Va bene."),
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="BACKCHANNEL",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )
    budget = constraint.preflight_generation_budget(
        collector,
        max_new_tokens=128,
    )
    assert budget["required_new_tokens"] <= 128
    assert budget["heard_token_cap"] == 0

    speak_budget = AudioResponseV2ActionConstraint(
        expected_action="SPEAK",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    ).preflight_generation_budget(
        AudioResponseV2Collector(expected_action="SPEAK"),
        max_new_tokens=128,
    )
    assert speak_budget["required_new_tokens"] <= 128

    body_prefix = '{"heard":"","action":"BACKCHANNEL","say":"'
    collector.parser.feed(body_prefix)
    phrase_prefixes = (
        "{",
        '"heard":""',
        ',"action":"',
        "BACKCHANNEL",
        '"',
        ',"say":"',
    )
    generated_prefix_ids = [
        token_id
        for text in phrase_prefixes
        for token_id in constraint._encode_roundtrippable(
            text,
            reason="json_structure_tokenization_not_roundtrippable",
        )
    ]

    def decode(token_ids):
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(token_ids, skip_special_tokens=True)

    decoded_prefix = decode(generated_prefix_ids)
    assert decoded_prefix == body_prefix
    pool = constraint.backchannel_say_candidate_token_ids(collector)
    legal_ids = []
    for token_id in pool:
        candidate_text = decode([*generated_prefix_ids, token_id])[len(decoded_prefix):]
        if collector.is_valid_next_text(candidate_text, terminal=False):
            legal_ids.append(token_id)
    assert legal_ids

    vocab_size = max(int(tokenizer.vocab_size), len(tokenizer))
    invalid_top_ids = []
    legal_set = set(legal_ids)
    for token_id in range(vocab_size):
        if token_id in legal_set:
            continue
        candidate_text = decode([*generated_prefix_ids, token_id])[len(decoded_prefix):]
        if not collector.is_valid_next_text(candidate_text, terminal=False):
            invalid_top_ids.append(token_id)
            if len(invalid_top_ids) == 64:
                break
    assert len(invalid_top_ids) == 64

    logits = torch.full((1, vocab_size), -1000.0)
    for rank, token_id in enumerate(invalid_top_ids):
        logits[0, token_id] = 100.0 - rank
    for rank, token_id in enumerate(legal_ids):
        logits[0, token_id] = -1.0 - rank
    assert set(torch.topk(logits[0], 64).indices.tolist()).isdisjoint(legal_set)

    selected = PipelineEngine._best_native_logit_candidate_id(legal_ids, logits)
    expected = max(legal_ids, key=lambda token_id: float(logits[0, token_id]))
    assert selected == expected


class _PieceTokenizer:
    """Reversible one-token-per-piece tokenizer for budget tests."""

    eos_token_id = 3

    def __init__(self) -> None:
        self._by_text: dict[str, int] = {}
        self._by_id: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        token_id = self._by_text.get(text)
        if token_id is None:
            token_id = 1000 + len(self._by_text)
            self._by_text[text] = token_id
            self._by_id[token_id] = text
        return [token_id]

    def decode(self, token_ids, *, skip_special_tokens: bool, **_kwargs) -> str:
        assert skip_special_tokens
        return "".join(self._by_id[int(token_id)] for token_id in token_ids)


def test_selected_action_fallback_pool_uses_existing_tokenizer_ids() -> None:
    tokenizer = _PieceTokenizer()
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.feed_token(
        token_index=0,
        token_id=10,
        text='{"heard":"Venezia","action":"S',
        hidden=torch.ones(1),
    )
    constraint = AudioResponseV2ActionConstraint(
        expected_action="SPEAK",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )

    candidates = constraint.action_value_candidate_token_ids(collector)
    valid = [
        candidate_id for candidate_id in candidates
        if collector.is_valid_next_text(
            tokenizer.decode([candidate_id], skip_special_tokens=True),
            terminal=False,
        )
    ]

    assert candidates
    assert valid == [tokenizer.encode("PEAK", add_special_tokens=False)[0]]


def test_exact_field_caps_are_not_called_truncated_without_forced_closure() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.set_token_caps(heard=1, action=1, say=1)
    text = '{"heard":"Ciao","action":"SPEAK","say":"Ciao"}'
    collector.feed_token(token_index=0, token_id=10, text=text, hidden=torch.ones(1))

    assert collector.heard_field_token_count == 1
    assert collector.action_field_token_count == 1
    assert collector.say_field_token_count == 1
    assert not collector.heard_truncated
    assert not collector.say_truncated
    assert collector.finish(expected_text=text) == {
        "heard": "Ciao",
        "action": "SPEAK",
        "say": "Ciao",
    }


def test_say_hidden_states_snapshot_reused_forward_storage_before_next_token() -> None:
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    storage = torch.zeros((1, 1, 2), dtype=torch.float32)
    header = '{"heard":"Ciao","action":"SPEAK","say":"'
    text = header + "ABC" + '"}'
    collector.feed_token(token_index=0, token_id=10, text=header, hidden=storage.detach())

    for index, char in enumerate("ABC", start=1):
        storage.fill_(float(index))
        collector.feed_token(
            token_index=index,
            token_id=10 + index,
            text=char,
            hidden=storage.detach(),
        )
    storage.fill_(99.0)
    collector.feed_token(token_index=4, token_id=14, text='"}', hidden=storage.detach())

    parsed, say_hidden = finish_audio_response_v2_with_say_hidden(
        collector,
        expected_text=text,
    )
    assert parsed["say"] == "ABC"
    assert say_hidden.shape == (1, 3, 2)
    assert torch.equal(say_hidden[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_capped_heard_is_marked_truncated_only_when_safe_close_is_forced() -> None:
    tokenizer = _PieceTokenizer()
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.set_token_caps(heard=1, action=8, say=48)
    collector.feed_token(
        token_index=0,
        token_id=10,
        text='{"heard":"Ciao',
        hidden=torch.ones(1),
    )
    assert collector.heard_field_token_count == 1
    assert not collector.heard_truncated

    constraint = AudioResponseV2ActionConstraint(
        expected_action="SPEAK",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )
    closure_id = constraint.bounded_field_closure_token_id(
        collector,
        [(20, "!", False)],
        budget={"heard_token_cap": 1, "action_token_cap": 8, "say_token_cap": 48},
    )
    assert closure_id == tokenizer.encode('"', add_special_tokens=False)[0]
    assert collector.heard_truncated
    assert constraint.bounded_output


def test_generation_budget_preflights_all_fields_structure_and_eos() -> None:
    constraint = AudioResponseV2ActionConstraint(
        expected_action="SPEAK",
        tokenizer=_PieceTokenizer(),
        enforce_json_structure=True,
    )
    collector = AudioResponseV2Collector(expected_action="SPEAK")

    budget = constraint.preflight_generation_budget(
        collector,
        max_new_tokens=128,
        max_heard_tokens=48,
        max_action_tokens=8,
        max_say_tokens=48,
    )

    assert budget["required_new_tokens"] == 113
    assert budget["required_new_tokens"] == (
        budget["heard_token_cap"]
        + budget["action_token_cap"]
        + budget["say_token_cap"]
        + budget["fixed_structure_tokens"]
        + budget["eos_reserve_tokens"]
    )
    with pytest.raises(AudioResponseProtocolError, match="budget_preflight_failed"):
        constraint.preflight_generation_budget(
            collector,
            max_new_tokens=112,
            max_heard_tokens=48,
            max_action_tokens=8,
            max_say_tokens=48,
        )


def test_generation_budget_reserves_tokenizer_completion_for_partial_unicode_escape() -> None:
    tokenizer = _PieceTokenizer()
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    collector.set_token_caps(heard=4, action=8, say=48)
    collector.feed_token(token_index=0, token_id=10, text='{"heard":"A', hidden=None)
    collector.feed_token(token_index=1, token_id=11, text="\\", hidden=None)
    assert collector.parser.phase == "heard_value"

    constraint = AudioResponseV2ActionConstraint(
        expected_action="SPEAK",
        tokenizer=tokenizer,
        enforce_json_structure=True,
    )
    budget = {"max_new_tokens": 128, "heard_token_cap": 4, "action_token_cap": 8, "say_token_cap": 48}

    assert not constraint.candidate_within_generation_budget(
        collector,
        "u00",
        terminal=False,
        generated_tokens=2,
        budget=budget,
    )

    budget["heard_token_cap"] = 5
    assert constraint.candidate_within_generation_budget(
        collector,
        "u00",
        terminal=False,
        generated_tokens=2,
        budget=budget,
    )
