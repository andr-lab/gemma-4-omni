from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel import server_response_runtime as response_runtime
from duplex_kernel.config import ARIA_HARD_MAX_FRAMES
from duplex_kernel.engine import PipelineEngine
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
    CONTROLLER_GUIDED_SAY_V1,
)
from duplex_kernel.live_say import TurnLocalQwenSayStream
from duplex_kernel.session import TurnLocalSpeechFIFO
from duplex_kernel.server_response_control import LONG_RESPONSE_PROFILE
from duplex_kernel.server_response_control import (
    LONG_RESPONSE_MAX_CODEC_FRAMES,
    LONG_RESPONSE_MAX_TEXT_TOKENS,
    _controller_guided_say_generation_options,
)
from duplex_kernel.server_response_runtime import _response_completion_diagnostics
from duplex_kernel.unified_native_decoder import UnifiedNativeDecoder
from duplex_kernel.listener_microbatch_worker import ListenerMicrobatchWorker


class _FrameBudgetTalkerModel:
    def __init__(self) -> None:
        self.model = self
        self.codec_head = _FrameBudgetCodecHead()

    def __call__(self, *, inputs_embeds, use_cache, past_key_values=None):
        del use_cache, past_key_values
        return SimpleNamespace(
            last_hidden_state=inputs_embeds[:, -1:, :],
            past_key_values=object(),
        )


class _FrameBudgetCodecHead:
    out_features = 3072

    def __call__(self, hidden):
        return torch.zeros((*hidden.shape[:-1], self.out_features))


class _Tokenizer:
    eos_token_id = 99

    def __init__(self, pieces: dict[int, str] | None = None) -> None:
        self.pieces = pieces or {}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.pieces.get(int(token_id), "a") for token_id in token_ids)


def _thinker(tokenizer: _Tokenizer) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99])),
        tokenizer=tokenizer,
    )


def _native_decoder(tokenizer: _Tokenizer, sampled_ids):
    samples = iter(sampled_ids)

    async def forward_tokens(token_ids):
        return torch.zeros((1, 8)), torch.tensor([int(token_ids[-1])], dtype=torch.float32)

    def sample_token(_logits, _history):
        token_id = next(samples)
        return token_id

    return UnifiedNativeDecoder(
        _thinker(tokenizer),
        forward_tokens=forward_tokens,
        sample_token=sample_token,
        eos_token_ids={99},
    )


def test_long_profile_requests_768_text_tokens_and_1536_codec_frames():
    long = _controller_guided_say_generation_options(
        action="SPEAK",
        controller_guided_say=True,
        direct_audio_candidate=False,
        response_protocol=None,
        response_profile="long",
    )
    alias = _controller_guided_say_generation_options(
        action="SPEAK",
        controller_guided_say=True,
        direct_audio_candidate=False,
        response_protocol=None,
        response_profile="long_response",
    )
    ordinary = _controller_guided_say_generation_options(
        action="SPEAK",
        controller_guided_say=True,
        direct_audio_candidate=False,
        response_protocol=None,
    )

    assert long["max_tokens"] == alias["max_tokens"] == LONG_RESPONSE_MAX_TEXT_TOKENS == 768
    assert LONG_RESPONSE_MAX_CODEC_FRAMES == 1536
    assert ordinary["max_tokens"] == 256


def test_v2_backchannel_reserves_generation_budget_for_json_envelope():
    options = _controller_guided_say_generation_options(
        action="BACKCHANNEL",
        controller_guided_say=False,
        direct_audio_candidate=True,
        response_protocol=AUDIO_RESPONSE_PROTOCOL_V2,
    )

    assert options["max_tokens"] == AUDIO_RESPONSE_V2_MAX_NEW_TOKENS
    assert options["_min_new_tokens"] == 0


def test_completion_diagnostics_preserve_requested_effective_and_kv_cap():
    completion = _response_completion_diagnostics(
        token_limit_requested=768,
        token_limit=120,
        token_limit_effective=120,
        token_limit_cap_reason="kv_capacity",
        tokens_generated=120,
        text_terminal_reason="budget_exhausted",
        codec_metrics={"codec_frame_limit": 1536},
        audio_emitted=False,
    )

    assert completion["token_limit_requested"] == 768
    assert completion["token_limit_effective"] == 120
    assert completion["token_limit_cap_reason"] == "kv_capacity"
    assert completion["finish_reason"] == "length"


def test_native_preflight_caps_long_budget_before_decode_for_kv_and_audio_retention():
    worker = object.__new__(ListenerMicrobatchWorker)
    session_id = "long-profile-cap"
    worker._active_queued_audio_drains = set()
    worker._coordinator = SimpleNamespace(
        capability=SimpleNamespace(max_seq_len=800),
    )
    worker._outstanding_frames = {session_id: 0}
    worker._tick_expected = None
    worker._queued_audio_soft = {session_id: (object(), object(), object())}
    worker._policy_q2_soft_pending = {session_id: (object(),)}
    worker._reserved_tail_tokens = {session_id: 4}
    worker._check_unified_session_cas = lambda *_args, **_kwargs: SimpleNamespace(
        generation_id=7,
        cache_position=40,
    )

    reservation = worker._reserve_response_decode_budget(
        session_id,
        expected_generation_id=7,
        expected_cache_position=40,
        requested_tokens=768,
        transition_tokens=2,
        sentence_completion_tokens=32,
    )

    assert reservation["effective_tokens"] == 746
    assert reservation["sentence_completion_tokens"] == 0
    assert reservation["cap_reason"] == "kv_capacity"
    assert reservation["queued_audio_tokens"] == 6
    assert reservation["policy_pending_audio_tokens"] == 2
    assert reservation["close_reservation_tokens"] == 4
    assert reservation["reservation_tokens"] == 752


def test_natural_eos_within_long_budget_finishes_as_stop():
    decoder = _native_decoder(
        _Tokenizer({10: "Ciao", 11: "."}),
        (10, 11, 99),
    )

    decoded = asyncio.run(decoder.decode(
        transition_token_ids=[1],
        max_new_tokens=768,
        graceful_sentence_completion=True,
        sentence_completion_max_tokens=32,
    ))

    assert decoded.termination_reason == "eos"
    assert decoded.finish_reason == "stop"
    assert [token.token_id for token in decoded.generated_tokens] == [10, 11]


def test_native_sentence_completion_uses_at_most_32_extra_tokens_and_keeps_length():
    tokenizer = _Tokenizer({10: "Ciao", 11: " mondo", 12: ".", 13: " Altro"})
    decoder = _native_decoder(tokenizer, (10, 11, 12, 13))
    published: list[int] = []

    decoded = asyncio.run(decoder.decode(
        transition_token_ids=[1],
        max_new_tokens=2,
        graceful_sentence_completion=True,
        sentence_completion_max_tokens=32,
        on_token=lambda token_id, _hidden, _timing: published.append(token_id),
    ))

    assert published == [10, 11, 12]
    assert [token.token_id for token in decoded.generated_tokens] == [10, 11, 12]
    assert decoded.termination_reason == "sentence_boundary_after_budget"
    assert decoded.finish_reason == "length"

    capped = _native_decoder(_Tokenizer(), range(10, 45))
    exhausted = asyncio.run(capped.decode(
        transition_token_ids=[1],
        max_new_tokens=1,
        graceful_sentence_completion=True,
        sentence_completion_max_tokens=32,
    ))
    assert len(exhausted.generated_tokens) == 33
    assert exhausted.termination_reason == "budget_exhausted"
    assert exhausted.finish_reason == "length"


def test_turn_local_qwen_default_source_capacity_scales_to_long_profile():
    stream = TurnLocalQwenSayStream(
        qwen_tokenizer=object(),
        text_eos_token_id=99,
        fifo=TurnLocalSpeechFIFO(),
        generation_id=1,
        max_tokens=3072,
    )

    assert stream.max_source_chars == 98_304


def test_long_profile_passes_scaled_capacities_to_live_qwen_stream(monkeypatch):
    captured: dict[str, object] = {}

    class _StopAtLiveStream(Exception):
        pass

    def capture_live_stream(**kwargs):
        captured.update(kwargs)
        raise _StopAtLiveStream

    class _ThinkerContext:
        system_prompt = None

        def set_messages(self, messages):
            self._messages = list(messages)

        def messages(self):
            return list(self._messages)

    monkeypatch.setattr(
        response_runtime,
        "_configured_audio_response_protocol",
        lambda _engine: CONTROLLER_GUIDED_SAY_V1,
    )
    monkeypatch.setattr(
        response_runtime,
        "_direct_audio_prompt_history",
        lambda clean, *, history_is_committed: clean,
    )
    monkeypatch.setattr(
        response_runtime,
        "_prepare_direct_audio_processor_inputs",
        lambda *_args, **_kwargs: ([1], None, 0.0),
    )
    monkeypatch.setattr(
        response_runtime,
        "_resolve_request_voice_conditioning",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(response_runtime, "LiveQwenSayStream", capture_live_stream)

    engine = SimpleNamespace(
        talker=SimpleNamespace(
            proc=SimpleNamespace(tokenizer=object()),
            cfg=SimpleNamespace(tts_eos_token_id=99),
        ),
    )
    session = SimpleNamespace(
        instructions="",
        tools=(),
        response_profile="standard",
        language="it",
        _generation_lock=threading.RLock(),
        thinker_ctx=_ThinkerContext(),
        is_generation_current=lambda *, generation_id: generation_id == 1,
    )
    request = {
        "_generation_id": 1,
        "_direct_audio_pcm": [0.0],
        "modalities": ["audio"],
        "response_profile": LONG_RESPONSE_PROFILE,
        "max_tokens": LONG_RESPONSE_MAX_TEXT_TOKENS,
    }

    with pytest.raises(_StopAtLiveStream):
        response_runtime.run_response(
            engine,
            session,
            [{"role": "user", "content": "rispondi"}],
            request,
            action="SPEAK",
        )

    assert captured["max_tokens"] == 3072
    assert captured["max_source_chars"] == 98_304
    assert captured["max_source_pieces"] == 12_288
    assert captured["max_retokenization_work"] == 72_000_000


def test_acoustic_producer_limit_stops_native_thinker_without_cancellation():
    decoder = _native_decoder(_Tokenizer(), (10, 11, 12))
    producer_limit = threading.Event()
    emitted: list[int] = []

    def on_token(token_id, _hidden, _timing):
        emitted.append(token_id)
        producer_limit.set()

    decoded = asyncio.run(decoder.decode(
        transition_token_ids=[1],
        max_new_tokens=768,
        graceful_sentence_completion=True,
        sentence_completion_max_tokens=32,
        producer_limit_check=producer_limit.is_set,
        on_token=on_token,
    ))

    assert emitted == [10]
    assert decoded.termination_reason == "producer_limit"
    assert decoded.finish_reason == "length"
    assert decoded.finish_reason != "cancelled"


def test_live_long_response_frame_budget_uses_configured_1536_cap():
    model = _FrameBudgetTalkerModel()

    class _Talker:
        tcfg = SimpleNamespace(codec_eos_token_id=2150)
        talker = model

        def build_prefill_parts(
            self,
            _text,
            *,
            raw_text_token_ids,
            include_text_eos,
            voice_conditioning=None,
        ):
            assert raw_text_token_ids == [7]
            assert include_text_eos is False
            assert voice_conditioning is None
            part = torch.zeros((1, 1, 4))
            return part, part, part, part

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = _Talker()
    engine.subtalker_graph = object()
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine._stop_event = False
    engine.allow_talker_eos = True
    engine._cb0_sample = lambda *_args, **_kwargs: 2150
    engine._codec_eos_allowed = lambda _generated: True
    stream = SimpleNamespace(
        visible_text=" ".join(["parola"] * 240),
        observed_qwen_token_count=700,
        codec_eos_allowed=True,
    )
    metrics: dict[str, object] = {}

    engine.talk(
        "risposta",
        torch.empty((1, 0, 4)),
        n_words=1,
        min_frames=48,
        max_frames=1536,
        metrics=metrics,
        raw_text_token_ids=[7],
        include_text_eos=False,
        live_qwen_text_stream=stream,
    )

    assert ARIA_HARD_MAX_FRAMES == 1536
    assert metrics["configured_max_frames"] == 1536
    assert metrics["requested_acoustic_frame_budget"] == 1536
    assert metrics["live_qwen_text_token_count"] == 700
    assert metrics["effective_max_frames"] == 1536
    assert metrics["live_text_eos_drained_frame_cap"] == 1536
