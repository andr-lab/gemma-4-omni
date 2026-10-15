from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from duplex_kernel.engine import PipelineEngine, _estimate_talker_frame_budget
from duplex_kernel.future_prefix import ConditioningFrameBuffer


EOS_ID = 2150
VOCAB_SIZE = 3072
HIDDEN_SIZE = 4


class _FakeCodecHead:
    out_features = VOCAB_SIZE

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*hidden.shape[:-1], VOCAB_SIZE),
            dtype=hidden.dtype,
            device=hidden.device,
        )


class _FakeTalkerConfig:
    codec_eos_token_id = EOS_ID


class _FakeQwenTalker:
    codec_head = _FakeCodecHead()


class _FakeTalker:
    def __init__(self) -> None:
        self.tcfg = _FakeTalkerConfig()
        self.talker = _FakeQwenTalker()


def _engine_for_sampling(*, allow_talker_eos: bool = True) -> PipelineEngine:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = _FakeTalker()
    engine.allow_talker_eos = allow_talker_eos
    return engine


def test_cb0_mask_preserves_only_codec_eos_above_audio_vocab() -> None:
    engine = _engine_for_sampling()
    logits = torch.zeros(VOCAB_SIZE)
    logits[42] = 5
    logits[2047] = 6
    logits[2048] = 100
    logits[2100] = 100
    logits[EOS_ID] = 7
    logits[2500] = 100

    masked = engine._mask_cb0_logits(logits)

    assert torch.isfinite(masked[42])
    assert torch.isfinite(masked[2047])
    assert torch.isfinite(masked[EOS_ID])
    assert masked[2048] == -float("inf")
    assert masked[2100] == -float("inf")
    assert masked[2500] == -float("inf")


def test_cb0_mask_suppresses_codec_eos_when_disabled() -> None:
    engine = _engine_for_sampling(allow_talker_eos=False)
    logits = torch.zeros(VOCAB_SIZE)
    logits[EOS_ID] = 7

    masked = engine._mask_cb0_logits(logits, allow_eos=False)

    assert masked[EOS_ID] == -float("inf")


def test_codec_eos_must_be_a_special_token() -> None:
    engine = _engine_for_sampling()
    engine.talker.tcfg.codec_eos_token_id = 42

    with pytest.raises(RuntimeError, match="codec EOS"):
        engine._codec_eos_token_id()


def test_non_eos_cb0_sampling_preserves_legacy_rng_and_topk() -> None:
    engine = _engine_for_sampling(allow_talker_eos=False)
    logits = torch.linspace(-3.0, 3.0, VOCAB_SIZE)
    logits[2048:] = torch.arange(VOCAB_SIZE - 2048, dtype=logits.dtype)

    torch.manual_seed(1234)
    actual = engine._cb0_sample(logits, temperature=0.6, allow_eos=False)

    torch.manual_seed(1234)
    legacy = logits.clone()
    legacy[2048:] = -float("inf")
    topk_vals, topk_idx = torch.topk(legacy, k=20, dim=-1)
    filtered = torch.full_like(legacy, -float("inf"))
    filtered.scatter_(-1, topk_idx, topk_vals)
    probs = F.softmax(filtered / 0.6, dim=-1)
    expected = int(torch.multinomial(probs, 1).item())

    assert actual == expected


class _CountingSubTalker:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return torch.zeros(15, dtype=torch.long)


class _CountingTalkerModel:
    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.model = self
        self.codec_head = _FakeCodecHead()
        self.code_predictor = SimpleNamespace(
            get_input_embeddings=lambda: [torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE) for _ in range(15)]
        )
        self._embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.model_calls = 0

    def parameters(self):
        return iter((self._parameter,))

    def get_input_embeddings(self):
        return self._embedding

    def __call__(self, *, inputs_embeds, use_cache, past_key_values=None):
        del use_cache, past_key_values
        self.model_calls += 1
        return SimpleNamespace(
            last_hidden_state=inputs_embeds[:, -1:, :],
            past_key_values=object(),
        )


class _CountingLegacyTalker:
    def __init__(self) -> None:
        self.tcfg = _FakeTalkerConfig()
        self.talker = _CountingTalkerModel()
        self._decode_calls = 0

    def build_prefill_parts(self, text: str):
        del text
        zero = torch.zeros((1, 1, HIDDEN_SIZE))
        return zero, zero, zero, zero

    def decode(
        self, codes: torch.Tensor, *, preclamp_callback=None
    ) -> torch.Tensor:
        del preclamp_callback
        self._decode_calls += 1
        return torch.zeros((codes.shape[1] * 1920,), dtype=torch.float32)


class _RolloverStatefulSpeaker:
    instances: list["_RolloverStatefulSpeaker"] = []

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.max_prefix_frames = 64
        self.pushes: list[int] = []
        self.local_frame = 0
        self.commits = 0
        self.closed = False
        type(self).instances.append(self)

    def begin(self, text: str, voice_conditioning=None) -> None:
        del text, voice_conditioning
        self.local_frame = 0

    def push_v9_frame(self, frame_id: int, frame: torch.Tensor) -> None:
        del frame
        self.pushes.append(frame_id)

    def step_frame(self, *, allow_reduced_horizon: bool = False):
        del allow_reduced_horizon
        output = SimpleNamespace(
            logits=torch.zeros((1, 1, VOCAB_SIZE)),
            last_hidden=torch.zeros((1, 1, HIDDEN_SIZE)),
            frame_id=self.local_frame,
        )
        self.local_frame += 1
        return output

    def commit_codes(self, codes) -> None:
        del codes
        self.commits += 1

    def can_terminal_query(self) -> bool:
        return self.local_frame > 0

    def step_terminal(self):
        return SimpleNamespace(
            logits=torch.zeros((1, 1, VOCAB_SIZE)),
            last_hidden=torch.zeros((1, 1, HIDDEN_SIZE)),
            frame_id=self.local_frame,
            is_terminal_query=True,
        )

    def close(self) -> None:
        self.closed = True


class _EosStatefulSpeaker:
    instances: list["_EosStatefulSpeaker"] = []

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.max_prefix_frames = 64
        self.pushes = 0
        self.commit_calls = 0
        self.closed = False
        type(self).instances.append(self)

    def begin(self, text: str, voice_conditioning=None) -> None:
        del text, voice_conditioning

    def push_v9_frame(self, frame_id: int, frame: torch.Tensor) -> None:
        del frame_id, frame
        self.pushes += 1

    def step_frame(self, *, allow_reduced_horizon: bool = False):
        del allow_reduced_horizon
        return SimpleNamespace(
            logits=torch.zeros((1, 1, VOCAB_SIZE)),
            last_hidden=torch.zeros((1, 1, HIDDEN_SIZE)),
            frame_id=0,
        )

    def commit_codes(self, codes) -> None:
        del codes
        self.commit_calls += 1

    def close(self) -> None:
        self.closed = True


def _engine_for_termination(talker) -> PipelineEngine:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = talker
    engine.subtalker_graph = _CountingSubTalker()
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.runtime_backend = "pytorch"
    engine._stop_event = False
    engine._talker_lock = threading.RLock()
    engine.allow_talker_eos = True
    engine._postprocess = lambda wav: wav
    return engine


def test_eos_is_not_sent_to_subtalker_or_decoder() -> None:
    talker = _CountingLegacyTalker()
    engine = _engine_for_termination(talker)
    sampled: list[bool] = []
    engine._cb0_sample = lambda logits, temperature=0.6, **kwargs: (
        sampled.append(kwargs["allow_eos"]), EOS_ID
    )[1]
    metrics: dict[str, object] = {}

    output = engine.talk(
        "ciao",
        torch.zeros((1, 1, HIDDEN_SIZE)),
        n_words=1,
        min_frames=1,
        metrics=metrics,
    )

    assert output.size == 0
    assert sampled == [True]
    assert engine.subtalker_graph.calls == 0
    assert talker._decode_calls == 0
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 0
    assert metrics["generated_cb0_frames"] == 0
    assert metrics["terminated_by_budget"] is False


def test_eos_flushes_previously_generated_streaming_audio() -> None:
    talker = _CountingLegacyTalker()
    engine = _engine_for_termination(talker)
    sampled = iter([42, EOS_ID])
    engine._cb0_sample = lambda logits, temperature=0.6, **kwargs: next(sampled)
    callbacks: list[np.ndarray] = []
    metrics: dict[str, object] = {}

    output = engine.talk(
        "ciao",
        torch.zeros((1, 1, HIDDEN_SIZE)),
        n_words=1,
        min_frames=1,
        on_audio_chunk=callbacks.append,
        metrics=metrics,
    )

    assert output.size == 1920
    assert len(callbacks) == 1
    assert engine.subtalker_graph.calls == 1
    assert talker._decode_calls == 1
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 1
    assert metrics["generated_cb0_frames"] == 1
    assert metrics["terminated_by_budget"] is False


def test_stateful_rollover_keeps_64_frame_acoustic_budget_and_terminal_query(monkeypatch: pytest.MonkeyPatch) -> None:
    talker = _CountingLegacyTalker()
    engine = _engine_for_termination(talker)
    monkeypatch.setattr("duplex_kernel.engine.FuturePrefixQwen", _RolloverStatefulSpeaker)
    samples = iter([42] * 65 + [EOS_ID])
    engine._cb0_sample = lambda logits, temperature=0.6, **kwargs: next(samples)
    metrics: dict[str, object] = {}
    frame_ids: list[int] = []
    chunks: list[np.ndarray] = []

    output = engine.talk_stateful(
        "ciao",
        torch.zeros((1, 65, 2048)),
        n_words=1,
        min_frames=65,
        on_cb0_frame=lambda frame_id, _timestamp: frame_ids.append(frame_id),
        on_audio_chunk=chunks.append,
        metrics=metrics,
    )

    assert output.size == 65 * 1920
    assert frame_ids == list(range(65))
    assert chunks
    assert metrics["generated_cb0_frames"] == 65
    assert metrics["segment_rollovers"] == 1
    assert metrics["terminal_queries"] == 1
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 65
    assert engine.subtalker_graph.calls == 65
    assert len(_RolloverStatefulSpeaker.instances[-1].pushes) == 65
    assert _RolloverStatefulSpeaker.instances[-1].pushes[-1] == 0


class _BufferedRolloverSpeaker:
    instances: list["_BufferedRolloverSpeaker"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.max_prefix_frames = int(kwargs.get("max_prefix_frames", 64))
        self.max_audio_frames = int(kwargs.get("max_audio_frames", 64))
        self.buffer = SimpleNamespace(buffered_frames=0, available_through=-1)
        self.local_frame = 0
        self.segment_pushes: list[list[int]] = []
        self.segment_texts: list[str] = []
        self.closed = False
        type(self).instances.append(self)

    def runtime_info(self) -> dict[str, object]:
        return {"runtime_backend": "test_fake", "native_enabled": False}

    def begin(self, text: str, voice_conditioning=None) -> None:
        del voice_conditioning
        self.local_frame = 0
        self.buffer.buffered_frames = 0
        self.buffer.available_through = -1
        self.segment_pushes.append([])
        self.segment_texts.append(text)

    def push_v9_frame(self, frame_id: int, frame: torch.Tensor) -> None:
        assert frame_id == self.buffer.buffered_frames
        assert torch.count_nonzero(frame) == 0
        self.segment_pushes[-1].append(frame_id)
        self.buffer.buffered_frames += 1
        self.buffer.available_through = self.buffer.buffered_frames - 1

    def step_frame(self, *, allow_reduced_horizon: bool = False):
        del allow_reduced_horizon
        output = SimpleNamespace(
            logits=torch.zeros((1, 1, VOCAB_SIZE)),
            last_hidden=torch.zeros((1, 1, HIDDEN_SIZE)),
            frame_id=self.local_frame,
        )
        self.local_frame += 1
        return output

    def commit_codes(self, codes) -> None:
        del codes

    def can_terminal_query(self) -> bool:
        return self.local_frame > 0

    def step_terminal(self):
        return SimpleNamespace(
            logits=torch.zeros((1, 1, VOCAB_SIZE)),
            last_hidden=torch.zeros((1, 1, HIDDEN_SIZE)),
            frame_id=self.local_frame,
            is_terminal_query=True,
        )

    def close(self) -> None:
        self.closed = True


def test_native_stateful_runtime_reaches_audio_only_terminal_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_for_termination(_CountingLegacyTalker())
    engine.runtime_backend = "native"
    monkeypatch.setattr("duplex_kernel.engine.FuturePrefixQwen", _RolloverStatefulSpeaker)
    samples = iter([42] * 6 + [EOS_ID])
    engine._cb0_sample = lambda *args, **kwargs: next(samples)
    metrics: dict[str, object] = {}

    output = engine.talk_stateful(
        "ciao",
        torch.zeros((1, 6, 2048)),
        n_words=1,
        min_frames=6,
        metrics=metrics,
    )

    assert output.size == 6 * 1920
    assert metrics["generated_cb0_frames"] == 6
    assert metrics["terminal_queries"] == 1
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 6
    assert engine.subtalker_graph.calls == 6
    eos_queries = metrics["cb0_eos_diagnostics"]["queries"]
    assert len(eos_queries) == 7
    assert eos_queries[-1]["kind"] == "terminal"
    assert eos_queries[-1]["frame_id"] == 6
    assert _RolloverStatefulSpeaker.instances[-1].closed is True


def test_stateful_multisegment_rejects_unmapped_trailing_text_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_for_termination(_CountingLegacyTalker())
    monkeypatch.setattr("duplex_kernel.engine.FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S", 0.01)
    buffer = ConditioningFrameBuffer(max_frames=8)
    buffer.begin_segment(generation_id=1, segment_id="response", base_frame_id=0)
    text = "one two three four five six seven eight nine ten eleven twelve"

    with pytest.raises(ValueError, match="trailing_text_hidden richiede"):
        engine.talk_stateful(
            text,
            n_words=12,
            conditioning_buffer=buffer,
            trailing_text_hidden=torch.zeros((1, 12, HIDDEN_SIZE)),
        )


def test_stateful_conditioning_timeout_is_not_reported_as_successful_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_for_termination(_CountingLegacyTalker())
    monkeypatch.setattr("duplex_kernel.engine.FuturePrefixQwen", _BufferedRolloverSpeaker)
    monkeypatch.setattr("duplex_kernel.engine.FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S", 0.01)
    buffer = ConditioningFrameBuffer(max_frames=8)
    buffer.begin_segment(generation_id=1, segment_id="response", base_frame_id=0)

    with pytest.raises(TimeoutError, match="Qwen conditioning frame non disponibile"):
        engine.talk_stateful(
            "ciao",
            n_words=1,
            conditioning_buffer=buffer,
        )

    assert _BufferedRolloverSpeaker.instances[-1].closed is True


def test_stateful_intermediate_segment_eos_continues_with_remaining_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_for_termination(_CountingLegacyTalker())
    monkeypatch.setattr("duplex_kernel.engine.FuturePrefixQwen", _BufferedRolloverSpeaker)
    monkeypatch.setattr("duplex_kernel.engine.FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S", 0.01)
    buffer = ConditioningFrameBuffer(max_frames=128)
    buffer.begin_segment(generation_id=1, segment_id="response", base_frame_id=0)
    for frame_id in range(1, 109):
        assert buffer.append_values(
            frame_id=frame_id,
            h_pfx=torch.zeros((1, 2048)),
            produced_at_ns=frame_id,
            generation_id=1,
            segment_id="response",
        )

    text_words = [f"word{index}" for index in range(20)]
    text = " ".join(text_words)
    samples = iter([42, 42, EOS_ID, 42, 42, 42, 42, EOS_ID])
    engine._cb0_sample = lambda logits, temperature=0.6, **kwargs: next(samples)
    metrics: dict[str, object] = {}
    output = engine.talk_stateful(
        text,
        n_words=20,
        conditioning_buffer=buffer,
        metrics=metrics,
    )

    assert output.size == 6 * 1920
    assert metrics["generated_cb0_frames"] == 6
    assert metrics["segment_rollovers"] == 1
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 6
    assert metrics["intermediate_segment_eos"] == [2]
    assert metrics["spoken_text_units"] == 20
    assert metrics["numeric_extra_spoken_units"] == 0
    assert metrics["requested_acoustic_frame_budget"] == 164
    assert metrics["acoustic_frame_budget"] == 164
    speaker = _BufferedRolloverSpeaker.instances[-1]
    assert speaker.segment_texts == [" ".join(text_words[:12]), " ".join(text_words[12:])]


def test_stateful_streaming_conditioning_survives_anchor_silence_and_rollover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _CountingLegacyTalker()
    engine = _engine_for_termination(talker)
    monkeypatch.setattr("duplex_kernel.engine.FuturePrefixQwen", _BufferedRolloverSpeaker)
    monkeypatch.setattr("duplex_kernel.engine.FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S", 0.01)
    buffer = ConditioningFrameBuffer(max_frames=256)
    buffer.begin_segment(generation_id=1, segment_id="response", base_frame_id=10)
    for frame_id in range(11, 151):
        assert buffer.append_values(
            frame_id=frame_id,
            h_pfx=torch.zeros((1, 2048)),
            produced_at_ns=frame_id,
            generation_id=1,
            segment_id="response",
        )

    text_words = [f"word{index}" for index in range(40)]
    text = " ".join(text_words)
    samples = iter([42] * 140 + [EOS_ID])
    engine._cb0_sample = lambda logits, temperature=0.6, **kwargs: next(samples)
    metrics: dict[str, object] = {}
    output = engine.talk_stateful(
        text,
        n_words=40,
        conditioning_buffer=buffer,
        max_frames=140,
        metrics=metrics,
    )

    assert output.size == 140 * 1920
    assert metrics["requested_acoustic_frame_budget"] == 304
    assert metrics["acoustic_frame_budget"] == 140
    assert metrics["generated_cb0_frames"] == 140
    assert metrics["segment_rollovers"] == 2
    assert metrics["terminal_queries"] == 1
    assert metrics["eos_detected"] is True
    speaker = _BufferedRolloverSpeaker.instances[-1]
    assert speaker.segment_pushes == [list(range(64)), list(range(64)), list(range(12))]
    assert [word for part in speaker.segment_texts for word in part.split()] == text_words


def test_stateful_eos_is_not_committed_or_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    talker = _CountingLegacyTalker()
    engine = _engine_for_termination(talker)
    monkeypatch.setattr("duplex_kernel.engine.FuturePrefixQwen", _EosStatefulSpeaker)

    def known_logits(self, *, allow_reduced_horizon: bool = False):
        del self, allow_reduced_horizon
        logits = torch.zeros((1, 1, VOCAB_SIZE))
        logits[0, 0, 42] = 4.0
        logits[0, 0, EOS_ID] = 2.0
        return SimpleNamespace(
            logits=logits,
            last_hidden=torch.zeros((1, 1, HIDDEN_SIZE)),
            frame_id=0,
        )

    monkeypatch.setattr(_EosStatefulSpeaker, "step_frame", known_logits)
    engine._cb0_sample = lambda logits, temperature=0.6, **kwargs: EOS_ID
    metrics: dict[str, object] = {}

    output = engine.talk_stateful(
        "ciao",
        torch.zeros((1, 1, 2048)),
        n_words=1,
        min_frames=1,
        metrics=metrics,
    )

    speaker = _EosStatefulSpeaker.instances[-1]
    assert output.size == 0
    assert speaker.commit_calls == 0
    assert engine.subtalker_graph.calls == 0
    assert talker._decode_calls == 0
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 0
    assert metrics["generated_cb0_frames"] == 0
    assert metrics["terminated_by_budget"] is False
    eos_diag = metrics["cb0_eos_diagnostics"]
    assert eos_diag["codec_eos_token_id"] == EOS_ID
    assert eos_diag["support_size"] == 2049
    log_z = np.log(2047.0 + np.exp(4.0) + np.exp(2.0))
    assert eos_diag["queries"] == [
        {
            "kind": "acoustic",
            "frame_id": 0,
            "eos_logit": 2.0,
            "eos_rank": 2,
            "eos_log_probability": pytest.approx(2.0 - log_z),
            "eos_probability": pytest.approx(np.exp(2.0 - log_z)),
            "argmax_id": 42,
        }
    ]


def test_multidigit_code_gets_spoken_unit_budget_and_reaches_native_eos() -> None:
    engine = _engine_for_termination(_CountingLegacyTalker())
    sampled = 0

    def sample(_logits, temperature=0.6, **_kwargs):
        nonlocal sampled
        del temperature
        sampled += 1
        return EOS_ID if sampled == 70 else 42

    engine._cb0_sample = sample
    metrics: dict[str, object] = {}
    output = engine.talk(
        "Perfetto, Marco. Ho registrato il tuo codice ordine: 4829.",
        torch.zeros((1, 1, HIDDEN_SIZE)),
        n_words=9,
        min_frames=48,
        metrics=metrics,
    )

    assert metrics["spoken_text_units"] == 12
    assert metrics["numeric_extra_spoken_units"] == 3
    assert metrics["requested_acoustic_frame_budget"] == 108
    assert metrics["generated_cb0_frames"] == 69
    assert metrics["eos_detected"] is True
    assert metrics["eos_frame_id"] == 69
    assert metrics["terminated_by_budget"] is False
    assert output.size == 69 * 1920


@pytest.mark.parametrize(
    ("text", "n_words", "expected_units", "expected_extra", "expected_frames"),
    [
        ("Ciao, Marco.", 2, 2, 0, 38),
        ("word19 word20", 20, 20, 0, 164),
        (" ".join(f"word{i}" for i in range(100)), 100, 100, 0, 512),
        ("codice 4829.", 2, 5, 3, 59),
        ("Ordine: (48-29), grazie.", 3, 6, 3, 66),
        ("Importo 12,50 euro.", 3, 6, 3, 66),
        ("date 12 34", 3, 5, 2, 59),
    ],
)
def test_spoken_unit_budget_handles_numeric_punctuation(
    text: str,
    n_words: int,
    expected_units: int,
    expected_extra: int,
    expected_frames: int,
) -> None:
    units, extra, frames = _estimate_talker_frame_budget(
        text,
        n_words,
        min_frames=1,
        short_streaming=False,
    )

    assert units == expected_units
    assert extra == expected_extra
    assert frames == expected_frames


def test_streaming_talker_cancel_after_first_pcm_chunk_is_bounded() -> None:
    engine = _engine_for_termination(_CountingLegacyTalker())
    engine._cb0_sample = lambda _logits, temperature=0.6, **_kwargs: 42
    chunks: list[np.ndarray] = []

    def receive_chunk(pcm: np.ndarray) -> None:
        chunks.append(pcm)
        engine.stop_generation()

    metrics: dict[str, object] = {}
    output = engine.talk(
        "Ciao.",
        torch.zeros((1, 1, HIDDEN_SIZE)),
        n_words=1,
        min_frames=1,
        on_audio_chunk=receive_chunk,
        metrics=metrics,
    )

    assert len(chunks) == 1
    assert chunks[0].size == 2 * 1920
    assert output.size == 0
    assert metrics["generated_cb0_frames"] == 2
    assert metrics["cancelled"] is True
    assert metrics["terminated_by_budget"] is False


def test_speak_frame_budget_preserves_full_text_and_hard_cap() -> None:
    _, _, ordinary = _estimate_talker_frame_budget("Quattro parole con un testo.", 4, 48, short_streaming=False)
    _, _, bounded = _estimate_talker_frame_budget("una risposta molto lunga", 100, 48, short_streaming=False)
    assert ordinary == 52  # 4 * 7 + 24
    assert bounded == 256
