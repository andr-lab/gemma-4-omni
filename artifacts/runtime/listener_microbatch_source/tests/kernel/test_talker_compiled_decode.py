from types import SimpleNamespace

import torch
import pytest

from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
import duplex_kernel.engine as engine_module
from duplex_kernel.engine import (
    PipelineEngine,
    _CompiledTalkerDecodeError,
    _QwenDynamicCacheQ1,
)


class _FakeCache:
    def __init__(self, length: int = 3) -> None:
        self.length = length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.length


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, *, inputs_embeds, past_key_values=None, use_cache):
        self.calls.append((inputs_embeds.shape, use_cache))
        assert use_cache is True
        if past_key_values is None:
            past_key_values = _FakeCache(0)
        past_key_values.length += int(inputs_embeds.shape[1])
        return SimpleNamespace(
            last_hidden_state=inputs_embeds + 1.0,
            past_key_values=past_key_values,
        )


def test_qwen_q1_wrapper_returns_last_hidden_and_same_mutated_cache() -> None:
    model = _FakeQwen()
    cache = _FakeCache()
    inputs = torch.tensor([[[1.0], [2.0]]])

    hidden, returned_cache = _QwenDynamicCacheQ1(model)(inputs, cache)

    assert hidden.shape == (1, 1, 1)
    assert hidden.item() == 3.0
    assert returned_cache is cache
    assert cache.length == 5
    assert model.calls == [((1, 2, 1), True)]


def test_prepare_compiled_q1_uses_dynamic_non_fullgraph_compile(monkeypatch) -> None:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine._talker_compiled_decode_requested = True
    engine.talker = SimpleNamespace(talker=SimpleNamespace(model=_FakeQwen()))
    engine.talker_q1_compile_status = "requested"
    engine.talker_q1_backend = "eager"
    engine.talker_q1_compile_fallback_reason = None
    engine._talker_q1_compile_warmup_calls = 0
    compile_calls = []

    def fake_compile(module, **kwargs):
        compile_calls.append((module, kwargs))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(engine, "_talker_q1_dynamo_counts", lambda: (0, 0))

    engine._prepare_compiled_talker_q1()

    assert engine.talker_q1_compile_status == "warming"
    assert engine._compiled_talker_q1 is compile_calls[0][0]
    assert compile_calls[0][1] == {
        "dynamic": True,
        "fullgraph": False,
        "mode": "default",
    }


def test_qwen_q1_forward_uses_compiled_path_only_for_v2() -> None:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.talker_q1_compile_status = "ready"
    engine._talker_q1_compile_warmup_calls = 0

    def compiled(inputs, cache):
        cache.length += int(inputs.shape[1])
        return inputs + 9.0, cache

    engine._compiled_talker_q1 = compiled
    model = _FakeQwen()
    inputs = torch.zeros((1, 1, 1))
    cache = _FakeCache()

    hidden, returned_cache = engine._qwen_q1_forward(model, inputs, cache)

    assert hidden.item() == 9.0
    assert returned_cache is cache
    assert model.calls == []

    engine.audio_response_protocol = None
    hidden, returned_cache = engine._qwen_q1_forward(model, inputs, cache)

    assert hidden.item() == 1.0
    assert returned_cache is cache
    assert len(model.calls) == 1


def test_compiled_warmup_uses_eager_prefill_and_three_changing_cache_positions(monkeypatch) -> None:
    model = _FakeQwen()
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.talker = SimpleNamespace(talker=SimpleNamespace(model=model))
    engine._compiled_talker_q1 = _QwenDynamicCacheQ1(model)
    engine._talker_q1_compile_warmup_calls = 0
    engine._talker_q1_dynamo_before = (0, 0)
    engine.talker_q1_backend = "eager"
    engine.talker_q1_compile_status = "warming"
    inputs_seen = []

    def fake_prefill(_talker, _text, _voice):
        return (
            torch.zeros((1, 3, 1)),
            torch.zeros((1, 2, 1)),
            torch.zeros((1, 1, 1)),
            torch.zeros((1, 1, 1)),
        )

    def fake_conditioning(_talker, codes, *, tts_pad_embed, generation_step):
        del tts_pad_embed
        inputs_seen.append((codes.shape, generation_step))
        return torch.full((1, 1, 1), float(generation_step))

    warmup_voice = {
        "mode": "x_vector_only",
        "speaker_embedding": torch.ones(2048),
    }
    monkeypatch.setattr(
        engine_module,
        "_resolve_talker_voice_conditioning",
        lambda _talker, conditioning: conditioning,
    )
    monkeypatch.setattr(engine_module, "_build_talker_prefill_parts", fake_prefill)
    monkeypatch.setattr(engine_module, "build_qwen_audio_conditioning", fake_conditioning)
    monkeypatch.setattr(engine, "_talker_q1_dynamo_counts", lambda: (1, 0))

    engine._warmup_compiled_talker_q1("Ciao.", torch.ones((1, 2, 1)), warmup_voice)

    assert model.calls[:2] == [((1, 3, 1), True), ((1, 5, 1), True)]
    assert inputs_seen == [((1, 1, 16), 0), ((1, 1, 16), 1), ((1, 1, 16), 2)]
    assert engine._talker_q1_compile_warmup_calls == 3
    assert engine.talker_q1_backend == "torch_compile_dynamic"
    assert engine.talker_q1_compile_status == "ready"


def test_compiled_runtime_failure_is_not_silently_retried_eager() -> None:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.talker_q1_compile_status = "ready"
    engine._talker_q1_compile_warmup_calls = 0
    engine._compiled_talker_q1 = lambda _inputs, _cache: (_ for _ in ()).throw(
        RuntimeError("compiled graph failed")
    )
    model = _FakeQwen()

    with pytest.raises(_CompiledTalkerDecodeError, match="compiled Qwen Q1 failed"):
        engine._qwen_q1_forward(model, torch.zeros((1, 1, 1)), _FakeCache())

    assert model.calls == []
