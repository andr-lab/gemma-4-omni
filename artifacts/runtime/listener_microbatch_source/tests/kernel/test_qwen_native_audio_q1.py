from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.native_qwen_talker import (
    _advance_audio_q1_position,
    _dynamic_cache_kv_layers,
)
from duplex_kernel.frame_residual_adapter import FrameResidualQueryFiLMAdapter
from duplex_kernel.engine import PipelineEngine
import duplex_kernel.engine as engine_module


ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _cache(sequence_length: int = 5) -> tuple[object, list[tuple[torch.Tensor, torch.Tensor]]]:
    pairs = [
        (
            torch.randn((1, 8, sequence_length, 128), dtype=torch.bfloat16),
            torch.randn((1, 8, sequence_length, 128), dtype=torch.bfloat16),
        )
        for _ in range(28)
    ]
    layers = [SimpleNamespace(keys=key, values=value) for key, value in pairs]
    cache = SimpleNamespace(layers=layers, get_seq_length=lambda: sequence_length)
    return cache, pairs


def test_dynamic_cache_import_view_keeps_exact_bf16_layer_storage() -> None:
    cache, original = _cache()

    sequence_length, layers = _dynamic_cache_kv_layers(cache)

    assert sequence_length == 5
    assert len(layers) == 28
    assert layers[0][0] is original[0][0]
    assert layers[0][1] is original[0][1]
    assert layers[-1][0] is original[-1][0]
    assert all(key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16 for key, value in layers)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda cache: cache.layers.pop(), "28"),
        (lambda cache: setattr(cache.layers[0], "keys", torch.empty((1, 8, 4, 128), dtype=torch.bfloat16)), "shape"),
        (lambda cache: setattr(cache.layers[0], "values", torch.empty((1, 8, 5, 128), dtype=torch.float16)), "BF16"),
        (lambda cache: setattr(cache.layers[0], "keys", torch.empty((1, 8, 5, 256), dtype=torch.bfloat16)), "shape"),
        (lambda cache: setattr(cache.layers[0], "keys", torch.empty((1, 8, 5, 256), dtype=torch.bfloat16)[:, :, :, ::2]), "contiguo"),
        (lambda cache: setattr(cache, "get_seq_length", lambda: 4), "shape"),
    ],
)
def test_dynamic_cache_import_view_fails_closed(mutate, message: str) -> None:
    cache, _ = _cache()
    mutate(cache)

    with pytest.raises(ValueError, match=message):
        _dynamic_cache_kv_layers(cache)


def test_native_abi_has_a_distinct_recurrent_audio_q1_import_and_step() -> None:
    header = (ROOT / "native/qwen_talker/qwen_talker.h").read_text(encoding="utf-8")
    implementation = (ROOT / "native/qwen_talker/qwen_talker.cu").read_text(encoding="utf-8")
    adapter = (ROOT / "duplex_kernel/native_qwen_talker.py").read_text(encoding="utf-8")

    assert "qwen_talker_import_audio_cache_async" in header
    assert "qwen_talker_import_audio_cache_async" in implementation
    assert "qwen_talker_step_audio_async" in header
    assert "qwen_talker_step_audio_async" in implementation
    assert "def import_dynamic_cache_async(" in adapter
    assert "def step_audio_async(" in adapter
    # The old terminal API remains distinct and retains its one-shot guard.
    assert "qwen_talker_step_terminal_async" in implementation
    assert "ctx->terminal_query_done" in implementation


def test_native_joint_prefix_transitions_to_audio_q1_without_cache_reimport() -> None:
    implementation = _source("native/qwen_talker/qwen_talker.cu")
    adapter = _source("duplex_kernel/native_qwen_talker.py")
    future_prefix = _source("duplex_kernel/future_prefix.py")

    audio_step = implementation.split("extern \"C\" int qwen_talker_step_audio_async", 1)[1].split(
        "extern \"C\" void qwen_talker_destroy", 1
    )[0]
    terminal_step = implementation.split("extern \"C\" int qwen_talker_step_terminal_async", 1)[1].split(
        "extern \"C\" int qwen_talker_step_audio_async", 1
    )[0]

    assert "ctx->next_prefix_position != ctx->audio_start" in audio_step
    assert "ctx->mode = qwen_execution_mode::recurrent_audio_q1" in audio_step
    assert "visible_prefix_position = ctx->audio_start - 1u" in audio_step
    assert "def step_audio_async(" in adapter
    assert "self._native_qwen.step_audio_async(" in future_prefix
    assert "joint_q2_mode" in terminal_step
    assert "ctx->mode == qwen_execution_mode::recurrent_audio_q1" in terminal_step


def test_native_q1_suffix_starts_only_after_real_prefix_and_keeps_absolute_audio_positions(monkeypatch) -> None:
    import ctypes
    import duplex_kernel.native_qwen_talker as native_module
    from duplex_kernel.native_qwen_talker import NativeQwenTalker

    calls = []

    class _Lib:
        def qwen_talker_step_audio_async(self, *args):
            calls.append(args)
            return 0

    executor = NativeQwenTalker.__new__(NativeQwenTalker)
    executor._closed = False
    executor._ctx = ctypes.c_void_p(1)
    executor.device = torch.device("cpu")
    executor.max_cache_len = 80
    executor.last_hidden = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
    executor._lib = _Lib()
    executor._audio_q1_imported = False
    executor._audio_q1_next_position = None
    executor._audio_q1_limit_position = None
    executor._joint_prefix_start = 11
    executor._joint_audio_start = 14
    executor._joint_next_prefix = 13
    executor._joint_next_audio = 17

    monkeypatch.setattr(
        NativeQwenTalker,
        "_embedding_ptr",
        staticmethod(lambda _tensor, _label: 123),
    )
    monkeypatch.setattr(native_module, "_stream_handle", lambda _device: 0)
    audio = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="prefix reali contigui"):
        executor.step_audio_async(audio, max_audio_frames=32)
    assert calls == []

    executor._joint_next_prefix = 14
    result = executor.step_audio_async(audio, max_audio_frames=32)

    assert result is executor.last_hidden
    assert len(calls) == 1
    assert executor._audio_q1_next_position == 18
    assert executor._audio_q1_limit_position == 46


def test_q1_suffix_mask_exposes_real_prefix_and_causal_audio_positions_only() -> None:
    from duplex_kernel.future_prefix import build_causal_prefix_mask

    prefix_start = 4
    real_prefix_frames = 3
    audio_start = prefix_start + real_prefix_frames
    audio_budget = 8
    total_length = audio_start + audio_budget + 1
    mask = build_causal_prefix_mask(
        batch_size=1,
        prefix_start=prefix_start,
        prefix_frames=real_prefix_frames,
        audio_start=audio_start,
        audio_frames=audio_budget + 1,
        available_prefix_end=real_prefix_frames - 1,
        lookahead_frames=0,
        dtype=torch.float32,
        device="cpu",
        total_length=total_length,
    )

    suffix_frame_index = real_prefix_frames
    query_position = audio_start + suffix_frame_index
    assert audio_start == prefix_start + real_prefix_frames
    assert torch.all(mask[0, 0, query_position, prefix_start:audio_start] == 0)
    assert torch.all(mask[0, 0, query_position, audio_start : query_position + 1] == 0)
    assert mask[0, 0, query_position, query_position + 1] == torch.finfo(torch.float32).min


def test_native_q1_suffix_starts_only_after_real_prefix_and_keeps_absolute_audio_positions(monkeypatch) -> None:
    import ctypes
    import duplex_kernel.native_qwen_talker as native_module
    from duplex_kernel.native_qwen_talker import NativeQwenTalker

    calls = []

    class _Lib:
        def qwen_talker_step_audio_async(self, *args):
            calls.append(args)
            return 0

    executor = NativeQwenTalker.__new__(NativeQwenTalker)
    executor._closed = False
    executor._ctx = ctypes.c_void_p(1)
    executor.device = torch.device("cpu")
    executor.max_cache_len = 80
    executor.last_hidden = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
    executor._lib = _Lib()
    executor._audio_q1_imported = False
    executor._audio_q1_next_position = None
    executor._audio_q1_limit_position = None
    executor._joint_prefix_start = 11
    executor._joint_audio_start = 14
    executor._joint_next_prefix = 13
    executor._joint_next_audio = 17

    monkeypatch.setattr(
        NativeQwenTalker,
        "_embedding_ptr",
        staticmethod(lambda _tensor, _label: 123),
    )
    monkeypatch.setattr(native_module, "_stream_handle", lambda _device: 0)
    monkeypatch.setattr(
        native_module,
        "_meta",
        lambda _tensor: native_module.QwenTensorMeta(),
    )
    audio = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="prefix reali contigui"):
        executor.step_audio_async(audio, max_audio_frames=32)
    assert calls == []

    executor._joint_next_prefix = 14
    result = executor.step_audio_async(audio, max_audio_frames=32)

    assert result is executor.last_hidden
    assert len(calls) == 1
    assert executor._audio_q1_next_position == 18
    assert executor._audio_q1_limit_position == 46


def test_q1_suffix_mask_exposes_real_prefix_and_causal_audio_positions_only() -> None:
    from duplex_kernel.future_prefix import build_causal_prefix_mask

    prefix_start = 4
    real_prefix_frames = 3
    audio_start = prefix_start + real_prefix_frames
    audio_budget = 8
    total_length = audio_start + audio_budget + 1
    mask = build_causal_prefix_mask(
        batch_size=1,
        prefix_start=prefix_start,
        prefix_frames=real_prefix_frames,
        audio_start=audio_start,
        audio_frames=audio_budget + 1,
        available_prefix_end=real_prefix_frames - 1,
        lookahead_frames=0,
        dtype=torch.float32,
        device="cpu",
        total_length=total_length,
    )

    suffix_frame_index = real_prefix_frames
    query_position = audio_start + suffix_frame_index
    assert audio_start == prefix_start + real_prefix_frames
    assert torch.all(mask[0, 0, query_position, prefix_start:audio_start] == 0)
    assert torch.all(mask[0, 0, query_position, audio_start : query_position + 1] == 0)
    assert mask[0, 0, query_position, query_position + 1] == torch.finfo(torch.float32).min


def test_audio_q1_position_counter_advances_for_forty_frames_then_fails_closed() -> None:
    start = 127
    limit = start + 40

    position = start
    for offset in range(1, 41):
        position = _advance_audio_q1_position(position, limit)
        assert position == start + offset

    with pytest.raises(RuntimeError, match="limite frame audio"):
        _advance_audio_q1_position(position, limit)


def test_recurrent_q1_import_reserves_exact_prefill_and_steps_only_audio_positions() -> None:
    implementation = _source("native/qwen_talker/qwen_talker.cu")
    adapter = _source("duplex_kernel/native_qwen_talker.py")

    imported = implementation.split("extern \"C\" int qwen_talker_import_audio_cache_async", 1)[1].split(
        "extern \"C\" int qwen_talker_debug_execute_layer_async", 1
    )[0]
    step = implementation.split("extern \"C\" int qwen_talker_step_audio_async", 1)[1].split(
        "extern \"C\" void qwen_talker_destroy", 1
    )[0]
    assert "source.k_meta->dims[2] != prefill_length" in imported
    assert "source.v_meta->dims[2] != prefill_length" in imported
    assert "required_capacity > ctx->config.max_cache_len" in imported
    assert "ctx->prefix_start = 0u" in imported
    assert "ctx->audio_start = prefill_length" in imported
    assert "ctx->mode = qwen_execution_mode::recurrent_audio_q1" in imported
    assert "audio_position = ctx->next_audio_position" in step
    assert "ctx->next_audio_position >= ctx->audio_q1_limit_position" in step
    assert "ctx->next_audio_position = audio_position + 1u" in step
    assert "qwen_build_audio_only_mask" in implementation
    assert "sequence_length + max_audio_frames > self.max_cache_len" in adapter
    assert "next_position = _advance_audio_q1_position(" in adapter
    assert "ctx->terminal_query_done" in step


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self):
        return self.length


class _FakeQwen(torch.nn.Module):
    def forward(self, *, inputs_embeds, past_key_values=None, use_cache):
        assert use_cache is True
        if past_key_values is None:
            past_key_values = _FakeCache(0)
        past_key_values.length += int(inputs_embeds.shape[1])
        return SimpleNamespace(
            last_hidden_state=inputs_embeds + 1,
            past_key_values=past_key_values,
        )


class _FakeNativeQ1:
    instances = []

    def __init__(self, talker, *, max_cache_len):
        self.talker = talker
        self.max_cache_len = max_cache_len
        self.imported = None
        self.steps = []
        self.closed = False
        self.instances.append(self)

    def import_dynamic_cache_async(self, cache, *, max_audio_frames):
        self.imported = (cache, max_audio_frames)

    def step_audio_async(self, inputs):
        self.steps.append(inputs)
        return inputs + 3

    def close(self):
        self.closed = True


def test_native_q1_executor_imports_exact_hf_prefill_cache_and_capacity(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_NATIVE_LIBRARY_PATH", "/unused/test-library.so")
    model = _FakeQwen()
    talker = SimpleNamespace(model=model)
    cache = _FakeCache(19)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine._talker_native_audio_q1_graph_requested = False
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2

    monkeypatch.setattr("duplex_kernel.native_qwen_talker.NativeQwenTalker", _FakeNativeQ1)
    executor = engine._new_native_talker_q1(talker, cache, max_audio_frames=41)

    assert executor.max_cache_len == 60
    assert executor.imported == (cache, 41)


def test_live_native_q1_import_reserves_bounded_capacity_for_growing_text_budget() -> None:
    live_stream = object()

    assert engine_module._native_q1_import_audio_frame_budget(
        57,
        live_qwen_text_stream=live_stream,
        live_frame_cap=512,
    ) == 512
    assert engine_module._native_q1_import_audio_frame_budget(
        57,
        live_qwen_text_stream=None,
        live_frame_cap=512,
    ) == 57


def test_live_frame_residual_preserves_only_global_frame_zero_and_uses_exact_packet_hidden() -> None:
    adapter = FrameResidualQueryFiLMAdapter()
    with torch.no_grad():
        adapter.output.weight.normal_(mean=0.0, std=0.02)
        adapter.output.bias.fill_(0.01)
    torch.manual_seed(123)
    query = torch.randn((1, 1, 2048))
    hidden_first = torch.randn((1, 1, 1536))
    hidden_last = torch.randn((1, 1, 1536))
    seen_hidden = []
    adapter.register_forward_pre_hook(
        lambda _module, args: seen_hidden.append(args[0].detach().clone())
    )

    frame0, tail0 = engine_module._apply_live_frame_residual(
        adapter, hidden_first, query, generation_step=0, tail_hold=False
    )
    frame1, tail1 = engine_module._apply_live_frame_residual(
        adapter, hidden_first, query, generation_step=1, tail_hold=False
    )
    frame2, tail2 = engine_module._apply_live_frame_residual(
        adapter, hidden_last, query, generation_step=2, tail_hold=True
    )

    assert torch.equal(frame0, query)
    assert torch.linalg.vector_norm(frame1 - query).item() > 0.0
    assert not torch.allclose(frame1, frame2)
    assert torch.equal(seen_hidden[0], hidden_first)
    assert torch.equal(seen_hidden[1], hidden_last)
    assert (tail0, tail1, tail2) == (False, False, True)


def test_native_q1_startup_warmup_uses_three_audio_queries_after_nonzero_prefix(monkeypatch) -> None:
    _FakeNativeQ1.instances.clear()
    model = _FakeQwen()
    talker = SimpleNamespace(model=model)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine._talker_native_audio_q1_graph_requested = False
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.talker = SimpleNamespace(talker=talker)
    engine.talker_q1_native_warmup_calls = 0
    inputs_seen = []

    def fake_prefill(_talker, _text, _voice):
        return (
            torch.zeros((1, 2, 2048)),
            torch.zeros((1, 2, 2048)),
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
        )

    def fake_conditioning(_talker, codes, *, tts_pad_embed, generation_step):
        del tts_pad_embed
        inputs_seen.append((codes.shape, generation_step))
        return torch.full((1, 1, 2048), float(generation_step + 1))

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
    monkeypatch.setattr(engine, "_new_native_talker_q1", lambda _t, cache, *, max_audio_frames: _FakeNativeQ1(_t, max_cache_len=cache.length + max_audio_frames))

    engine._warmup_native_talker_q1("Ciao.", torch.ones((1, 16, 2048)), warmup_voice)

    # One conditioning query builds the warmup embedding before the three Q1 steps.
    assert len(inputs_seen) == 4
    assert inputs_seen == [
        ((1, 1, 16), 0),  # warmup embedding
        ((1, 1, 16), 0),
        ((1, 1, 16), 1),
        ((1, 1, 16), 2),
    ]
    assert engine.talker_q1_native_warmup_calls == 3
    assert _FakeNativeQ1.instances[-1].closed is True


def test_native_q1_step_is_selected_by_v2_and_counts_calls() -> None:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine._talker_native_audio_q1_graph_requested = False
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.talker_q1_native_warmup_calls = 3
    executor = _FakeNativeQ1(None, max_cache_len=10)
    inputs = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
    metrics = {}

    hidden = engine._native_talker_q1_step(executor, inputs, metrics=metrics)

    assert torch.equal(hidden, inputs + 3)
    assert metrics["talker_q1_backend"] == "native_audio_q1"
    assert metrics["native_talker_q1_calls"] == 1


def test_native_q1_step_uses_async_executor_when_graph_was_not_requested() -> None:
    class _GraphCapableButUncapturedExecutor:
        def __init__(self):
            self.graph_calls = 0
            self.async_calls = 0

        def step_audio_graph_async(self, _inputs):
            self.graph_calls += 1
            raise RuntimeError("graph Q1 non catturato per questo slot")

        def step_audio_async(self, inputs):
            self.async_calls += 1
            return inputs + 5

    engine = PipelineEngine.__new__(PipelineEngine)
    engine._talker_native_audio_q1_graph_requested = False
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    executor = _GraphCapableButUncapturedExecutor()
    inputs = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)

    hidden = engine._native_talker_q1_step(executor, inputs)

    assert torch.equal(hidden, inputs + 5)
    assert executor.graph_calls == 0
    assert executor.async_calls == 1


def test_native_q1_step_uses_graph_only_when_requested_and_fails_closed_without_it() -> None:
    class _GraphExecutor:
        def __init__(self):
            self.graph_calls = 0
            self.async_calls = 0

        def step_audio_graph_async(self, inputs):
            self.graph_calls += 1
            return inputs + 7

        def step_audio_async(self, inputs):
            self.async_calls += 1
            return inputs + 5

    inputs = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine._talker_native_audio_q1_graph_requested = True
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    executor = _GraphExecutor()
    metrics = {}

    hidden = engine._native_talker_q1_step(executor, inputs, metrics=metrics)

    assert torch.equal(hidden, inputs + 7)
    assert executor.graph_calls == 1
    assert executor.async_calls == 0
    assert metrics["native_talker_q1_graph_calls"] == 1

    missing_graph = SimpleNamespace(step_audio_async=lambda value: value)
    with pytest.raises(engine_module._NativeTalkerDecodeError, match="missing graph-step method"):
        engine._native_talker_q1_step(missing_graph, inputs)


def test_native_q1_opt_in_takes_precedence_over_compiled_decode(monkeypatch) -> None:
    monkeypatch.setenv("DUPLEX_TALKER_NATIVE_AUDIO_Q1", "1")
    monkeypatch.setenv("DUPLEX_TALKER_COMPILED_DECODE", "1")
    monkeypatch.delenv("QWEN_NATIVE_LIBRARY_PATH", raising=False)

    engine = PipelineEngine(device="cpu")

    assert engine._talker_native_audio_q1_requested is True
    assert engine._talker_compiled_decode_requested is True
    assert engine.talker_q1_compile_status == "skipped_native_precedence"
    assert engine.talker_q1_native_status == "requested"
    assert engine_module.os.environ["QWEN_NATIVE_LIBRARY_PATH"].endswith(
        "/.tmp/native_qwen_audio_q1/libqwen_talker_audio_q1.so"
    )
