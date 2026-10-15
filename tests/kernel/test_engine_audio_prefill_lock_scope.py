from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.engine import PipelineEngine


class _FakeCache:
    def __init__(self, length: int = 0) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class _FakeTextModel:
    def __init__(self, lock: threading.Lock) -> None:
        self.lock = lock
        self.embed_tokens = torch.nn.Embedding(8, 2)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2))
        self.audio_feature_lock_states: list[bool] = []
        self.language_model_lock_states: list[bool] = []
        self.language_model_call_count = 0
        self.inputs_embeds_seen: torch.Tensor | None = None
        self.inputs_embeds_parts: list[torch.Tensor] = []
        self.attention_mask_seen: torch.Tensor | None = None
        self.position_ids_seen: torch.Tensor | None = None
        text_config = SimpleNamespace(
            pad_token_id=0,
            hidden_size_per_layer_input=0,
            use_bidirectional_attention=None,
            is_causal=True,
            layer_types=["full_attention"],
            num_hidden_layers=1,
            _attn_implementation="eager",
        )
        self.config = SimpleNamespace(
            image_token_id=98,
            video_token_id=97,
            audio_token_id=99,
            text_config=text_config,
            _attn_implementation="eager",
            get_text_config=lambda: text_config,
        )
        self.language_model = self

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def get_placeholder_mask(self, input_ids=None, inputs_embeds=None):
        assert input_ids is not None
        return input_ids == 98, input_ids == 97, input_ids == 99

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_audio_features(self, input_features, input_features_mask, *, return_dict):
        assert return_dict is True
        self.audio_feature_lock_states.append(self.lock.locked())
        return SimpleNamespace(
            pooler_output=torch.tensor([[[10.0, 11.0], [12.0, 13.0]]]),
            attention_mask=torch.tensor([[True, True]]),
        )

    def forward(
        self,
        *,
        per_layer_inputs,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        return_dict,
        output_hidden_states,
        logits_to_keep,
    ):
        assert per_layer_inputs is None
        assert use_cache is True and return_dict is True and output_hidden_states is True
        assert logits_to_keep == 1
        self.language_model_lock_states.append(self.lock.locked())
        self.language_model_call_count += 1
        self.inputs_embeds_seen = inputs_embeds.detach().clone()
        self.inputs_embeds_parts.append(inputs_embeds.detach().clone())
        self.attention_mask_seen = attention_mask["full_attention"].detach().clone()
        self.position_ids_seen = position_ids.detach().clone()
        last_hidden = inputs_embeds * 2
        cache = past_key_values if past_key_values is not None else _FakeCache()
        cache.length += int(inputs_embeds.shape[1])
        return SimpleNamespace(
            last_hidden_state=last_hidden,
            hidden_states=(last_hidden,),
            past_key_values=cache,
        )


class _FakeThinker:
    def __init__(self, model: _FakeTextModel) -> None:
        self.model = model
        self.config = model.config
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))

    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, position_ids):
        query_length = int(inputs_embeds.shape[1])
        query_mask = attention_mask[:, None, None, :].expand(
            attention_mask.shape[0], 1, query_length, attention_mask.shape[1]
        )
        return {"full_attention": query_mask, "positions": position_ids}

    def __call__(self, **kwargs):
        # The current path makes this call under the shared lock and the official
        # model runs both get_audio_features and its language model inside it.
        model = self.model
        image_mask, video_mask, audio_mask = model.get_placeholder_mask(input_ids=kwargs["input_ids"])
        multimodal_mask = image_mask | video_mask | audio_mask
        safe_ids = kwargs["input_ids"].clone()
        safe_ids[multimodal_mask] = model.config.text_config.pad_token_id
        embeds = model.get_input_embeddings()(safe_ids)
        feature_output = model.get_audio_features(
            kwargs["input_features"], kwargs["input_features_mask"], return_dict=True
        )
        features = feature_output.pooler_output[feature_output.attention_mask]
        embeds = embeds.masked_scatter(audio_mask.unsqueeze(-1).expand_as(embeds), features)
        positions = torch.arange(embeds.shape[1]).unsqueeze(0)
        lm_output = model.forward(
            per_layer_inputs=None,
            attention_mask={"full_attention": kwargs["attention_mask"], "positions": positions},
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=embeds,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        return SimpleNamespace(
            logits=self.lm_head(lm_output.last_hidden_state[:, -1:, :]),
            hidden_states=lm_output.hidden_states,
            past_key_values=lm_output.past_key_values,
        )


def test_audio_encoder_runs_outside_listener_lock_and_lm_keeps_original_scatter_contract() -> None:
    shared_lock = threading.Lock()
    model = _FakeTextModel(shared_lock)
    thinker = _FakeThinker(model)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = thinker
    engine._thinker_wrapper = SimpleNamespace(processor=None)
    engine._thinker_lock = shared_lock
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    runtime_metrics = {
        "thinker_prefill_lock_wait_ms": 0.0,
        "thinker_prefill_lock_hold_ms": 0.0,
        "thinker_prefill_lock_hold_max_ms": 0.0,
        "thinker_prefill_lock_hold_80ms_frames": 0.0,
    }
    input_ids = torch.tensor([[1, 99, 99, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    input_features = torch.ones((1, 2, 4))
    input_features_mask = torch.ones((1, 2), dtype=torch.bool)

    output, hidden_parts, cache, final_mask, _ = engine._run_thinker_audio_prefill(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        },
        is_compiled=False,
        static_cache=None,
        past_length=0,
        cache_init_started=None,
        runtime_metrics=runtime_metrics,
    )

    expected_ids = input_ids.clone()
    expected_ids[expected_ids == 99] = 0
    expected_embeds = model.get_input_embeddings()(expected_ids)
    expected_embeds = expected_embeds.masked_scatter(
        (input_ids == 99).unsqueeze(-1).expand_as(expected_embeds),
        torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
    )
    assert model.audio_feature_lock_states == [False]
    assert model.language_model_lock_states == [True]
    assert torch.equal(model.inputs_embeds_seen, expected_embeds)
    assert torch.equal(hidden_parts[0], expected_embeds * 2)
    assert torch.equal(output.logits, thinker.lm_head((expected_embeds * 2)[:, -1:, :]))
    assert output.past_key_values is cache
    assert torch.equal(final_mask, attention_mask)
    assert engine._thinker_audio_forward_active.is_set() is False


def test_opt_in_audio_prefill_chunks_only_the_language_model_after_full_audio_scatter() -> None:
    shared_lock = threading.Lock()
    model = _FakeTextModel(shared_lock)
    thinker = _FakeThinker(model)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = thinker
    engine._thinker_wrapper = SimpleNamespace(processor=None)
    engine._thinker_lock = shared_lock
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    engine.thinker_audio_prefill_chunk_tokens = 2
    runtime_metrics = {
        "thinker_prefill_lock_wait_ms": 0.0,
        "thinker_prefill_lock_hold_ms": 0.0,
        "thinker_prefill_lock_hold_max_ms": 0.0,
        "thinker_prefill_lock_80ms_frames": 0.0,
    }
    input_ids = torch.tensor([[1, 99, 99, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    output, hidden_parts, _, _, _ = engine._run_thinker_audio_prefill(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": torch.ones((1, 2, 4)),
            "input_features_mask": torch.ones((1, 2), dtype=torch.bool),
        },
        is_compiled=False,
        static_cache=None,
        past_length=0,
        cache_init_started=None,
        runtime_metrics=runtime_metrics,
    )

    assert model.audio_feature_lock_states == [False]
    assert model.language_model_call_count == 2
    assert model.language_model_lock_states == [True, True]
    assert output.final_hidden_state.shape[1] == 4
    assert hidden_parts[0].shape[1] == 4
    assert runtime_metrics["thinker_audio_lm_prefill_chunks"] == 2
    expected_ids = input_ids.clone()
    expected_ids[expected_ids == 99] = 0
    expected_embeds = model.get_input_embeddings()(expected_ids)
    expected_embeds = expected_embeds.masked_scatter(
        (input_ids == 99).unsqueeze(-1).expand_as(expected_embeds),
        torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
    )
    assert torch.equal(torch.cat(model.inputs_embeds_parts, dim=1), expected_embeds)


def test_tiny_gemma4_mixed_attention_audio_lm_chunks_match_cached_whole_prefill() -> None:
    from transformers.masking_utils import create_masks_for_generate
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(1234)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=4,
        num_global_key_value_heads=2,
        global_head_dim=8,
        use_cache=True,
        use_bidirectional_attention=None,
    )
    config._attn_implementation = "eager"
    language_model = Gemma4TextModel(config).eval()
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    all_embeds = language_model.embed_tokens(input_ids)
    all_per_layer = language_model.get_per_layer_inputs(input_ids, all_embeds)
    all_positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    all_mask = torch.ones_like(input_ids)
    full_masks = create_masks_for_generate(
        config, all_embeds, all_mask, None, all_positions
    )

    with torch.inference_mode():
        reference = language_model(
            inputs_embeds=all_embeds,
            per_layer_inputs=all_per_layer,
            attention_mask=full_masks,
            position_ids=all_positions,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    prefix_length = 3
    prefix_ids = input_ids[:, :prefix_length]
    prefix_embeds = all_embeds[:, :prefix_length]
    prefix_positions = all_positions[:, :prefix_length]
    prefix_mask = all_mask[:, :prefix_length]
    prefix_masks = create_masks_for_generate(
        config, prefix_embeds, prefix_mask, None, prefix_positions
    )
    with torch.inference_mode():
        prefix_output = language_model(
            inputs_embeds=prefix_embeds,
            per_layer_inputs=all_per_layer[:, :prefix_length],
            attention_mask=prefix_masks,
            position_ids=prefix_positions,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    checked_chunks: list[tuple[int, int]] = []

    def checked_mask_builder(mask_config, chunk_embeds, chunk_attention_mask, cache, chunk_positions):
        masks = create_masks_for_generate(
            mask_config, chunk_embeds, chunk_attention_mask, cache, chunk_positions
        )
        start = int(chunk_positions[0, 0].item())
        end = start + int(chunk_embeds.shape[1])
        for layer_type in ("full_attention", "sliding_attention"):
            actual = masks[layer_type]
            expected = full_masks[layer_type][..., start:end, :end]
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        checked_chunks.append((start, end))
        return masks

    thinker = SimpleNamespace(
        config=config,
        model=SimpleNamespace(language_model=language_model),
        create_masks_for_generate=checked_mask_builder,
    )
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = thinker
    engine._thinker_lock = threading.Lock()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    metrics = {"thinker_prefill_lock_wait_ms": 0.0, "thinker_prefill_lock_hold_ms": 0.0,
               "thinker_prefill_lock_hold_max_ms": 0.0}
    chunk_output = engine._run_chunked_thinker_audio_language_model(
        inputs_embeds=all_embeds[:, prefix_length:],
        per_layer_inputs=all_per_layer[:, prefix_length:],
        attention_mask=all_mask,
        position_ids=all_positions[:, prefix_length:],
        cache_position=None,
        cache=prefix_output.past_key_values,
        is_compiled=False,
        chunk_tokens=4,
        runtime_metrics=metrics,
        listener_admission_wait=None,
    )

    torch.testing.assert_close(
        chunk_output.last_hidden_state,
        reference.last_hidden_state[:, prefix_length:],
        rtol=1e-4,
        atol=1e-5,
    )
    with torch.inference_mode():
        lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).eval()
        chunk_logits = lm_head(chunk_output.last_hidden_state[:, -1:, :])
        reference_logits = lm_head(reference.last_hidden_state[:, -1:, :])
    torch.testing.assert_close(chunk_logits, reference_logits, rtol=1e-4, atol=1e-5)
    assert len(chunk_output.hidden_states) == len(reference.hidden_states)
    for actual, expected in zip(chunk_output.hidden_states, reference.hidden_states):
        torch.testing.assert_close(actual, expected[:, prefix_length:], rtol=1e-4, atol=1e-5)
    for layer_index in range(config.num_hidden_layers):
        actual_cache_layer = chunk_output.past_key_values.layers[layer_index]
        expected_cache_layer = reference.past_key_values.layers[layer_index]
        actual_key, actual_value = actual_cache_layer.keys, actual_cache_layer.values
        expected_key, expected_value = expected_cache_layer.keys, expected_cache_layer.values
        assert isinstance(actual_key, torch.Tensor) and isinstance(actual_value, torch.Tensor)
        assert isinstance(expected_key, torch.Tensor) and isinstance(expected_value, torch.Tensor)
        torch.testing.assert_close(actual_key, expected_key, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-4, atol=1e-5)
    assert checked_chunks == [(3, 7), (7, 11), (11, 13)]
    assert metrics["thinker_audio_lm_prefill_chunks"] == 3


def test_chunked_gemma4_prefill_fails_closed_on_noncontiguous_positions() -> None:
    text_config = SimpleNamespace(
        use_bidirectional_attention=None,
        is_causal=True,
        layer_types=["full_attention"],
        num_hidden_layers=1,
        _attn_implementation="eager",
    )
    config = SimpleNamespace(get_text_config=lambda: text_config)
    thinker = SimpleNamespace(
        config=config,
        model=SimpleNamespace(language_model=torch.nn.Identity()),
        create_masks_for_generate=lambda *args: {"full_attention": None},
    )
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = thinker
    engine._thinker_lock = threading.Lock()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    text_config._attn_implementation = "flash_attention_2"
    with pytest.raises(RuntimeError, match="active attention backend"):
        engine._run_chunked_thinker_audio_language_model(
            inputs_embeds=torch.zeros((1, 2, 8)),
            per_layer_inputs=None,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            position_ids=torch.tensor([[0, 1]]),
            cache_position=None,
            cache=None,
            is_compiled=False,
            chunk_tokens=2,
            runtime_metrics={"thinker_prefill_lock_wait_ms": 0.0, "thinker_prefill_lock_hold_ms": 0.0,
                             "thinker_prefill_lock_hold_max_ms": 0.0},
            listener_admission_wait=None,
        )
    text_config._attn_implementation = "eager"
    with pytest.raises(RuntimeError, match="contiguous absolute Gemma4 position IDs"):
        engine._run_chunked_thinker_audio_language_model(
            inputs_embeds=torch.zeros((1, 3, 8)),
            per_layer_inputs=None,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            position_ids=torch.tensor([[0, 2, 3]]),
            cache_position=None,
            cache=None,
            is_compiled=False,
            chunk_tokens=2,
            runtime_metrics={"thinker_prefill_lock_wait_ms": 0.0, "thinker_prefill_lock_hold_ms": 0.0,
                             "thinker_prefill_lock_hold_max_ms": 0.0},
            listener_admission_wait=None,
        )
    thinker.create_masks_for_generate = lambda *args: {
        "full_attention": torch.zeros((1, 1, 1, 1))
    }
    with pytest.raises(RuntimeError, match="mask query rows do not match"):
        engine._run_chunked_thinker_audio_language_model(
            inputs_embeds=torch.zeros((1, 2, 8)),
            per_layer_inputs=None,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            position_ids=torch.tensor([[0, 1]]),
            cache_position=None,
            cache=None,
            is_compiled=False,
            chunk_tokens=2,
            runtime_metrics={"thinker_prefill_lock_wait_ms": 0.0, "thinker_prefill_lock_hold_ms": 0.0,
                             "thinker_prefill_lock_hold_max_ms": 0.0},
            listener_admission_wait=None,
        )


class _ObservedLock:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.is_locked = False

    def acquire_fair(self, _kind: str, *, stop_check=None) -> bool:
        assert not self.is_locked
        self.is_locked = True
        self.events.append("lock_acquire")
        return True

    def release(self) -> None:
        assert self.is_locked
        self.events.append("lock_release")
        self.is_locked = False


class _ObservedStream:
    def __init__(self, events: list[str], error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    def synchronize(self) -> None:
        self.events.append("stream_sync")
        if self.error is not None:
            raise self.error


class _ObservedCudaEvent:
    def __init__(self, **_kwargs) -> None:
        pass

    def record(self, _stream) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, _other) -> float:
        return 0.0


def _minimal_chunk_engine(lock, language_model) -> PipelineEngine:
    text_config = SimpleNamespace(
        use_bidirectional_attention=None,
        is_causal=True,
        layer_types=["full_attention"],
        num_hidden_layers=1,
        _attn_implementation="eager",
    )
    config = SimpleNamespace(get_text_config=lambda: text_config)
    thinker = SimpleNamespace(
        config=config,
        model=SimpleNamespace(language_model=language_model),
        create_masks_for_generate=lambda _config, embeds, _mask, _cache, _positions: {
            "full_attention": torch.ones(
                (1, 1, int(embeds.shape[1]), 1), dtype=torch.bool
            )
        },
    )
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cuda:0"
    engine.thinker = thinker
    engine._thinker_lock = lock
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    return engine


def _chunked_call(engine: PipelineEngine, metrics: dict[str, float]) -> None:
    engine._run_chunked_thinker_audio_language_model(
        inputs_embeds=torch.zeros((1, 1, 2)),
        per_layer_inputs=None,
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        position_ids=torch.tensor([[0]]),
        cache_position=None,
        cache=None,
        is_compiled=False,
        chunk_tokens=1,
        runtime_metrics=metrics,
        listener_admission_wait=None,
    )


def _lock_metrics() -> dict[str, float]:
    return {
        "thinker_prefill_lock_wait_ms": 0.0,
        "thinker_prefill_lock_hold_ms": 0.0,
        "thinker_prefill_lock_hold_max_ms": 0.0,
        "thinker_prefill_lock_80ms_frames": 0.0,
    }


def test_audio_prefill_chunk_releases_lock_and_drains_stream_if_event_setup_fails(monkeypatch) -> None:
    events: list[str] = []
    stream = _ObservedStream(events)
    lock = _ObservedLock(events)
    engine = _minimal_chunk_engine(lock, lambda **_kwargs: pytest.fail("forward must not run"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)

    def fail_event_setup(**_kwargs):
        raise RuntimeError("event setup failed")

    monkeypatch.setattr(torch.cuda, "Event", fail_event_setup)
    with pytest.raises(RuntimeError, match="event setup failed"):
        _chunked_call(engine, _lock_metrics())

    assert events == ["lock_acquire", "stream_sync", "lock_release"]
    assert not lock.is_locked


def test_audio_prefill_chunk_syncs_before_release_and_preserves_forward_error(monkeypatch) -> None:
    events: list[str] = []
    sync_error = RuntimeError("stream drain failed")
    stream = _ObservedStream(events, sync_error)
    lock = _ObservedLock(events)
    forward_error = RuntimeError("forward failed")

    def fail_after_queue(**_kwargs):
        events.append("forward")
        raise forward_error

    engine = _minimal_chunk_engine(lock, fail_after_queue)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(torch.cuda, "Event", _ObservedCudaEvent)

    with pytest.raises(RuntimeError, match="forward failed") as exc_info:
        _chunked_call(engine, _lock_metrics())

    assert exc_info.value is forward_error
    assert events == ["lock_acquire", "forward", "stream_sync", "lock_release"]
    assert not lock.is_locked


def test_audio_prefill_chunk_propagates_sync_only_failure_after_releasing_lock(monkeypatch) -> None:
    events: list[str] = []
    sync_error = RuntimeError("stream drain failed")
    stream = _ObservedStream(events, sync_error)
    lock = _ObservedLock(events)

    def succeed(**kwargs):
        return SimpleNamespace(
            last_hidden_state=kwargs["inputs_embeds"],
            hidden_states=(kwargs["inputs_embeds"],),
            past_key_values=SimpleNamespace(length=1),
        )

    engine = _minimal_chunk_engine(lock, succeed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(torch.cuda, "Event", _ObservedCudaEvent)

    with pytest.raises(RuntimeError, match="stream drain failed") as exc_info:
        _chunked_call(engine, _lock_metrics())

    assert exc_info.value is sync_error
    assert events == ["lock_acquire", "stream_sync", "lock_release"]
    assert not lock.is_locked
