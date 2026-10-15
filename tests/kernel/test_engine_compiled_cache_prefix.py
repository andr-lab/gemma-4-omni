from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from duplex_kernel.engine import PipelineEngine
from duplex_kernel.thinker_cache import static_cache_sequence_length


class _TinyConfig(PretrainedConfig):
    def __init__(self) -> None:
        super().__init__()
        self.num_hidden_layers = 1
        self.layer_types = ["sliding_attention"]
        self.sliding_window = 8
        self.attention_chunk_size = None
        self.audio_token_id = 99

    def get_text_config(self, decoder: bool = True):
        assert decoder
        return self


class _TinyThinker:
    def __init__(self) -> None:
        self.config = _TinyConfig()
        self.model = self
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.seen_caches = []
        self.calls = []

    def get_placeholder_mask(self, *, input_ids):
        empty = torch.zeros_like(input_ids, dtype=torch.bool)
        return empty, empty, input_ids == self.config.audio_token_id

    def __call__(
        self,
        *,
        input_ids,
        past_key_values,
        input_features=None,
        input_features_mask=None,
        **_kwargs,
    ):
        cache = past_key_values
        past_length = int(cache.get_seq_length())
        self.seen_caches.append(cache)
        token_values = input_ids.to(torch.float32).reshape(1, 1, -1, 1)
        if input_features is not None:
            audio_mask = self.get_placeholder_mask(input_ids=input_ids)[2]
            audio_values = input_features.to(torch.float32).mean(dim=-1).reshape(-1)
            if int(audio_mask.sum()) != audio_values.numel():
                raise ValueError("fake audio feature and placeholder counts differ")
            token_values = token_values.clone()
            token_values[0, 0, audio_mask[0], 0] = audio_values
        keys = token_values.expand(1, 2, -1, 4).contiguous()
        values = keys + 100.0
        cache.update(keys, values, 0)
        cache_length = int(cache.get_seq_length())
        context = cache.layers[0].keys[:, :, :cache_length, :].sum()
        self.calls.append({
            "input_ids": input_ids.clone(),
            "input_features": None if input_features is None else input_features.clone(),
            "input_features_mask": None if input_features_mask is None else input_features_mask.clone(),
            "past_length": past_length,
            "context": context.detach().clone(),
        })
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2) + context
        logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
        logits[..., 2] = 1.0 + context
        return SimpleNamespace(
            hidden_states=(hidden,), logits=logits, past_key_values=cache,
        )


class _Tokenizer:
    eos_token_id = 3
    eot_id = None

    @staticmethod
    def decode(_tokens, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "ok"


def _engine() -> PipelineEngine:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.audio_text_prefix_cache_reuse_enabled = True
    engine.thinker = _TinyThinker()
    engine._compiled_thinker = lambda **kwargs: engine.thinker(**kwargs)
    engine.thinker_static_cache_max_len = 16
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = _Tokenizer()
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    return engine


def test_audio_text_prefix_cache_reuse_is_opt_in_and_rejects_ambiguous_env(monkeypatch) -> None:
    monkeypatch.delenv("DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE", raising=False)
    default = PipelineEngine(device="cpu", thinker_compile_mode="off")
    assert default.audio_text_prefix_cache_reuse_enabled is False

    monkeypatch.setenv("DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE", "1")
    opted_in = PipelineEngine(device="cpu", thinker_compile_mode="off")
    assert opted_in.audio_text_prefix_cache_reuse_enabled is True

    monkeypatch.setenv("DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE", "true")
    with pytest.raises(ValueError, match="must be exactly '0' or '1'"):
        PipelineEngine(device="cpu", thinker_compile_mode="off")


def _state():
    return SimpleNamespace(
        thinker_static_cache=None,
        thinker_static_cache_ids=[],
        thinker_static_cache_fingerprint=None,
        stale_kv_writes=0,
        _generation_lock=threading.RLock(),
    )


def test_compiled_thinker_reuses_only_exact_prompt_prefix_and_keeps_decode_private() -> None:
    engine = _engine()
    state = _state()

    first = engine.think([10, 11], max_new_tokens=2, temperature=0.0, state=state)
    committed = state.thinker_static_cache
    committed_keys = committed.layers[0].keys[:, :, :2].clone()
    assert static_cache_sequence_length(committed) == 2
    assert first["runtime_metrics"]["thinker_cache_publish"] == "committed"

    second = engine.think([10, 11, 12], max_new_tokens=1, temperature=0.0, state=state)
    assert second["runtime_metrics"]["thinker_cache_hit"] is True
    assert second["runtime_metrics"]["thinker_cache_reused_tokens"] == 2
    assert second["runtime_metrics"]["thinker_prefill_tokens"] == 1
    assert static_cache_sequence_length(committed) == 2
    assert torch.equal(committed.layers[0].keys[:, :, :2], committed_keys)
    assert state.thinker_static_cache_ids == [10, 11, 12]

    changed = engine.think([9, 11, 99], max_new_tokens=1, temperature=0.0, state=state)
    assert changed["runtime_metrics"]["thinker_cache_hit"] is False
    assert changed["runtime_metrics"]["thinker_cache_reused_tokens"] == 0
    assert changed["runtime_metrics"]["thinker_prefill_tokens"] == 3
    assert changed["runtime_metrics"]["thinker_cache_reason"] == "no_common_prefix"


def test_compiled_thinker_publishes_prompt_context_on_current_generation_cancel_only() -> None:
    engine = _engine()
    state = _state()

    cancelled = engine.think(
        [20, 21], max_new_tokens=2, temperature=0.0, state=state,
        generation_id=4, stop_check=lambda: True,
    )
    assert cancelled["cancelled"] is True
    assert state.thinker_static_cache_ids == [20, 21]
    assert static_cache_sequence_length(state.thinker_static_cache) == 2

    class StaleState:
        def __init__(self):
            self.thinker_static_cache = None
            self.thinker_static_cache_ids = []
            self.thinker_static_cache_fingerprint = None
            self.stale_kv_writes = 0
            self._generation_lock = threading.RLock()

        @staticmethod
        def is_generation_current(*, generation_id: int) -> bool:
            return generation_id == 0

    stale = StaleState()
    stale_result = engine.think(
        [30, 31], max_new_tokens=1, temperature=0.0, state=stale,
        generation_id=9, stop_check=lambda: True,
    )
    assert stale_result["cancelled"] is True
    assert stale.thinker_static_cache is None
    assert stale.thinker_static_cache_ids == []
    assert stale.stale_kv_writes == 1
    assert stale_result["runtime_metrics"]["thinker_cache_publish"] == "stale_generation"


def test_compiled_multimodal_prefill_reuses_only_exact_text_prefix_before_audio() -> None:
    engine = _engine()
    state = _state()
    first_audio = {
        "input_ids": torch.tensor([[10, 11, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "input_features": torch.tensor([[[2.0], [3.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }
    engine.think(
        [10, 11, 99, 99], max_new_tokens=1, temperature=0.0, state=state, input_audio=first_audio,
    )
    first_cache = state.thinker_static_cache

    second_audio = {
        "input_ids": torch.tensor([[10, 11, 12, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "input_features": torch.tensor([[[7.0], [8.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }
    second = engine.think(
        [10, 11, 12, 99, 99], max_new_tokens=1, temperature=0.0, state=state, input_audio=second_audio,
    )

    baseline = _engine().think(
        [10, 11, 12, 99, 99], max_new_tokens=1, temperature=0.0,
        state=_state(), input_audio=second_audio,
    )

    assert second["runtime_metrics"]["thinker_cache_hit"] is True
    assert second["runtime_metrics"]["thinker_cache_reused_tokens"] == 2
    assert second["runtime_metrics"]["thinker_prefill_tokens"] == 3
    assert second["runtime_metrics"]["thinker_cache_reason"] == "exact_text_prefix_before_audio"
    assert engine.thinker.calls[2]["input_ids"].tolist() == [[12, 99, 99]]
    assert engine.thinker.calls[2]["past_length"] == 2
    assert torch.equal(engine.thinker.calls[2]["input_features"], second_audio["input_features"])
    assert engine.thinker.seen_caches[2] is not first_cache
    branch_keys = engine.thinker.seen_caches[2].layers[0].keys
    assert branch_keys[0, 0, :2, 0].tolist() == [10.0, 11.0]
    assert branch_keys[0, 0, 3:5, 0].tolist() == [7.0, 8.0]
    assert torch.equal(
        second["hidden"][:, second["hidden_prompt_len"]:, :],
        baseline["hidden"][:, baseline["hidden_prompt_len"]:, :],
    )
    assert state.thinker_static_cache_ids == [10, 11, 12]
    assert static_cache_sequence_length(state.thinker_static_cache) == 3


def test_compiled_audio_cross_response_prefix_reuse_can_be_disabled_without_disabling_decode_kv() -> None:
    engine = _engine()
    state = _state()
    first_audio = {
        "input_ids": torch.tensor([[10, 11, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "input_features": torch.tensor([[[2.0], [3.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }
    engine.think(
        [10, 11, 99, 99], max_new_tokens=1, temperature=0.0, state=state,
        input_audio=first_audio,
    )
    committed = state.thinker_static_cache
    committed_ids = list(state.thinker_static_cache_ids)
    committed_keys = committed.layers[0].keys[:, :, :2].clone()
    engine.audio_text_prefix_cache_reuse_enabled = False

    second_audio = {
        "input_ids": torch.tensor([[10, 11, 12, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "input_features": torch.tensor([[[7.0], [8.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }
    second = engine.think(
        [10, 11, 12, 99, 99], max_new_tokens=1, temperature=0.0, state=state,
        input_audio=second_audio,
    )

    assert second["runtime_metrics"]["thinker_cache_hit"] is False
    assert second["runtime_metrics"]["thinker_cache_reused_tokens"] == 0
    assert second["runtime_metrics"]["thinker_cache_reason"] == "fresh:audio_cross_response_prefix_reuse_disabled"
    assert second["runtime_metrics"]["thinker_prefill_tokens"] == 5
    assert second["runtime_metrics"]["thinker_decode_steps"] == 1
    assert engine.thinker.calls[2]["input_ids"].tolist() == [[10, 11, 12, 99, 99]]
    assert engine.thinker.calls[2]["past_length"] == 0
    assert engine.thinker.seen_caches[2] is not committed
    assert state.thinker_static_cache is committed
    assert state.thinker_static_cache_ids == committed_ids == [10, 11]
    assert torch.equal(committed.layers[0].keys[:, :, :2], committed_keys)


def test_audio_response_v2_uses_request_bucket_and_keeps_cache_fresh() -> None:
    engine = _engine()
    engine.thinker_static_cache_max_len = 4096
    engine.audio_text_prefix_cache_reuse_enabled = True
    state = _state()
    first_audio = {
        "input_ids": torch.tensor([[10, 11, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "input_features": torch.tensor([[[2.0], [3.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }
    engine.think(
        [10, 11, 99, 99], max_new_tokens=1, temperature=0.0, state=state,
        input_audio=first_audio,
    )
    committed = state.thinker_static_cache
    assert committed is not None
    assert state.thinker_static_cache_ids == [10, 11]

    engine.audio_response_protocol = "heard_action_say_v2"
    long_ids = torch.full((1, 385), 10, dtype=torch.long)
    long_ids[0, -2:] = 99
    second_audio = {
        "input_ids": long_ids,
        "attention_mask": torch.ones_like(long_ids),
        "input_features": torch.tensor([[[7.0], [8.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }

    result = engine.think(
        long_ids[0].tolist(), max_new_tokens=128, temperature=0.0,
        state=state, input_audio=second_audio, stop_check=lambda: True,
    )

    assert result["runtime_metrics"]["thinker_cache_bucket_capacity"] == 1024
    assert result["runtime_metrics"]["thinker_cache_hit"] is False
    assert result["runtime_metrics"]["thinker_cache_reused_tokens"] == 0
    assert result["runtime_metrics"]["thinker_cache_reason"] == (
        "fresh:audio_cross_response_prefix_reuse_disabled_for_v2"
    )
    assert result["runtime_metrics"]["thinker_prefill_tokens"] == 385
    assert engine.thinker.calls[2]["input_ids"].shape == (1, 385)
    assert engine.thinker.calls[2]["past_length"] == 0
    bucket_cache = engine.thinker.seen_caches[2]
    assert bucket_cache is not committed
    assert bucket_cache.layers[0].max_cache_len == 1024
    assert state.thinker_static_cache is committed
    assert state.thinker_static_cache_ids == [10, 11]


def test_compiled_audio_prefix_cache_fail_closes_when_placeholder_mask_is_unavailable() -> None:
    engine = _engine()
    engine.thinker.model = SimpleNamespace()
    state = _state()
    audio = {
        "input_ids": torch.tensor([[10, 11, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "input_features": torch.tensor([[[2.0], [3.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }

    result = engine.think(
        [10, 11, 99, 99], max_new_tokens=1, temperature=0.0, state=state, input_audio=audio,
    )

    assert result["runtime_metrics"]["thinker_cache_hit"] is False
    assert result["runtime_metrics"]["thinker_cache_reason"] == "fresh:audio_placeholder_mask_unavailable"
    assert state.thinker_static_cache is None
    assert state.thinker_static_cache_ids == []


def test_compiled_audio_prefix_cache_publishes_only_when_generation_is_current() -> None:
    engine = _engine()
    audio = {
        "input_ids": torch.tensor([[10, 11, 99, 99]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "input_features": torch.tensor([[[2.0], [3.0]]]),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }

    current = _state()
    cancelled = engine.think(
        [10, 11, 99, 99], max_new_tokens=1, temperature=0.0, state=current,
        generation_id=4, stop_check=lambda: True, input_audio=audio,
    )
    assert cancelled["cancelled"] is True
    assert current.thinker_static_cache_ids == [10, 11]
    assert static_cache_sequence_length(current.thinker_static_cache) == 2

    class StaleState:
        def __init__(self) -> None:
            self.thinker_static_cache = None
            self.thinker_static_cache_ids = []
            self.thinker_static_cache_fingerprint = None
            self.stale_kv_writes = 0
            self._generation_lock = threading.RLock()

        @staticmethod
        def is_generation_current(*, generation_id: int) -> bool:
            return generation_id == 0

    stale = StaleState()
    result = engine.think(
        [20, 21, 99, 99], max_new_tokens=1, temperature=0.0, state=stale,
        generation_id=9, stop_check=lambda: True, input_audio=audio | {
            "input_ids": torch.tensor([[20, 21, 99, 99]], dtype=torch.long),
        },
    )
    assert result["cancelled"] is True
    assert result["runtime_metrics"]["thinker_cache_publish"] == "stale_generation"
    assert stale.thinker_static_cache is None
    assert stale.thinker_static_cache_ids == []
    assert stale.stale_kv_writes == 1
