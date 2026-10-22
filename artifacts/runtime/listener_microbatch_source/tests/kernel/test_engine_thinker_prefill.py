from __future__ import annotations

import threading
import time
import traceback
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from duplex_kernel.engine import PipelineEngine
from duplex_kernel.models.thinker import ThinkerModel


class _CausalCacheModel:
    """Tiny deterministic causal model that appends unprocessed IDs to its cache."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], int]] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
        logits_to_keep: int = 0,
    ):
        assert use_cache and output_hidden_states and return_dict
        assert logits_to_keep in (0, 1)
        prior = (
            torch.empty((1, 0), dtype=torch.long)
            if past_key_values is None
            else past_key_values
        )
        full_ids = torch.cat((prior, input_ids), dim=1)
        assert attention_mask is not None
        assert tuple(attention_mask.shape) == tuple(full_ids.shape)
        self.calls.append((input_ids[0].tolist(), int(prior.shape[1])))
        causal_hidden = torch.cumsum(full_ids.to(torch.float32), dim=1)[:, prior.shape[1]:]
        logits = torch.stack((causal_hidden, -causal_hidden), dim=-1)
        return SimpleNamespace(
            hidden_states=(causal_hidden.unsqueeze(-1),),
            logits=logits,
            past_key_values=full_ids,
        )


def _engine(model: _CausalCacheModel, chunk_tokens: int) -> PipelineEngine:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = model
    engine.thinker_prefill_chunk_tokens = chunk_tokens
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    return engine


def test_chunked_full_prefill_matches_one_shot_causal_logits_hidden_and_cache() -> None:
    token_ids = list(range(1, 36))
    one_shot = _CausalCacheModel()
    reference = one_shot(
        input_ids=torch.tensor([token_ids]),
        attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    chunked_model = _CausalCacheModel()
    engine = _engine(chunked_model, 16)
    last, hidden_parts, cache, mask = engine._run_thinker_prefill(token_ids)

    assert chunked_model.calls == [
        (token_ids[:16], 0),
        (token_ids[16:32], 16),
        (token_ids[32:], 32),
    ]
    assert torch.equal(torch.cat(hidden_parts, dim=1), reference.hidden_states[-1])
    assert torch.equal(last.logits[:, -1, :], reference.logits[:, -1, :])
    assert torch.equal(cache, reference.past_key_values)
    assert mask is not None and tuple(mask.shape) == (1, len(token_ids))
    assert not engine._thinker_lock.locked()


def test_thinker_generation_config_eos_is_traced_once_without_forwarding_or_speaking() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        eot_id = None

        @staticmethod
        def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "Ok." if 2 in tokens else ""

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 106, 50])
        config = SimpleNamespace(eos_token_id=[1, 106])

        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: torch.Tensor | None = None,
            use_cache: bool,
            logits_to_keep: int,
            output_hidden_states: bool,
            return_dict: bool,
        ) -> SimpleNamespace:
            assert use_cache and logits_to_keep == 1 and output_hidden_states and return_dict
            ids = input_ids[0].tolist()
            self.calls.append(ids)
            prior = torch.empty((1, 0), dtype=torch.long) if past_key_values is None else past_key_values
            full_ids = torch.cat((prior, input_ids), dim=1)
            vocab_logits = torch.full((1, 1, 128), -100.0)
            vocab_logits[0, 0, 2 if len(self.calls) == 1 else 106] = 100.0
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            return SimpleNamespace(
                logits=vocab_logits,
                hidden_states=(hidden,),
                past_key_values=full_ids,
            )

    model = Model()
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = model
    engine._compiled_thinker = None
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = Tokenizer()
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    observed_ids: list[int] = []

    result = engine.think(
        [1, 7], max_new_tokens=2, temperature=0.0, top_p=1.0,
        on_token_id=observed_ids.append,
    )

    assert result["text"] == "Ok."
    assert result["gen_ids"] == [2]
    assert result["terminal_reason"] == "eos"
    assert observed_ids == [2, 106]
    assert model.calls == [[1, 7], [2]]


def test_empty_thinker_error_preserves_first_eos_token_and_runtime_metrics() -> None:
    class Tokenizer:
        eos_token_id = 1
        eot_id = None

        @staticmethod
        def decode(_tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return ""

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 106, 50])
        config = SimpleNamespace(eos_token_id=[1, 106])

        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: torch.Tensor | None = None,
            use_cache: bool,
            logits_to_keep: int,
            output_hidden_states: bool,
            return_dict: bool,
        ) -> SimpleNamespace:
            assert use_cache and logits_to_keep == 1 and output_hidden_states and return_dict
            self.calls.append(input_ids[0].tolist())
            prior = torch.empty((1, 0), dtype=torch.long) if past_key_values is None else past_key_values
            full_ids = torch.cat((prior, input_ids), dim=1)
            logits = torch.full((1, input_ids.shape[1], 128), -100.0)
            logits[..., 106] = 100.0
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            return SimpleNamespace(
                logits=logits,
                hidden_states=(hidden,),
                past_key_values=full_ids,
            )

    model = Model()
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = model
    engine._compiled_thinker = None
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = Tokenizer()
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    observed_ids: list[int] = []

    with pytest.raises(RuntimeError, match="Thinker ha prodotto una risposta vuota") as raised:
        engine.think(
            [1, 7], max_new_tokens=8, temperature=0.0, top_p=1.0,
            on_token_id=observed_ids.append,
        )

    error = raised.value
    assert error.terminal_reason == "eos"
    assert error.terminal_token_id == 106
    assert error.empty_eos is True
    assert error.runtime_metrics["terminal_reason"] == "eos"
    assert error.runtime_metrics["terminal_token_id"] == 106
    assert observed_ids == [106]
    assert model.calls == [[1, 7]], "first EOS must stop before a further model forward"


@pytest.mark.parametrize("first_eos_id", [1, 106, 50])
def test_min_new_tokens_masks_first_eos_then_allows_normal_eos(first_eos_id: int) -> None:
    class Tokenizer:
        eos_token_id = 1
        eot_id = None

        @staticmethod
        def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "Capito." if 2 in tokens else ""

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 106, 50])
        config = SimpleNamespace(eos_token_id=[1, 106])

        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: torch.Tensor | None = None,
            use_cache: bool,
            logits_to_keep: int,
            output_hidden_states: bool,
            return_dict: bool,
        ) -> SimpleNamespace:
            assert use_cache and logits_to_keep == 1 and output_hidden_states and return_dict
            ids = input_ids[0].tolist()
            self.calls.append(ids)
            prior = torch.empty((1, 0), dtype=torch.long) if past_key_values is None else past_key_values
            full_ids = torch.cat((prior, input_ids), dim=1)
            logits = torch.full((1, 1, 128), -100.0)
            if len(self.calls) == 1:
                logits[0, 0, first_eos_id] = 100.0
                logits[0, 0, 2] = 99.0
            else:
                logits[0, 0, 106] = 100.0
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            return SimpleNamespace(
                logits=logits,
                hidden_states=(hidden,),
                past_key_values=full_ids,
            )

    model = Model()
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = model
    engine._compiled_thinker = None
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = Tokenizer()
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    observed_ids: list[int] = []

    result = engine.think(
        [1, 7], max_new_tokens=8, min_new_tokens=1,
        temperature=0.0, top_p=1.0, on_token_id=observed_ids.append,
    )

    assert result["text"] == "Capito."
    assert result["gen_ids"] == [2]
    assert result["terminal_reason"] == "eos"
    assert observed_ids == [2, 106]
    assert model.calls == [[1, 7], [2]], "EOS must be masked first, then stop without forwarding it"


def test_chunked_cached_suffix_matches_full_recompute() -> None:
    prefix = [3, 5, 7, 11]
    suffix = list(range(13, 42))
    model = _CausalCacheModel()
    engine = _engine(model, 16)
    prefix_out, prefix_hidden, prefix_cache, _ = engine._run_thinker_prefill(prefix)
    assert torch.equal(torch.cat(prefix_hidden, dim=1), prefix_out.hidden_states[-1])

    suffix_out, suffix_hidden, cache, mask = engine._run_thinker_prefill(
        suffix, past_key_values=prefix_cache, past_length=len(prefix)
    )
    full_model = _CausalCacheModel()
    full = full_model(
        input_ids=torch.tensor([prefix + suffix]),
        attention_mask=torch.ones((1, len(prefix) + len(suffix)), dtype=torch.long),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    expected_suffix_hidden = full.hidden_states[-1][:, len(prefix):]
    assert torch.equal(torch.cat(suffix_hidden, dim=1), expected_suffix_hidden)
    assert torch.equal(suffix_out.logits[:, -1, :], full.logits[:, -1, :])
    assert torch.equal(cache, full.past_key_values)


def test_runtime_cache_first_real_q4_is_eager_then_compiled_chunks_use_causal_positions() -> None:
    class StaticCache:
        def __init__(self, max_cache_len: int) -> None:
            self.max_cache_len = max_cache_len
            self.ids = torch.full((1, max_cache_len), -1, dtype=torch.long)

        def get_max_cache_shape(self) -> int:
            return self.max_cache_len

    class Model:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[int], list[int], list[int]]] = []

        def forward(self, kind: str, **kwargs: Any) -> SimpleNamespace:
            input_ids = kwargs["input_ids"]
            cache = kwargs["past_key_values"]
            attention_mask = kwargs["attention_mask"]
            cache_position = kwargs["cache_position"]
            position_ids = kwargs["position_ids"]
            assert tuple(attention_mask.shape) == (1, cache.max_cache_len)
            assert cache_position.ndim == 1
            assert torch.equal(position_ids, cache_position.unsqueeze(0))
            visible = int(cache_position[-1]) + 1
            assert torch.equal(attention_mask[0, :visible], torch.ones(visible, dtype=torch.long))
            assert torch.count_nonzero(attention_mask[0, visible:]) == 0
            cache.ids[:, cache_position] = input_ids
            self.calls.append((kind, input_ids[0].tolist(), cache_position.tolist(), attention_mask[0].tolist()))
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            logits = torch.stack((hidden, -hidden), dim=-1)
            return SimpleNamespace(
                hidden_states=(hidden,), logits=logits, past_key_values=cache,
            )

        def __call__(self, **kwargs: Any) -> SimpleNamespace:
            return self.forward("eager", **kwargs)

    model = Model()
    compiled_calls: list[list[int]] = []

    def compiled_model(**kwargs: Any) -> SimpleNamespace:
        compiled_calls.append(kwargs["input_ids"][0].tolist())
        return model.forward("compiled", **kwargs)

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = model
    engine._compiled_thinker = compiled_model
    engine.thinker_static_cache_max_len = 12
    engine.thinker_prefill_chunk_tokens = 4
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    cache = StaticCache(max_cache_len=12)

    output, hidden, returned_cache, _ = engine._run_thinker_prefill(
        list(range(1, 10)), is_compiled=True, static_cache=cache,
    )

    # The first Q4 is the real eager initialization pass for a fresh
    # Transformers 5.5 StaticCache; later fixed Q4 chunks use the compiled
    # forward, and the short final tail remains eager.
    assert compiled_calls == [[5, 6, 7, 8]]
    assert [row[0] for row in model.calls] == ["eager", "compiled", "eager"]
    assert model.calls[0][2] == [0, 1, 2, 3]
    assert model.calls[1][2] == [4, 5, 6, 7]
    assert model.calls[2][2] == [8]
    assert torch.equal(cache.ids[0, :9], torch.arange(1, 10))
    assert returned_cache is cache
    assert torch.equal(torch.cat(hidden, dim=1), torch.arange(1, 10, dtype=torch.float32).view(1, 9, 1))
    assert output.logits.shape[1] == 1


def test_zero_chunk_size_preserves_single_call_prefill() -> None:
    token_ids = list(range(1, 20))
    model = _CausalCacheModel()
    engine = _engine(model, 0)

    engine._run_thinker_prefill(token_ids)

    assert model.calls == [(token_ids, 0)]


def test_tiny_transformers_gemma4_chunked_cache_matches_one_shot() -> None:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
        num_global_key_value_heads=2,
        global_head_dim=8,
        use_cache=True,
    )
    model = Gemma4ForCausalLM(config).eval()
    tokens = list(range(3, 34))
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        reference = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    engine = _engine(model, 16)
    prefill_metrics: dict[str, object] = {}
    with torch.inference_mode():
        final, hidden_parts, cache, _ = engine._run_thinker_prefill(tokens, runtime_metrics=prefill_metrics)

    torch.testing.assert_close(final.logits[:, -1, :], reference.logits[:, -1, :], rtol=1e-4, atol=1e-5)
    assert final.logits.shape[1] == 1
    assert prefill_metrics["thinker_prefill_lock_hold_ms"] >= 0.0
    assert prefill_metrics["thinker_prefill_lock_wait_ms"] >= 0.0
    torch.testing.assert_close(
        torch.cat(hidden_parts, dim=1), reference.hidden_states[-1], rtol=1e-4, atol=1e-5
    )
    assert cache.get_seq_length() == len(tokens)

    prefix_length = 9
    suffix = tokens[prefix_length:]
    prefix_model = deepcopy(model)
    prefix_ids = torch.tensor([tokens[:prefix_length]], dtype=torch.long)
    with torch.inference_mode():
        prefix_out = prefix_model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        suffix_out, suffix_hidden, suffix_cache, suffix_mask = _engine(
            prefix_model, 16
        )._run_thinker_prefill(
            suffix,
            past_key_values=prefix_out.past_key_values,
            past_length=prefix_length,
        )

    torch.testing.assert_close(
        torch.cat(suffix_hidden, dim=1),
        reference.hidden_states[-1][:, prefix_length:],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        suffix_out.logits[:, -1, :], reference.logits[:, -1, :], rtol=1e-4, atol=1e-5
    )
    assert suffix_out.logits.shape[1] == 1
    assert suffix_mask is not None and tuple(suffix_mask.shape) == (1, len(tokens))
    assert suffix_cache.get_seq_length() == len(tokens)


def test_thinker_static_cache_uses_full_storage_copy_without_mutating_model_config() -> None:
    from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    config = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, sliding_window=16,
        vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
        num_global_key_value_heads=2, global_head_dim=8,
        num_kv_shared_layers=1,
    )
    config.layer_types = ["sliding_attention", "full_attention"]
    original_layer_types = list(config.layer_types)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.thinker = SimpleNamespace(config=config)
    engine.thinker_static_cache_max_len = 32

    cache = engine._new_thinker_static_cache()

    assert len(cache.layers) == 1  # one physical KV layer after the configured shared layer
    assert isinstance(cache.layers[0], StaticLayer)
    assert not isinstance(cache.layers[0], StaticSlidingWindowLayer)
    assert cache.layers[0].max_cache_len == 32
    assert config.layer_types == original_layer_types
    assert config.sliding_window == 16


def test_tiny_gemma4_inference_computes_only_last_prefill_logits() -> None:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, sliding_window=16,
        vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
        num_global_key_value_heads=2, global_head_dim=8, use_cache=True,
    )
    model = Gemma4ForCausalLM(config).eval()
    tokens = list(range(3, 16))
    token_tensor = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        reference = model(
            input_ids=token_tensor,
            attention_mask=torch.ones_like(token_tensor),
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        output, _, _, _ = _engine(model, 0)._run_thinker_prefill(tokens)

    assert output.logits.shape == (1, 1, config.vocab_size)
    torch.testing.assert_close(output.logits[:, -1, :], reference.logits[:, -1, :], rtol=1e-4, atol=1e-5)


def test_thinker_decode_yields_lock_when_listener_arrives_during_acquire() -> None:
    waiting = threading.Event()
    decode_entered = threading.Event()

    class ObservedLock:
        def __init__(self) -> None:
            self.inner = threading.Lock()
            self.acquire_count = 0
            self.release_count = 0
            self.second_attempt = threading.Event()
            self.allow_second = threading.Event()
            self.rejected_for_listener = threading.Event()

        def acquire(self) -> bool:
            self.acquire_count += 1
            if self.acquire_count == 2:
                waiting.set()
                self.second_attempt.set()
                assert self.allow_second.wait(2.0)
            return self.inner.acquire()

        def release(self) -> None:
            self.inner.release()
            self.release_count += 1
            if self.release_count == 2:
                self.rejected_for_listener.set()

        def __enter__(self) -> "ObservedLock":
            self.acquire()
            return self

        def __exit__(self, *_exc: object) -> None:
            self.release()

    class DecodeModel:
        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: torch.Tensor | None = None,
            use_cache: bool,
            output_hidden_states: bool,
            return_dict: bool,
            logits_to_keep: int = 0,
        ) -> SimpleNamespace:
            assert use_cache and output_hidden_states and return_dict
            assert logits_to_keep == 1
            if past_key_values is not None:
                decode_entered.set()
            prior = torch.empty((1, 0), dtype=torch.long) if past_key_values is None else past_key_values
            full_ids = torch.cat((prior, input_ids), dim=1)
            hidden = full_ids.to(torch.float32)[:, prior.shape[1]:].unsqueeze(-1)
            logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
            logits[..., 2] = 1.0
            return SimpleNamespace(
                hidden_states=(hidden,), logits=logits, past_key_values=full_ids,
            )

    class Tokenizer:
        eos_token_id = 3
        eot_id = None

        @staticmethod
        def decode(_tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "ok"

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = DecodeModel()
    engine._compiled_thinker = None
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = Tokenizer()
    engine.listener = SimpleNamespace(backend=SimpleNamespace(_waiting_for_lm=waiting))
    lock = ObservedLock()
    engine._thinker_lock = lock

    errors: list[BaseException] = []
    error_details: list[str] = []
    think_results: list[dict[str, object]] = []
    observed_token_ids: list[int] = []

    def run_think() -> None:
        try:
            think_results.append(engine.think(
                [1], max_new_tokens=1, temperature=0.0, top_p=1.0,
                on_token_id=observed_token_ids.append,
            ))
        except BaseException as exc:  # surfaced in the test thread below
            errors.append(exc)
            error_details.append(traceback.format_exc())

    worker = threading.Thread(target=run_think)
    worker.start()
    if not lock.second_attempt.wait(2.0):
        assert not errors, f"thinker è terminato prima del decode: {errors!r}\n{error_details[0]}"
        raise AssertionError("il thinker non ha raggiunto il decode entro il timeout")
    lock.allow_second.set()

    deadline = time.monotonic() + 2.0
    while not lock.rejected_for_listener.is_set() and not decode_entered.is_set():
        assert time.monotonic() < deadline, "thinker né cede né entra nel decode"
        time.sleep(0.001)

    assert lock.rejected_for_listener.is_set(), "thinker deve cedere il lock al listener in attesa"
    assert not decode_entered.is_set(), "decode non deve partire mentre il listener aspetta"
    waiting.clear()
    worker.join(2.0)
    assert not worker.is_alive()
    assert not errors
    runtime_metrics = think_results[0]["runtime_metrics"]
    assert runtime_metrics["thinker_decode_steps"] == 1
    assert runtime_metrics["thinker_prefill_chunks"] == 1
    assert runtime_metrics["thinker_prefill_ms"] >= 0.0
    assert runtime_metrics["thinker_decode_ms"] >= 0.0
    assert observed_token_ids == think_results[0]["gen_ids"] == [2]


def test_direct_audio_v2_q1_decode_does_not_wait_for_full_listener_admission() -> None:
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2

    admission_phases: list[str] = []

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        eot_id = None

        @staticmethod
        def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "Ok." if 2 in tokens else ""

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 106])
        config = SimpleNamespace(eos_token_id=[1, 106])
        model = SimpleNamespace(audio_tower=None)

        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            use_cache: bool,
            logits_to_keep: int,
            output_hidden_states: bool,
            return_dict: bool,
            past_key_values: torch.Tensor | None = None,
            input_features: torch.Tensor | None = None,
            input_features_mask: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            assert use_cache and logits_to_keep == 1 and output_hidden_states and return_dict
            prior = (
                torch.empty((1, 0), dtype=torch.long)
                if past_key_values is None else past_key_values
            )
            full_ids = torch.cat((prior, input_ids), dim=1)
            logits = torch.full((1, 1, 128), -100.0)
            if input_features is not None:
                assert input_features_mask is not None and past_key_values is None
                self.calls.append("audio_prefill")
                logits[..., 2] = 100.0
            else:
                self.calls.append("q1_decode")
                assert past_key_values is not None
                logits[..., 106] = 100.0
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            return SimpleNamespace(
                logits=logits,
                hidden_states=(hidden,),
                past_key_values=full_ids,
            )

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = Model()
    engine._compiled_thinker = None
    engine.thinker_prefill_chunk_tokens = 0
    engine.thinker_static_cache_max_len = 128
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.tokenizer = Tokenizer()
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    engine._prepare_thinker_audio_inputs = lambda *_args: {
        "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "input_features": torch.zeros((1, 2, 4)),
        "input_features_mask": torch.ones((1, 2), dtype=torch.long),
    }

    def listener_admission_wait() -> None:
        admission_phases.append("admission")

    result = engine.think(
        [4, 5, 6],
        input_audio={"pcm": np.zeros(1280, dtype=np.float32)},
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        listener_admission_wait=listener_admission_wait,
        disable_thinker_cadence_pacing=True,
    )

    assert result["gen_ids"] == [2]
    assert engine.thinker.calls == ["audio_prefill", "q1_decode"]
    assert admission_phases == ["admission"]
    assert result["runtime_metrics"]["thinker_prefill_chunks"] == 1
    assert result["runtime_metrics"]["thinker_decode_steps"] == 1


def test_native_asr_callback_guards_each_decode_but_not_audio_prefill() -> None:
    events: list[str] = []

    class Processor:
        audio_token = "<|audio|>"
        eoa_token = "<audio|>"

        def __call__(self, *, text: list[str], audio: list[np.ndarray], sampling_rate: int, return_tensors: str, padding: bool):
            assert text and audio and return_tensors == "pt" and padding
            assert sampling_rate == 16000
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "input_features": torch.zeros((1, 2, 4)),
                "input_features_mask": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 0

        @staticmethod
        def apply_chat_template(*_args: object, **_kwargs: object) -> str:
            return "Transcribe <|audio|>"

        @staticmethod
        def convert_tokens_to_ids(_token: str) -> int:
            return 3

        @staticmethod
        def decode(_tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "ciao"

    class Model:
        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            use_cache: bool,
            return_dict: bool,
            input_features: torch.Tensor | None = None,
            input_features_mask: torch.Tensor | None = None,
            past_key_values: torch.Tensor | None = None,
            logits_to_keep: int = 0,
        ) -> SimpleNamespace:
            assert use_cache and return_dict
            assert logits_to_keep == 1
            assert attention_mask.shape[0] == 1
            if input_features is not None:
                events.append("audio_prefill")
                assert input_features_mask is not None and past_key_values is None
                next_id = 2
            else:
                events.append("decode_forward")
                assert past_key_values is not None
                next_id = 3
            logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
            logits[..., next_id] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=input_ids)

    def decode_step(forward):
        events.append("lock_enter")
        result = forward()
        events.append("lock_exit")
        return result

    wrapper = ThinkerModel(Model(), Tokenizer(), Processor(), device="cpu")
    result = wrapper.transcribe_audio(
        np.zeros(320, dtype=np.float32), max_new_tokens=4, decode_step=decode_step
    )

    assert events == ["audio_prefill", "lock_enter", "decode_forward", "lock_exit"]
    assert result["text"] == "ciao" and result["gen_ids"] == [2]
    assert result["phase_timings_ms"]["decode_steps"] == 1
    assert all(value >= 0 for key, value in result["phase_timings_ms"].items() if key.endswith("_ms"))


def test_asr_uses_stage1_prompt_and_stops_on_generation_config_eos() -> None:
    from training.duplex_qlora.stage1 import _format_chat

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0
        audio_token = "<|audio|>"

        @staticmethod
        def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False and add_generation_prompt is True
            content = messages[0]["content"]
            return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>assistant\n"

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "<audio|>"
            return 258883

        @staticmethod
        def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "Venezia" if tokens == [2] else ""

    class Processor:
        audio_token = "<|audio|>"
        eoa_token = "<audio|>"

        def __init__(self) -> None:
            self.tokenizer = Tokenizer()
            self.calls: list[dict[str, object]] = []

        def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
            self.calls.append({
                "text": list(text),
                "audio": list(audio),
                "sampling_rate": sampling_rate,
                "return_tensors": return_tensors,
                "padding": padding,
            })
            return {
                "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
                "input_features": torch.zeros((1, 4, 8)),
                "input_features_mask": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 106, 50])
        config = SimpleNamespace(eos_token_id=[1, 106])

        def __init__(self) -> None:
            self.calls: list[tuple[list[int], bool]] = []

        def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep, return_dict,
                     input_features=None, input_features_mask=None, past_key_values=None):
            assert use_cache and logits_to_keep == 1 and return_dict
            is_prefill = input_features is not None
            assert is_prefill == (past_key_values is None)
            if is_prefill:
                assert input_features_mask is not None
                next_id = 2
            else:
                next_id = 106
            self.calls.append((input_ids[0].tolist(), is_prefill))
            logits = torch.full((1, input_ids.shape[1], 258884), -100.0)
            logits[..., next_id] = 100.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    processor = Processor()
    model = Model()
    pcm = np.zeros(1600, dtype=np.float32)
    wrapper = ThinkerModel(model, processor.tokenizer, processor, device="cpu")

    result = wrapper.transcribe_audio(pcm, max_new_tokens=32)
    expected_prompt = _format_chat(
        processor,
        {"task_kind": "asr", "language": "it"},
        add_target=False,
    )

    assert processor.calls[0]["text"] == [expected_prompt]
    assert "Transcribe the speech faithfully in Italian; preserve the original wording. <|audio|>" in expected_prompt
    assert processor.calls[0]["sampling_rate"] == 16000
    assert np.array_equal(processor.calls[0]["audio"][0], pcm)
    assert result["text"] == "Venezia"
    assert result["gen_ids"] == [2]
    assert result["terminal_token_id"] == 106
    assert result["terminal_reason"] == "eos"
    assert result["phase_timings_ms"]["decode_steps"] == 1
    assert model.calls == [([7, 8], True), ([2], False)]


def test_asr_cancellation_after_atomic_audio_prefill_skips_all_decode_tokens() -> None:
    cancelled = threading.Event()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def apply_chat_template(_messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert not tokenize and add_generation_prompt
            return "audio prompt"

        @staticmethod
        def convert_tokens_to_ids(_token: str) -> int:
            return 3

        @staticmethod
        def decode(_tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return ""

    class Processor:
        audio_token = "<|audio|>"
        eoa_token = "<audio|>"
        tokenizer = Tokenizer()

        @staticmethod
        def __call__(*, text, audio, sampling_rate, return_tensors, padding):
            assert text == ["audio prompt"] and len(audio) == 1
            assert sampling_rate == 16000 and return_tensors == "pt" and padding
            return {
                "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
                "input_features": torch.zeros((1, 1, 8)),
                "input_features_mask": torch.ones((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    class Model:
        generation_config = SimpleNamespace(eos_token_id=1)
        config = SimpleNamespace(eos_token_id=1)

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *, input_ids, **_kwargs):
            self.calls += 1
            cancelled.set()
            logits = torch.zeros((1, input_ids.shape[1], 5), dtype=torch.float32)
            logits[..., 2] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    model = Model()
    result = ThinkerModel(model, Tokenizer(), Processor(), device="cpu").transcribe_audio(
        np.zeros(320, dtype=np.float32),
        max_new_tokens=4,
        stop_check=cancelled.is_set,
    )

    assert model.calls == 1  # The audio prefill is atomic; no decode token follows it.
    assert result["cancelled"] is True
    assert result["text"] == ""
    assert result["gen_ids"] == []
    assert result["terminal_reason"] == "cancelled"
    assert result["phase_timings_ms"]["decode_steps"] == 0


def test_engine_asr_uses_listener_priority_for_audio_prefill_and_decode() -> None:
    waiting = threading.Event()
    waiting.set()
    shared_lock = threading.Lock()
    listener = SimpleNamespace(backend=SimpleNamespace(_waiting_for_lm=waiting))
    events: list[str] = []

    class Wrapper:
        def transcribe_audio(self, _pcm, *, max_new_tokens: int, decode_step, prefill_step):
            assert max_new_tokens == 64
            assert waiting.is_set()
            def prefill_forward():
                assert not waiting.is_set(), "il prefill deve cedere al frame listener in attesa"
                assert shared_lock.locked()
                events.append("audio_prefill")
                return object()

            prefill_step(prefill_forward)

            def forward():
                assert not waiting.is_set(), "il decode deve aspettare la fine del frame listener"
                assert shared_lock.locked()
                events.append("decode")
                return object()

            decode_step(forward)
            return {
                "text": "ciao",
                "phase_timings_ms": {
                    "feature_prepare_ms": 2.0,
                    "audio_prefill_ms": 7.0,
                    "decode_ms": 3.0,
                    "decode_steps": 1,
                },
            }

    engine = PipelineEngine.__new__(PipelineEngine)
    engine._thinker_wrapper = Wrapper()
    engine.listener = listener
    engine._thinker_lock = shared_lock

    clearer = threading.Thread(target=lambda: (time.sleep(0.02), waiting.clear()))
    clearer.start()
    result = engine.transcribe_audio(np.zeros(320, dtype=np.float32))
    clearer.join(2.0)

    assert not clearer.is_alive()
    assert events == ["audio_prefill", "decode"]
    timings = result["phase_timings_ms"]
    assert timings["feature_prepare_ms"] == 2.0
    assert timings["audio_prefill_ms"] == 7.0
    assert timings["shared_lm_prefill_lock_wait_ms"] >= 10.0
    assert timings["shared_lm_prefill_lock_hold_ms"] >= 0.0
    assert timings["shared_lm_lock_wait_ms"] >= 0.0
    assert timings["shared_lm_lock_hold_ms"] >= 0.0
    assert timings["total_asr_ms"] >= timings["audio_prefill_ms"]


def test_runtime_unsupported_static_cache_format_fails_closed_without_session_sharing() -> None:
    import transformers

    class TinyStaticCache:
        instances: list["TinyStaticCache"] = []

        def __init__(self, *, config, max_batch_size: int, max_cache_len: int, device, dtype) -> None:
            self.config = config
            self.max_batch_size = max_batch_size
            self.max_cache_len = max_cache_len
            self.entries = torch.empty((max_batch_size, 0), device=device, dtype=torch.long)
            self.initialized = False
            self.instances.append(self)

    class FakeGemmaConfig:
        layer_types = ["sliding_attention"]
        num_hidden_layers = 1
        sliding_window = 16
        num_kv_shared_layers = 1

        def get_text_config(self, decoder: bool = True):
            assert decoder
            return self

    class CompiledModel:
        def __init__(self) -> None:
            self.config = FakeGemmaConfig()

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            past_key_values: TinyStaticCache,
            use_cache: bool,
            output_hidden_states: bool,
            return_dict: bool,
            **_kwargs: object,
        ) -> SimpleNamespace:
            assert use_cache and output_hidden_states and return_dict
            past_key_values.entries = torch.cat((past_key_values.entries, input_ids), dim=1)
            past_key_values.initialized = True
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
            logits[..., 2] = 1.0
            return SimpleNamespace(
                hidden_states=(hidden,), logits=logits, past_key_values=past_key_values,
            )

    class Tokenizer:
        eos_token_id = 3
        eot_id = None

        @staticmethod
        def decode(_tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "ok"

    def session_state():
        return SimpleNamespace(
            thinker_cache_ids=[], thinker_kv=None, thinker_hidden=None, stale_kv_writes=0,
            _generation_lock=threading.Lock(),
        )

    old_static_cache = transformers.StaticCache
    transformers.StaticCache = TinyStaticCache
    try:
        engine = PipelineEngine.__new__(PipelineEngine)
        engine.device = "cpu"
        engine.dtype = torch.float32
        engine.thinker = CompiledModel()
        def compiled_forward(**kwargs: object) -> SimpleNamespace:
            cache = kwargs["past_key_values"]
            assert cache.initialized, "il forward compilato deve vedere cache già inizializzata"
            return engine.thinker(**kwargs)

        engine._compiled_thinker = compiled_forward
        engine.thinker_static_cache_max_len = 8
        engine.thinker_prefill_chunk_tokens = 0
        engine.tokenizer = Tokenizer()
        engine.listener = None
        engine._thinker_lock = threading.Lock()
        first_state = session_state()
        second_state = session_state()

        first_result = engine.think([1], max_new_tokens=1, temperature=0.0, state=first_state)
        first_cache = TinyStaticCache.instances[-2]
        first_entries = first_cache.entries.clone()
        second_result = engine.think([5], max_new_tokens=1, temperature=0.0, state=second_state)
    finally:
        transformers.StaticCache = old_static_cache

    assert len(TinyStaticCache.instances) == 4
    assert first_result["runtime_metrics"]["thinker_cache_init_method"] == "eager_first_real_chunk"
    assert first_result["runtime_metrics"]["thinker_cache_init_ms"] >= 0.0
    assert first_result["runtime_metrics"]["thinker_cache_snapshot_reason"].startswith("copy_unsupported:")
    assert second_result["runtime_metrics"]["thinker_cache_init_method"] == "eager_first_real_chunk"
    assert second_result["runtime_metrics"]["thinker_cache_init_ms"] >= 0.0
    assert first_state.thinker_kv is None
    assert second_state.thinker_kv is None
    assert first_cache.initialized
    assert first_cache is not TinyStaticCache.instances[-2]
    assert torch.equal(first_cache.entries, first_entries)


def test_compiled_static_cache_rejects_prompt_and_budget_over_capacity() -> None:
    class Model:
        config = SimpleNamespace()

        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            raise AssertionError("il capacity check deve precedere ogni forward")

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.thinker = Model()
    engine._compiled_thinker = engine.thinker
    engine.thinker_static_cache_max_len = 8
    engine.thinker_prefill_chunk_tokens = 4
    engine.tokenizer = SimpleNamespace(eos_token_id=1, eot_id=None)
    engine.listener = None
    engine._thinker_lock = threading.Lock()

    try:
        engine.think([1, 2, 3, 4, 5, 6], max_new_tokens=3)
    except ValueError as exc:
        assert "6 + 3 > 8" in str(exc)
        assert "aumentare la capacità" in str(exc)
    else:
        raise AssertionError("la capacità statica non deve troncare il budget implicitamente")


def test_compiled_thinker_warmup_runs_after_all_backend_initialization(monkeypatch, tmp_path) -> None:
    import duplex_kernel.engine as engine_module
    import duplex_kernel.listener as listener_module
    import duplex_kernel.subtalker_graph as subtalker_module
    from duplex_kernel.config import V9_PROFILE_QUALITY_FINAL_ORACLE

    events: list[str] = []
    thinker_wrapper = SimpleNamespace(model=object(), tokenizer=object())

    def fake_load_thinker(*_args, **_kwargs):
        events.append("thinker")
        return thinker_wrapper

    class FakeListener:
        @staticmethod
        def from_thinker(_thinker, *, runtime_profile: str):
            assert runtime_profile == "production"
            events.append("listener")
            return SimpleNamespace(backend=SimpleNamespace(_lock=threading.Lock()))

    class FakeTalker:
        def __init__(self):
            self.talker = SimpleNamespace(
                code_predictor=object(),
                get_input_embeddings=lambda: object(),
            )

    def fake_load_talker(*_args, **_kwargs):
        events.append("talker")
        return FakeTalker()

    def fake_load_v9(*_args, **_kwargs):
        events.append("v9")
        return SimpleNamespace(_v9_recipe="fake-v9"), object()

    class FakeSubtalkerGraph:
        def __init__(self, *_args, **_kwargs):
            events.append("subtalker_init")

        def capture(self):
            events.append("subtalker_capture")
            return SimpleNamespace(captured=True)

    monkeypatch.setattr(engine_module, "load_thinker", fake_load_thinker)
    monkeypatch.setattr(engine_module, "load_talker", fake_load_talker)
    monkeypatch.setattr(engine_module, "load_v9_conditioning", fake_load_v9)
    monkeypatch.setattr(listener_module, "GemmaListener", FakeListener)
    monkeypatch.setattr(subtalker_module, "SubtalkerGraph", FakeSubtalkerGraph)

    engine = PipelineEngine(
        device="cpu",
        v9_profile=V9_PROFILE_QUALITY_FINAL_ORACLE,
        thinker_compile_mode="default",
    )
    monkeypatch.setattr(engine, "_prepare_compiled_thinker", lambda: events.append("compile_warmup"))

    assert engine.load(workspace=str(tmp_path))
    assert engine.loaded
    assert events == [
        "thinker",
        "listener",
        "talker",
        "v9",
        "subtalker_init",
        "subtalker_capture",
        "compile_warmup",
    ]


def test_asr_static_cache_decode_uses_physical_cache_position_and_preserves_padding_mask() -> None:
    events: list[str] = []

    class Cache:
        def __init__(self) -> None:
            self.length = 0

        def get_seq_length(self) -> int:
            return self.length

    class Processor:
        audio_token = "<|audio|>"
        eoa_token = "<audio|>"

        def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
            assert text and len(audio) == 1 and sampling_rate == 16000
            assert return_tensors == "pt" and padding is True
            return {
                "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
                "input_features": torch.ones((1, 2, 4), dtype=torch.float32),
                "input_features_mask": torch.ones((1, 2), dtype=torch.long),
                # Exercise a padded token: the cache position remains physical
                # length 3, while the attention prefix preserves the 0 bit.
                "attention_mask": torch.tensor([[1, 0, 1]], dtype=torch.long),
            }

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "<|audio|>"

        @staticmethod
        def convert_tokens_to_ids(_token: str) -> int:
            return 3

        @staticmethod
        def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "ciao" if tokens == [2] else ""

    class Model:
        def __call__(self, *, input_ids, attention_mask, input_features, input_features_mask,
                     past_key_values, use_cache, logits_to_keep, return_dict):
            assert use_cache and logits_to_keep == 1 and return_dict
            assert input_features is not None and input_features_mask is not None
            assert tuple(attention_mask.shape) == (1, 3)
            assert past_key_values is cache
            events.append("audio_prefill")
            cache.length = input_ids.shape[1]
            logits = torch.zeros((1, input_ids.shape[1], 5), dtype=torch.float32)
            logits[..., 2] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=cache)

    cache = Cache()
    decode_positions: list[tuple[int, list[int]]] = []

    def compiled_decode(token, past, attention_mask):
        assert token.shape == (1, 1)
        position = past.get_seq_length()
        decode_positions.append((position, attention_mask[0].tolist()))
        events.append("compiled_decode")
        past.length += 1
        logits = torch.zeros((1, 1, 5), dtype=torch.float32)
        logits[..., 3] = 1.0  # EOA: stop after one actual generated token.
        return SimpleNamespace(logits=logits, past_key_values=past)

    def decode_step(forward):
        events.append("lock_enter")
        result = forward()
        events.append("lock_exit")
        return result

    result = ThinkerModel(Model(), Tokenizer(), Processor(), device="cpu").transcribe_audio(
        np.zeros(320, dtype=np.float32),
        max_new_tokens=2,
        decode_step=decode_step,
        past_key_values=cache,
        decode_forward=compiled_decode,
        static_cache_max_len=8,
    )

    assert events == ["audio_prefill", "lock_enter", "compiled_decode", "lock_exit"]
    assert decode_positions == [(3, [1, 0, 1, 1])]
    assert cache.get_seq_length() == 4
    assert result["gen_ids"] == [2]
    assert result["text"] == "ciao"


def test_static_thinker_kwargs_keep_asr_prefix_bits_in_official_gemma4_mask(monkeypatch) -> None:
    import transformers.masking_utils as masking_utils

    original_config = object()
    cache = SimpleNamespace()
    captured: dict[str, Any] = {}

    class Model:
        config = original_config

        @staticmethod
        def get_input_embeddings():
            return lambda input_ids: input_ids.to(torch.float32).unsqueeze(-1)

    def create_masks_for_generate(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {"official_mask": torch.tensor(1)}

    monkeypatch.setattr(masking_utils, "create_masks_for_generate", create_masks_for_generate)
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = Model()
    engine.thinker_static_cache_max_len = 8
    prompt_mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.long)
    input_ids = torch.tensor([[42]], dtype=torch.long)

    kwargs = engine._static_thinker_kwargs(
        input_ids,
        cache,
        start_position=3,
        visible_tokens=4,
        precompute_attention_masks=True,
        attention_mask_prefix=prompt_mask,
    )

    assert captured["config"] is original_config
    assert captured["past_key_values"] is cache
    assert captured["position_ids"].tolist() == [[3]]
    assert captured["attention_mask"].tolist() == [[1, 0, 1, 1, 0, 0, 0, 0]]
    assert kwargs["attention_mask"] is not None
    assert "official_mask" in kwargs["attention_mask"]


def test_backchannel_uses_only_training_phrase_trie_and_other_actions_remain_unconstrained() -> None:
    from duplex_kernel.engine import _build_backchannel_token_trie

    class Tokenizer:
        eos_token_id = 1
        eot_id = None

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return {"Capito.": [10, 11], "Sì.": [12, 13]}[text]

        @staticmethod
        def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return {
                (10, 11): "Capito.",
                (12, 13): "Sì.",
                (99,): "La frase non è un backchannel.",
            }.get(tuple(tokens), "")

    class ChoiceModel:
        generation_config = SimpleNamespace(eos_token_id=[1, 106, 50])
        config = SimpleNamespace(eos_token_id=[1, 106])

        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            past_key_values: torch.Tensor | None = None,
            use_cache: bool,
            logits_to_keep: int,
            output_hidden_states: bool,
            return_dict: bool,
        ) -> Any:
            assert use_cache and output_hidden_states and return_dict
            assert logits_to_keep == 1
            prior = past_key_values if past_key_values is not None else torch.empty((1, 0), dtype=torch.long)
            full = torch.cat((prior, input_ids), dim=1)
            assert attention_mask.shape == full.shape
            self.calls.append(input_ids[0].tolist())
            logits = torch.zeros((1, input_ids.shape[1], 120), dtype=torch.float32)
            current = int(input_ids[0, -1])
            if past_key_values is None:
                logits[..., 99] = 100.0  # highest raw logit, deliberately outside the phrase trie
                logits[..., 10] = 10.0
                logits[..., 12] = 9.0
            elif current == 10:
                logits[..., 99] = 100.0
                logits[..., 11] = 10.0
            elif current == 11:
                logits[..., 106] = 100.0
            elif current == 99:
                logits[..., 106] = 100.0
            hidden = torch.ones((1, input_ids.shape[1], 2), dtype=torch.float32)
            return SimpleNamespace(
                logits=logits,
                hidden_states=(hidden,),
                past_key_values=full,
            )

    tokenizer = Tokenizer()
    trie = _build_backchannel_token_trie(
        ["Capito.", "Sì."], tokenizer, eos_token_ids={1, 106, 50}
    )
    assert trie[()] == frozenset({10, 12})
    assert trie[(10,)] == frozenset({11})
    assert trie[(10, 11)] == frozenset({1, 50, 106})
    assert 99 not in trie[()]

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine.thinker = ChoiceModel()
    engine.thinker_prefill_chunk_tokens = 0
    engine.tokenizer = tokenizer
    engine.listener = None
    engine._thinker_lock = threading.Lock()
    engine._backchannel_token_trie = trie
    engine._backchannel_phrase_count = 2
    observed: list[int] = []

    constrained = engine.think(
        [1, 7], max_new_tokens=16, min_new_tokens=1,
        temperature=0.0, top_p=1.0, action="BACKCHANNEL",
        on_token_id=observed.append,
    )
    assert constrained["text"] == "Capito."
    assert constrained["gen_ids"] == [10, 11]
    assert observed == [10, 11, 106]
    assert constrained["terminal_reason"] == "eos"
    assert engine.thinker.calls == [[1, 7], [10], [11]]

    unconstrained_model = ChoiceModel()
    engine.thinker = unconstrained_model
    unconstrained = engine.think(
        [1, 7], max_new_tokens=4, temperature=0.0, top_p=1.0,
        action="SPEAK",
    )
    assert unconstrained["gen_ids"] == [99]
    assert unconstrained["text"] == "La frase non è un backchannel."
