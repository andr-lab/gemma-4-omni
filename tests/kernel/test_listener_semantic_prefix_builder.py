from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.listener import (
    AUDIO_START_AFTER_TEXT,
    GemmaAudioTowerBackend,
    GemmaListenerState,
    _gemma_lm_cache_can_advance,
)


class _Cache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self, _layer=0) -> int:
        return self.length

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)


class _LanguageModel:
    def __init__(self) -> None:
        self.config = object()
        self.embed_tokens = _EmbedTokens()

    def get_per_layer_inputs(self, token_ids, embeds):
        del token_ids
        return embeds


class _EmbedTokens:
    def __init__(self) -> None:
        self.weight = SimpleNamespace(dtype=torch.float32)

    def __call__(self, token_ids):
        return torch.zeros((*token_ids.shape, 1536), dtype=torch.float32, device=token_ids.device)


class _ShapeStable:
    def __init__(self) -> None:
        self.calls = []

    def forward(
        self,
        *,
        inputs_embeds,
        per_layer_inputs,
        attention_mask,
        position_ids,
        past_key_values,
        shared_kv_states,
    ):
        del per_layer_inputs, attention_mask, shared_kv_states
        prior = 0 if past_key_values is None else past_key_values.get_seq_length(0)
        self.calls.append((tuple(inputs_embeds.shape), position_ids.cpu().tolist(), prior))
        return SimpleNamespace(past_key_values=_Cache(prior + int(inputs_embeds.shape[1])))


def test_semantic_prefix_cache_uses_actual_prefix_then_separate_prior_q2_queries(monkeypatch):
    transformers_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    mask_calls = []

    def create_masks(_config, embeds, attention, *, past_key_values, position_ids):
        mask_calls.append((tuple(embeds.shape), attention.clone(), past_key_values, position_ids.clone()))
        return object()

    monkeypatch.setattr(transformers_modeling, "create_masks_for_generate", create_masks)
    backend = object.__new__(GemmaAudioTowerBackend)
    backend._shape_stable_lm = _ShapeStable()
    backend.gemma_model = SimpleNamespace(language_model=_LanguageModel())
    backend.device = "cpu"
    backend.hidden_size = 1536
    backend._frame_pad_ple = None
    backend.pad_token_id = 0
    backend.boa_token_id = 777

    first = torch.full((2, 1536), 1.0)
    second = torch.full((2, 1536), 2.0)
    result = GemmaAudioTowerBackend.build_semantic_prefix_cache(
        backend,
        (11, 12, 777),
        replay_q2=(first, second),
        replay_frame_ids=(4, 5),
    )

    assert result["prefix_len"] == 3
    assert result["cache_position"] == 7
    assert result["replay_frame_ids"] == (4, 5)
    assert result["past_key_values"].get_seq_length() == 7
    assert [call[0] for call in backend._shape_stable_lm.calls] == [
        (1, 3, 1536), (1, 2, 1536), (1, 2, 1536)
    ]
    assert backend._shape_stable_lm.calls[0][1] == [[0, 1, 2]]
    assert backend._shape_stable_lm.calls[1][1] == [[3, 4]]
    assert backend._shape_stable_lm.calls[2][1] == [[5, 6]]
    assert [call[2] for call in backend._shape_stable_lm.calls] == [0, 3, 5]
    assert [call[0] for call in mask_calls] == [(1, 3, 1536), (1, 2, 1536), (1, 2, 1536)]
    assert mask_calls[0][1].shape == (1, 3)
    assert mask_calls[1][1].shape == (1, 5)
    assert mask_calls[2][1].shape == (1, 7)


def test_semantic_prefix_prefill_chunks_above_canonical_limit_and_crops_cache(monkeypatch):
    transformers_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    mask_calls = []

    def create_masks(_config, embeds, attention, *, past_key_values, position_ids):
        mask_calls.append((tuple(embeds.shape), attention.clone(), position_ids.clone()))
        return object()

    monkeypatch.setattr(transformers_modeling, "create_masks_for_generate", create_masks)

    class _PaddedShapeStable(_ShapeStable):
        target_length = 291

        def forward(self, *, inputs_embeds, **kwargs):
            past_key_values = kwargs["past_key_values"]
            prior = 0 if past_key_values is None else past_key_values.get_seq_length(0)
            output = super().forward(inputs_embeds=inputs_embeds, **kwargs)
            # The real wrapper pads projection/attention workspaces. Simulate a
            # cache that includes that padded tail so every chunk must crop it.
            output.past_key_values.length = prior + self.target_length
            return output

    backend = object.__new__(GemmaAudioTowerBackend)
    backend._shape_stable_lm = _PaddedShapeStable()
    backend.gemma_model = SimpleNamespace(language_model=_LanguageModel())
    backend.device = "cpu"
    backend.hidden_size = 1536
    backend._frame_pad_ple = None
    backend.pad_token_id = 0
    backend.boa_token_id = 777
    prefix_ids = tuple(range(348)) + (777,)

    result = GemmaAudioTowerBackend.build_semantic_prefix_cache(backend, prefix_ids)

    assert result["prefix_len"] == 349
    assert result["cache_position"] == 349
    assert result["past_key_values"].get_seq_length() == 349
    assert [call[0] for call in backend._shape_stable_lm.calls] == [
        (1, 291, 1536), (1, 58, 1536)
    ]
    assert backend._shape_stable_lm.calls[0][1] == [list(range(291))]
    assert backend._shape_stable_lm.calls[1][1] == [list(range(291, 349))]
    assert [call[2] for call in backend._shape_stable_lm.calls] == [0, 291]
    assert [call[0] for call in mask_calls] == [(1, 291, 1536), (1, 58, 1536)]
    assert [tuple(call[1].shape) for call in mask_calls] == [(1, 291), (1, 349)]
    assert mask_calls[0][2].tolist() == [list(range(291))]
    assert mask_calls[1][2].tolist() == [list(range(291, 349))]


@pytest.mark.parametrize("replay_count", [0, 2])
def test_native_semantic_prefix_import_uses_actual_length_and_replay_cursor(monkeypatch, replay_count):
    transformers_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

    def create_masks(_config, _embeds, _attention, *, past_key_values, position_ids):
        del past_key_values, position_ids
        return object()

    monkeypatch.setattr(transformers_modeling, "create_masks_for_generate", create_masks)

    class _Native:
        output_tensor = torch.zeros((1, 2, 1536))

        def __init__(self):
            self.reset_count = 0
            self.imports = []
            self.current_position = AUDIO_START_AFTER_TEXT

        def reset(self):
            self.reset_count += 1

        def import_cache_async(self, cache, *, current_position, prefix_len, prefix_mask):
            self.imports.append((cache, current_position, prefix_len, prefix_mask.clone()))
            self.current_position = current_position

    class _FairLock:
        def acquire_fair(self, _purpose):
            pass

        def release(self):
            pass

    backend = object.__new__(GemmaAudioTowerBackend)
    backend._use_native_lm = True
    backend._shape_stable_lm = _ShapeStable()
    backend.gemma_model = SimpleNamespace(language_model=_LanguageModel())
    backend.device = "cpu"
    backend.hidden_size = 1536
    backend._frame_pad_ple = None
    backend.pad_token_id = 0
    backend.boa_token_id = 777
    backend.runtime_profile = "compatibility"
    backend._native_runtime = None
    backend._lock = _FairLock()

    native = _Native()
    state = GemmaListenerState(next_frame_id=9)
    state.gemma_lm_native = native
    replay_ids = tuple(range(9 - replay_count, 9))
    replay = (
            torch.arange(replay_count * 2 * 1536, dtype=torch.float32).reshape(
                1, replay_count * 2, 1536
        )
        if replay_count else None
    )

    result = GemmaAudioTowerBackend.prime_semantic_prefix(
        backend,
        state,
        (11, 12, 777),
        replay_audio_soft=replay,
        replay_frame_ids=replay_ids,
    )

    assert native.reset_count == 1
    assert len(native.imports) == 1
    imported_cache, current_position, prefix_len, prefix_mask = native.imports[0]
    assert imported_cache is result["past_key_values"]
    assert prefix_len == 3
    assert current_position == 3 + 2 * replay_count
    assert tuple(prefix_mask.shape) == (1, 3)
    assert torch.equal(prefix_mask, torch.ones((1, 3), dtype=torch.long))
    assert state.gemma_lm_prefix_len == 3
    assert torch.equal(state.gemma_lm_prefix_attention, prefix_mask)
    assert state.gemma_lm_audio_frames == replay_count
    assert state.gemma_lm_source_frame_ids == list(replay_ids)
    assert state.gemma_lm_context_start_frame == (replay_ids[0] if replay_ids else 9)
    assert state.gemma_lm_last_source_frame_id == (replay_ids[-1] if replay_ids else None)
    assert state.gemma_lm_cache is native
    state.semantic_context_revision = 1
    assert _gemma_lm_cache_can_advance(state)


def test_semantic_prefix_rejects_request_length_that_disagrees_with_token_ids():
    backend = object.__new__(GemmaAudioTowerBackend)
    backend.boa_token_id = 777
    with pytest.raises(ValueError, match="does not match its token IDs"):
        GemmaAudioTowerBackend.prime_semantic_prefix(
            backend,
            GemmaListenerState(),
            (11, 12, 777),
            prefix_len=4,
        )


def test_invalidating_native_transaction_clears_semantic_prefix_for_next_refresh():
    class _Native:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    state = GemmaListenerState(
        gemma_lm_native=_Native(),
        gemma_lm_cache=object(),
        gemma_lm_prefix_len=141,
        gemma_lm_prefix_attention=torch.ones((1, 141), dtype=torch.long),
        semantic_context_revision=4,
    )
    state.invalidate_gemma_lm_state()

    assert state.gemma_lm_native.reset_count == 1
    assert state.gemma_lm_cache is None
    assert state.gemma_lm_prefix_len == AUDIO_START_AFTER_TEXT
    assert state.gemma_lm_prefix_attention is None
    assert state.semantic_context_revision is None


def test_canonical_native_prefix_import_remains_at_34_without_semantic_kwargs(monkeypatch):
    transformers_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

    def create_masks(_config, _embeds, _attention, *, past_key_values, position_ids):
        del past_key_values, position_ids
        return object()

    monkeypatch.setattr(transformers_modeling, "create_masks_for_generate", create_masks)

    class _Native:
        output_tensor = torch.zeros((1, 2, 1536))

        def __init__(self):
            self.imports = []

        def reset(self):
            pass

        def import_cache_async(self, cache, *, current_position, **kwargs):
            self.imports.append((cache, current_position, kwargs))

    backend = object.__new__(GemmaAudioTowerBackend)
    backend._use_native_lm = True
    backend.gemma_model = SimpleNamespace(language_model=_LanguageModel(), config=object())
    backend.device = "cpu"
    backend.bos_token_id = 2
    backend.boa_token_id = 777
    backend._shape_stable_lm = _ShapeStable()
    backend._canonical_prefix_cache_enabled = False
    backend._canonical_prefix_cache_snapshot = None
    backend._canonical_prefix_cache_signature = None
    backend._canonical_prefix_compute_count = 0
    backend._canonical_prefix_reuse_count = 0
    backend._native_runtime = None

    native = _Native()
    state = GemmaListenerState(gemma_lm_native=native)
    input_ids = torch.zeros((1, AUDIO_START_AFTER_TEXT), dtype=torch.long)
    input_ids[0, 0] = 2
    input_ids[0, -1] = 777
    attention = torch.zeros((1, AUDIO_START_AFTER_TEXT), dtype=torch.long)
    attention[0, 0] = attention[0, -1] = 1
    embeds = torch.zeros((1, AUDIO_START_AFTER_TEXT, 1536))

    GemmaAudioTowerBackend._forward_incremental_lm(
        backend,
        embeds=embeds,
        per_layer=embeds,
        attention=attention,
        positions=torch.arange(AUDIO_START_AFTER_TEXT),
        state=state,
        input_ids=input_ids,
    )

    assert state.gemma_lm_prefix_len == AUDIO_START_AFTER_TEXT == 34
    assert state.gemma_lm_prefix_attention is None
    assert len(native.imports) == 1
    _cache, current_position, kwargs = native.imports[0]
    assert current_position == 34
    assert kwargs == {}


def test_shared_backend_switches_semantic_prefix_import_between_private_sessions(monkeypatch):
    transformers_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

    def create_masks(_config, _embeds, _attention, *, past_key_values, position_ids):
        del past_key_values, position_ids
        return object()

    monkeypatch.setattr(transformers_modeling, "create_masks_for_generate", create_masks)

    class _Native:
        def __init__(self):
            self.imports = []

        def reset(self):
            pass

        def import_cache_async(self, cache, *, current_position, prefix_len, prefix_mask):
            self.imports.append((cache, current_position, prefix_len, prefix_mask.clone()))

    class _FairLock:
        def __init__(self):
            self.owners = []

        def acquire_fair(self, purpose):
            self.owners.append((purpose, "acquire"))

        def release(self):
            self.owners.append(("listener", "release"))

    backend = object.__new__(GemmaAudioTowerBackend)
    backend._use_native_lm = True
    backend._shape_stable_lm = _ShapeStable()
    backend.gemma_model = SimpleNamespace(language_model=_LanguageModel())
    backend.device = "cpu"
    backend.hidden_size = 1536
    backend._frame_pad_ple = None
    backend.pad_token_id = 0
    backend.boa_token_id = 777
    backend.runtime_profile = "compatibility"
    backend._native_runtime = None
    backend._lock = _FairLock()

    state_a = GemmaListenerState(next_frame_id=4, gemma_lm_native=_Native())
    state_b = GemmaListenerState(next_frame_id=12, gemma_lm_native=_Native())
    result_a = GemmaAudioTowerBackend.prime_semantic_prefix(backend, state_a, (2, 3, 777))
    result_b = GemmaAudioTowerBackend.prime_semantic_prefix(backend, state_b, (2, 5, 6, 7, 777))

    assert backend._lock.owners == [
        ("listener", "acquire"), ("listener", "release"),
        ("listener", "acquire"), ("listener", "release"),
    ]
    cache_a, position_a, prefix_a, mask_a = state_a.gemma_lm_native.imports[0]
    cache_b, position_b, prefix_b, mask_b = state_b.gemma_lm_native.imports[0]
    assert cache_a is result_a["past_key_values"]
    assert cache_b is result_b["past_key_values"]
    assert (position_a, prefix_a, tuple(mask_a.shape)) == (3, 3, (1, 3))
    assert (position_b, prefix_b, tuple(mask_b.shape)) == (5, 5, (1, 5))
    assert state_a.gemma_lm_prefix_len == 3
    assert state_b.gemma_lm_prefix_len == 5
    assert state_a.gemma_lm_prefix_attention.shape == (1, 3)
    assert state_b.gemma_lm_prefix_attention.shape == (1, 5)
