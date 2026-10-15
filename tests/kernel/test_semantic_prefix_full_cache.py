from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace

from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


class _RecordingShapeStableGemmaListenerLM(ShapeStableGemmaListenerLM):
    """Record real ShapeStable layer outputs without replacing its math."""

    def __init__(self, language_model, *, target_length: int, stream_length: int = 4) -> None:
        super().__init__(language_model, target_length=target_length, stream_length=stream_length)
        self.calls: list[dict[str, object]] = []
        self._pending_layer_outputs: list[torch.Tensor] = []
        self._pending_ple_projection_inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def _decoder_layer(self, *args, **kwargs):
        output = super()._decoder_layer(*args, **kwargs)
        self._pending_layer_outputs.append(output.detach().clone())
        return output

    def _project_per_layer_inputs(self, inputs_embeds, token_identity):
        self._pending_ple_projection_inputs.append(
            (
                inputs_embeds.detach().clone(),
                None if token_identity is None else token_identity.detach().clone(),
            )
        )
        return super()._project_per_layer_inputs(inputs_embeds, token_identity)

    def forward(self, *, inputs_embeds, per_layer_inputs, attention_mask, position_ids,
                past_key_values, shared_kv_states):
        self._pending_layer_outputs = []
        self._pending_ple_projection_inputs = []
        output = super().forward(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states,
        )
        self.calls.append(
            {
                "q_len": int(inputs_embeds.shape[1]),
                "inputs_embeds": inputs_embeds.detach().clone(),
                "per_layer_inputs": (
                    None if per_layer_inputs is None else per_layer_inputs.detach().clone()
                ),
                "position_ids": position_ids.detach().clone(),
                "attention_mask": {
                    key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in attention_mask.items()
                },
                "hidden": output.last_hidden_state.detach().clone(),
                "layer_outputs": tuple(self._pending_layer_outputs),
                "ple_projection_inputs": tuple(self._pending_ple_projection_inputs),
            }
        )
        return output


def _tiny_gemma4_text_model(*, per_layer_input_size: int = 8):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    config = Gemma4TextConfig(
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_global_key_value_heads=2,
        global_head_dim=8,
        sliding_window=512,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=per_layer_input_size,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    config._attn_implementation = "sdpa"
    language_model = Gemma4TextModel(config).eval()
    for parameter in language_model.parameters():
        parameter.requires_grad_(False)
    return config, language_model


def _run_sequential_gemma4_replay(language_model, prefix_token_ids, replay_q4):
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    device = torch.device("cpu")
    stable = _RecordingShapeStableGemmaListenerLM(
        language_model, target_length=291, stream_length=4
    )
    prefix_ids = torch.tensor(prefix_token_ids, dtype=torch.long, device=device).view(1, -1)
    prefix_embeddings = language_model.embed_tokens(prefix_ids)
    prefix_len = int(prefix_ids.shape[1])
    prefix_positions = torch.arange(prefix_len, dtype=torch.long, device=device).view(1, -1)
    prefix_masks = create_masks_for_generate(
        language_model.config,
        prefix_embeddings,
        torch.ones((1, prefix_len), dtype=torch.long, device=device),
        past_key_values=None,
        position_ids=prefix_positions,
    )
    with torch.inference_mode():
        output = stable.forward(
            inputs_embeds=prefix_embeddings,
            per_layer_inputs=language_model.get_per_layer_inputs(prefix_ids, prefix_embeddings),
            attention_mask=prefix_masks,
            position_ids=prefix_positions,
            past_key_values=DynamicCache(),
            shared_kv_states=None,
        )

    cache = output.past_key_values
    position = prefix_len
    for q4 in replay_q4:
        frame_audio = q4.reshape(1, 4, language_model.config.hidden_size).to(
            dtype=language_model.embed_tokens.weight.dtype
        )
        positions = torch.arange(position, position + 4, dtype=torch.long, device=device).view(1, -1)
        masks = create_masks_for_generate(
            language_model.config,
            frame_audio,
            torch.ones((1, position + 4), dtype=torch.long, device=device),
            past_key_values=cache,
            position_ids=positions,
        )
        with torch.inference_mode():
            output = stable.forward(
                inputs_embeds=frame_audio,
                per_layer_inputs=language_model.get_per_layer_inputs(
                    torch.full((1, 4), language_model.config.pad_token_id, dtype=torch.long),
                    language_model.embed_tokens(
                        torch.full((1, 4), language_model.config.pad_token_id, dtype=torch.long)
                    ),
                ),
                attention_mask=masks,
                position_ids=positions,
                past_key_values=cache,
                shared_kv_states=getattr(cache, "shared_layers", None),
            )
        cache = output.past_key_values
        position += 4
    return stable, cache


def _make_semantic_prefix_backend(language_model, stable):
    from duplex_kernel.listener import GemmaAudioTowerBackend

    backend = object.__new__(GemmaAudioTowerBackend)
    backend._shape_stable_lm = stable
    backend.gemma_model = SimpleNamespace(language_model=language_model)
    backend.device = torch.device("cpu")
    backend.hidden_size = int(language_model.config.hidden_size)
    backend.pad_token_id = int(language_model.config.pad_token_id)
    backend._frame_pad_ple = None
    return backend


def _max_abs_and_relative(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-8)
    return float(difference.max()), float(relative.max())


def _run_prefix_and_q4_replay(language_model, cache, prefix_ids, replay_embeddings):
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    stable = ShapeStableGemmaListenerLM(language_model, target_length=291, stream_length=4)
    prefix_embeddings = language_model.embed_tokens(prefix_ids)
    prefix_len = int(prefix_ids.shape[1])
    position_ids = torch.arange(prefix_len, dtype=torch.long).view(1, -1)
    prefix_masks = create_masks_for_generate(
        language_model.config,
        prefix_embeddings,
        torch.ones((1, prefix_len), dtype=torch.long),
        past_key_values=None,
        position_ids=position_ids,
    )
    output = stable.forward(
        inputs_embeds=prefix_embeddings,
        per_layer_inputs=None,
        attention_mask=prefix_masks,
        position_ids=position_ids,
        past_key_values=cache,
        shared_kv_states=None,
    )
    hidden = [output.last_hidden_state.detach().clone()]

    position = prefix_len
    for frame_embeddings in replay_embeddings:
        q_len = int(frame_embeddings.shape[1])
        positions = torch.arange(position, position + q_len, dtype=torch.long).view(1, -1)
        masks = create_masks_for_generate(
            language_model.config,
            frame_embeddings,
            torch.ones((1, position + q_len), dtype=torch.long),
            past_key_values=cache,
            position_ids=positions,
        )
        output = stable.forward(
            inputs_embeds=frame_embeddings,
            per_layer_inputs=None,
            attention_mask=masks,
            position_ids=positions,
            past_key_values=cache,
            shared_kv_states=getattr(cache, "shared_layers", None),
        )
        hidden.append(output.last_hidden_state.detach().clone())
        position += q_len
    return torch.cat(hidden, dim=1), cache


def test_prefix_full_cache_preserves_sliding_mask_and_matches_sliding_cache() -> None:
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(14)
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_global_key_value_heads=2,
        global_head_dim=8,
        sliding_window=512,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    config._attn_implementation = "sdpa"
    language_model = Gemma4TextModel(config).eval()
    prefix_ids = torch.arange(291, dtype=torch.long).remainder(config.vocab_size).view(1, -1)
    replay_embeddings = [torch.randn((1, 4, config.hidden_size)) for _ in range(58)]

    full_hidden, full_cache = _run_prefix_and_q4_replay(
        language_model, DynamicCache(), prefix_ids, replay_embeddings
    )
    sliding_cache = DynamicCache()
    sliding_cache.layers = [DynamicSlidingWindowLayer(sliding_window=512), DynamicLayer()]
    sliding_hidden, sliding_cache = _run_prefix_and_q4_replay(
        language_model, sliding_cache, prefix_ids, replay_embeddings
    )

    assert config.sliding_window == 512
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert int(full_cache.get_seq_length(0)) == 523
    assert [int(layer.keys.shape[-2]) for layer in full_cache.layers] == [523, 523]
    assert int(sliding_cache.get_seq_length(0)) == 523
    assert [int(layer.keys.shape[-2]) for layer in sliding_cache.layers] == [511, 523]
    hidden_diff = (full_hidden - sliding_hidden).abs()
    hidden_max_abs = float(hidden_diff.max())
    hidden_max_rel = float((hidden_diff / sliding_hidden.abs().clamp_min(1e-8)).max())
    torch.testing.assert_close(full_hidden, sliding_hidden, rtol=2e-3, atol=2e-3)
    logit_projection = torch.nn.Linear(config.hidden_size, 11, bias=False)
    full_logits = logit_projection(full_hidden)
    sliding_logits = logit_projection(sliding_hidden)
    logits_diff = (full_logits - sliding_logits).abs()
    logits_max_abs = float(logits_diff.max())
    logits_max_rel = float((logits_diff / sliding_logits.abs().clamp_min(1e-8)).max())
    print(
        "prefix_cache_parity_metrics="
        f"dtype:{full_hidden.dtype},hidden_max_abs:{hidden_max_abs:.9g},"
        f"hidden_max_rel:{hidden_max_rel:.9g},logits_max_abs:{logits_max_abs:.9g},"
        f"logits_max_rel:{logits_max_rel:.9g}"
    )
    torch.testing.assert_close(
        full_logits, sliding_logits, rtol=2e-3, atol=2e-3
    )

    for full_layer, sliding_layer in zip(full_cache.layers, sliding_cache.layers, strict=True):
        torch.testing.assert_close(full_layer.keys[..., -sliding_layer.keys.shape[-2] :, :], sliding_layer.keys)
        torch.testing.assert_close(full_layer.values[..., -sliding_layer.values.shape[-2] :, :], sliding_layer.values)


def test_semantic_prefix_builder_explicitly_uses_full_physical_cache() -> None:
    from transformers.cache_utils import DynamicLayer
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from duplex_kernel.listener import GemmaAudioTowerBackend

    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_global_key_value_heads=2,
        global_head_dim=8,
        sliding_window=512,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    config._attn_implementation = "sdpa"

    class _LanguageModel:
        def __init__(self) -> None:
            self.config = config
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        def get_per_layer_inputs(self, _ids, _embeddings):
            return None

    language_model = _LanguageModel()

    class _ShapeStable:
        def __init__(self):
            self.query_lengths = []

        def forward(self, *, inputs_embeds, past_key_values, **_kwargs):
            assert past_key_values is not None, "prefix builder must choose its physical cache explicitly"
            q_len = int(inputs_embeds.shape[1])
            self.query_lengths.append(q_len)
            key = torch.zeros((1, config.num_key_value_heads, q_len, config.head_dim))
            for layer_idx in range(config.num_hidden_layers):
                past_key_values.update(key, key, layer_idx)
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=past_key_values)

    backend = object.__new__(GemmaAudioTowerBackend)
    stable = _ShapeStable()
    backend._shape_stable_lm = stable
    backend.gemma_model = SimpleNamespace(language_model=language_model)
    backend.device = torch.device("cpu")
    backend.hidden_size = config.hidden_size
    backend._frame_pad_per_layer_inputs = lambda _lm: None

    prefix_ids = tuple(i % config.vocab_size for i in range(291))
    replay_ids = tuple(range(1, 59))
    replay = tuple(torch.randn((4, config.hidden_size)) for _ in replay_ids)
    built = backend.build_semantic_prefix_cache(
        prefix_ids, replay_q4=replay, replay_frame_ids=replay_ids
    )

    assert config.sliding_window == 512
    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert all(isinstance(layer, DynamicLayer) for layer in built["past_key_values"].layers)
    assert int(built["past_key_values"].get_seq_length(0)) == 523
    assert built["cache_position"] == 523
    assert [int(layer.keys.shape[-2]) for layer in built["past_key_values"].layers] == [523, 523]
    assert stable.query_lengths == [291] + [4] * len(replay_ids)
