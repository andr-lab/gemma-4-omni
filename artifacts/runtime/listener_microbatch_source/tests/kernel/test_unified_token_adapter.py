from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.unified_token_adapter import UnifiedTokenAdapter


class _FakeEmbedding:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.bfloat16)[..., None].expand(*token_ids.shape, 1536).contiguous()


class _FakeTextModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_hidden_layers=35,
            hidden_size=1536,
            hidden_size_per_layer_input=256,
            pad_token_id=17,
            vocab_size=32768,
        )
        self.embed_tokens = _FakeEmbedding()
        self.calls: list[tuple[str, tuple[torch.Tensor, ...]]] = []
        self.norm_calls = 0
        self.norm = self._norm

    def _norm(self, hidden: torch.Tensor) -> torch.Tensor:
        self.norm_calls += 1
        return hidden

    def get_per_layer_inputs(
        self, token_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append(("identity", (token_ids.clone(), inputs_embeds.clone())))
        return token_ids.to(torch.bfloat16)[..., None, None].expand(*token_ids.shape, 35, 256).contiguous()

    def project_per_layer_inputs(
        self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append(("project", (inputs_embeds.clone(), per_layer_inputs.clone())))
        context = inputs_embeds[..., :1, None].expand(*per_layer_inputs.shape)
        return (context + per_layer_inputs).contiguous()


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [31, 32, 33]


def _fake_thinker() -> tuple[SimpleNamespace, _FakeTextModel]:
    language_model = _FakeTextModel()
    thinker = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(language_model=language_model)),
        tokenizer=_FakeTokenizer(),
    )
    return thinker, language_model


def test_prepare_projects_exact_runtime_token_identity_for_q1_and_q2() -> None:
    thinker, language_model = _fake_thinker()
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)

    q1 = adapter.prepare(torch.tensor([[5], [8]], dtype=torch.long))
    q2 = adapter.prepare(torch.tensor([[6, 7]], dtype=torch.long))

    assert tuple(q1.inputs_embeds.shape) == (2, 2, 1536)
    assert tuple(q1.per_layer_inputs.shape) == (2, 2, 35, 256)
    assert q1.logical_q_lens == (1, 1)
    assert q1.padded_input_ids.tolist() == [[5, 17], [8, 17]]
    assert q2.logical_q_lens == (2,)
    assert q2.padded_input_ids.tolist() == [[6, 7]]

    for result in (q1, q2):
        ids = result.padded_input_ids
        embeds = language_model.embed_tokens(ids)
        identity = language_model.get_per_layer_inputs(ids, embeds)
        expected = language_model.project_per_layer_inputs(embeds, identity)
        torch.testing.assert_close(result.inputs_embeds, embeds, rtol=0, atol=0)
        torch.testing.assert_close(result.per_layer_inputs, expected, rtol=0, atol=0)
        assert result.inputs_embeds.dtype == torch.bfloat16
        assert result.per_layer_inputs.dtype == torch.bfloat16
        assert result.inputs_embeds.is_contiguous()
        assert result.per_layer_inputs.is_contiguous()

    assert language_model.norm_calls == 0
    assert [name for name, _ in language_model.calls[:2]] == ["identity", "project"]


@pytest.mark.parametrize("logical_q", [3, 5, 6, 8])
def test_prepare_variable_text_q_uses_even_physical_width(logical_q: int) -> None:
    thinker, _ = _fake_thinker()
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)
    prepared = adapter.prepare(tuple(range(1, logical_q + 1)))
    physical_q = logical_q + (logical_q & 1)
    physical_q = max(2, physical_q)

    assert tuple(prepared.inputs_embeds.shape) == (1, physical_q, 1536)
    assert tuple(prepared.per_layer_inputs.shape) == (1, physical_q, 35, 256)
    assert prepared.logical_q_lens == (logical_q,)
    assert prepared.padded_input_ids[0, :logical_q].tolist() == list(range(1, logical_q + 1))
    if physical_q > logical_q:
        assert prepared.padded_input_ids[0, logical_q].item() == adapter.pad_token_id


def test_prepare_mixed_logical_lengths_overwrites_only_q1_second_slot() -> None:
    thinker, _ = _fake_thinker()
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)

    prepared = adapter.prepare(
        torch.tensor([[1, 2], [3, 99]], dtype=torch.long),
        logical_q_lens=(2, 1),
    )

    assert prepared.logical_q_lens == (2, 1)
    assert prepared.padded_input_ids.tolist() == [[1, 2], [3, 17]]


def test_stable_protocol_preparation_reuses_exact_cached_tensors() -> None:
    thinker, language_model = _fake_thinker()
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)
    transition = (41, 42, 43, 44, 45)

    first = adapter.prepare(
        transition,
        logical_q_lens=len(transition),
        protocol="controller_guided_say_v1",
    )
    calls_after_first = len(language_model.calls)
    second = adapter.prepare(
        transition,
        logical_q_lens=len(transition),
        protocol="controller_guided_say_v1",
    )

    assert second is first
    assert len(language_model.calls) == calls_after_first
    assert second.padded_input_ids[0, : len(transition)].tolist() == list(transition)
    # The protocol is part of the key; another protocol cannot reuse the entry.
    other_protocol = adapter.prepare(
        transition,
        logical_q_lens=len(transition),
        protocol="heard_action_say_v2",
    )
    assert other_protocol is not first
    assert len(language_model.calls) > calls_after_first


def test_prepare_sequence_chunks_chat_template_ids_into_physical_q2_steps() -> None:
    thinker, _ = _fake_thinker()
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hello"}]

    encoded = adapter.encode_chat(messages, add_generation_prompt=True)
    prepared = list(adapter.prepare_sequence(encoded))

    assert thinker.tokenizer.calls == [(messages, {"tokenize": True, "add_generation_prompt": True})]
    assert [step.padded_input_ids.tolist() for step in prepared] == [[[31, 32]], [[33, 17]]]
    assert [step.logical_q_lens for step in prepared] == [(2,), (1,)]


def test_logits_helper_applies_only_the_lm_head_to_native_final_norm_hidden() -> None:
    thinker, language_model = _fake_thinker()
    thinker.model.lm_head = lambda hidden: hidden[..., :4] * 3
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)
    hidden = torch.ones((1, 2, 1536), dtype=torch.bfloat16)

    logits = adapter.logits_from_final_norm_hidden(hidden)

    assert tuple(logits.shape) == (1, 2, 4)
    assert torch.equal(logits, torch.full((1, 2, 4), 3.0, dtype=torch.bfloat16))
    assert language_model.norm_calls == 0


def test_logits_helper_matches_model_forward_final_logit_softcap_dtype_and_order() -> None:
    thinker, language_model = _fake_thinker()
    thinker.config = SimpleNamespace(
        get_text_config=lambda: SimpleNamespace(final_logit_softcapping=2.0)
    )
    thinker.model.lm_head = lambda hidden: hidden[..., :4] * 3
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)
    hidden = torch.full((1, 1, 1536), 1.0, dtype=torch.bfloat16)

    logits = adapter.logits_from_final_norm_hidden(hidden)

    raw_logits = hidden[..., :4] * 3
    expected = torch.tanh(raw_logits / 2.0) * 2.0
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert logits.dtype == raw_logits.dtype
    assert language_model.norm_calls == 0


@pytest.mark.parametrize("input_ids", (torch.empty((0,), dtype=torch.long), torch.ones((1, 9), dtype=torch.long)))
def test_prepare_rejects_empty_or_over_limit_shapes(input_ids: torch.Tensor) -> None:
    thinker, _ = _fake_thinker()
    adapter = UnifiedTokenAdapter(thinker, require_cuda=False)

    with pytest.raises(ValueError):
        adapter.prepare(input_ids)
