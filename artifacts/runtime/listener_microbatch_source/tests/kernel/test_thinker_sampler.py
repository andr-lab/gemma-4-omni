from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.engine import PipelineEngine, ThinkerSampler


class _Tokenizer:
    eos_token_id = 4
    eot_id = 5

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(ord("a") + int(token_id)) for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) - ord("a") for char in text]


def _sampler(**kwargs) -> ThinkerSampler:
    options = {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    options.update(kwargs)
    return ThinkerSampler(
        tokenizer=_Tokenizer(),
        device="cpu",
        eos_token_ids={4, 5},
        **options,
    )


def test_sample_token_applies_repetition_penalty_only_to_supplied_generated_history():
    sampler = _sampler(repetition_penalty=2.0)
    logits = torch.tensor([[0.0, 2.0, 1.5, 0.0, 0.0, 0.0]])

    assert sampler.sample_token(logits, ())[1] == 1
    assert sampler.sample_token(logits, (1,))[1] == 2


def test_min_new_tokens_masks_eos_until_generated_history_reaches_threshold():
    sampler = _sampler(min_new_tokens=2)
    logits = torch.tensor([[0.0, 1.0, 0.0, 0.0, 3.0, 2.0]])

    assert sampler.sample_token(logits, ())[1] == 1
    assert sampler.sample_token(logits, (1,))[1] == 1
    assert sampler.sample_token(logits, (1, 1))[1] == 4


def test_backchannel_trie_restricts_candidates_and_updates_runtime_metric():
    metrics = {}
    sampler = _sampler(
        backchannel_trie={(): {1, 2}, (1,): {2}},
        runtime_metrics=metrics,
    )

    selected = sampler.sample_token(torch.tensor([[0.0, 0.1, 0.2, 9.0, 0.0, 0.0]]), ())
    assert selected[1] == 2
    assert metrics["backchannel_trie_steps"] == 1
    with pytest.raises(RuntimeError, match="non appartiene"):
        sampler.sample_token(torch.zeros((1, 6)), (3,))


def test_candidate_guard_selects_legal_top_candidate_and_uses_fallback_pool():
    metrics = {}
    sampler = _sampler(
        runtime_metrics=metrics,
        on_candidate_token=lambda token_id, _text, _terminal: token_id == 1,
        on_candidate_fallback=lambda _rejected: [1],
        best_native_logit_candidate_id=lambda candidates, logits: max(
            candidates, key=lambda candidate_id: float(logits.reshape(-1)[candidate_id])
        ),
    )
    logits = torch.full((1, 100), -10.0)
    logits[0, 0] = 10.0
    logits[0, 2:65] = 9.0
    logits[0, 1] = -100.0
    raw_candidate = sampler.sample_token(logits, ())

    selected = sampler.select_valid_candidate(
        raw_candidate,
        source_logits=logits,
        generated_ids=(),
    )

    assert raw_candidate[1] == 0
    assert selected[1] == 1
    assert metrics["audio_response_protocol_v2_candidate_checks"] >= 65
    assert metrics["audio_response_protocol_v2_lexicon_pool_overrides"] == 1


def test_shared_pipeline_preserves_initial_force_and_generated_forced_next_order():
    events = []
    sampler = _sampler(
        on_generation_start=lambda: events.append("generation_start") or 2,
        on_generated_token=lambda index, token_id, text, hidden: (
            events.append(("generated", index, token_id, text, tuple(hidden.shape))) or (3, "structure")
        ),
    )
    logits = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    initial = sampler.sample_initial(logits, ())
    assert initial[1] == 2
    assert events == ["generation_start"]

    raw_next = sampler.sample_token(logits, (2,))
    selected = sampler.after_generated_token(
        token_index=0,
        token_id=2,
        token_hidden=torch.ones((1, 1, 8)),
        generated_ids=(2,),
        source_logits=logits,
        sampled_next=raw_next,
    )
    assert selected[1] == 3
    assert events[-1] == ("generated", 0, 2, "c", (1, 1, 8))
    assert sampler.forced_structure_overrides == 2


def test_action_force_probes_same_source_logits_and_preserves_raw_sample_id():
    probes = []

    class ActionTokenizer(_Tokenizer):
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"SPEAK": [1], "BACKCHANNEL": [2], "REPLAN": [3]}[text]

    sampler = ThinkerSampler(
        tokenizer=ActionTokenizer(),
        device="cpu",
        eos_token_ids={4, 5},
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        on_generated_token=lambda *_args: (2, "action"),
        on_action_logits_probe=probes.append,
    )
    logits = torch.tensor([[0.0, 3.0, 2.0, 1.0, 0.0, 0.0]])
    raw_sample = sampler.sample_token(logits, (1,))

    selected = sampler.after_generated_token(
        token_index=0,
        token_id=1,
        token_hidden=torch.ones((1, 1, 8)),
        generated_ids=(1,),
        source_logits=logits,
        sampled_next=raw_sample,
    )

    assert raw_sample[1] == 1
    assert selected[1] == 2
    assert probes == [{
        "raw_argmax_token_id": 1,
        "sampled_token_id": 1,
        "forced_token_id": 2,
        "action_label_token_ids": {"SPEAK": [1], "BACKCHANNEL": [2], "REPLAN": [3]},
        "action_label_first_token_logits": {"SPEAK": 3.0, "BACKCHANNEL": 2.0, "REPLAN": 1.0},
        "raw_argmax_action_candidates": ["SPEAK"],
        "best_first_logit_action_candidates": ["SPEAK"],
    }]
    assert sampler.forced_action_overrides == 1


def test_engine_factory_resolves_eos_precedence_and_reuses_engine_constraints():
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.tokenizer = _Tokenizer()
    engine.device = "cpu"
    engine.thinker = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[7]),
        config=SimpleNamespace(eos_token_id=8),
    )
    sampler = engine.build_thinker_sampler(temperature=0.0, top_p=1.0, repetition_penalty=1.0)

    assert sampler.eos_token_ids == {7}
    assert sampler.sample_token(torch.tensor([[0.0, 9.0]]), ())[1] == 1


def test_thinker_sampler_uses_optional_private_generator_without_global_rng_mutation():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    generator_a = torch.Generator(device="cpu").manual_seed(12345)
    generator_b = torch.Generator(device="cpu").manual_seed(12345)
    sampler_a = _sampler(
        temperature=1.0,
        top_p=1.0,
        generator=generator_a,
    )
    sampler_b = _sampler(
        temperature=1.0,
        top_p=1.0,
        generator=generator_b,
    )
    global_state_before = torch.random.get_rng_state().clone()

    samples_a = [sampler_a.sample_token(logits, ())[1] for _ in range(12)]
    assert torch.equal(generator_b.get_state(), torch.Generator(device="cpu").manual_seed(12345).get_state())
    samples_b = [sampler_b.sample_token(logits, ())[1] for _ in range(12)]

    assert samples_a == samples_b
    assert torch.equal(generator_a.get_state(), generator_b.get_state())
    assert torch.equal(global_state_before, torch.random.get_rng_state())
