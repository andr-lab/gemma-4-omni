from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.engine import ThinkerSampler
from duplex_kernel.unified_native_decoder import UnifiedNativeDecoder


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 10
        return self.value


def _thinker() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[99]),
            config=SimpleNamespace(eos_token_id=98),
        ),
        tokenizer=SimpleNamespace(eos_token_id=97, eot_id=96),
    )


def test_decode_forwards_only_transition_suffix_and_samples_first_token_from_its_logits() -> None:
    forwarded: list[list[int]] = []
    sampled_from: list[int] = []
    emitted = []
    dispatch_count_at_emit = []
    first_logits_events = []
    first_tokens = []
    sampled_history = []
    sample_ids = iter((42, 43, 99))

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        hidden = torch.tensor([token_ids[-1]], dtype=torch.float32)
        logits = torch.tensor([[float(token_ids[-1]), 0.0, 1.0]])
        return logits, hidden

    async def sample_token(logits: torch.Tensor, generated_ids: tuple[int, ...]):
        sampled_from.append(int(logits[0, 0].item()))
        sampled_history.append(generated_ids)
        token_id = next(sample_ids)
        return torch.tensor([[token_id]]), token_id, logits

    async def on_token(token_id, hidden, timing) -> None:
        emitted.append((token_id, hidden, timing))
        dispatch_count_at_emit.append(len(forwarded))

    async def first_logits_ready(event) -> None:
        first_logits_events.append(event)

    async def first_token(token) -> None:
        first_tokens.append(token)

    decoder = UnifiedNativeDecoder(
        _thinker(),
        forward_tokens=forward_tokens,
        sample_token=sample_token,
        monotonic_ns=_Clock(),
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[10, 11, 12],
            max_new_tokens=5,
            first_logits_ready=first_logits_ready,
            first_token=first_token,
            on_token=on_token,
        )
    )

    assert forwarded == [[10, 11], [12], [42], [43]]
    assert sampled_from == [12, 42, 43]
    assert [token_id for token_id, _, _ in emitted] == [42, 43]
    assert dispatch_count_at_emit == [3, 4]
    assert sampled_history == [(), (42,), (42, 43)]
    assert first_tokens[0].token_id == emitted[0][0]
    assert len(first_logits_events) == 1
    assert int(first_logits_events[0].final_hidden.item()) == 12
    assert [int(hidden.item()) for _, hidden, _ in emitted] == [42, 43]
    assert result.termination_reason == "eos"
    assert result.terminal_token_id == 99
    assert result.generated_count == 2
    assert result.committed_count == 2
    assert result.sampled_count == 3
    assert result.transition_token_count == 3
    assert result.pending_token_id is None
    assert result.owner_action == "handle_terminal_eos"


def test_decode_does_not_commit_or_emit_eos_token() -> None:
    forwarded: list[list[int]] = []
    emitted = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([token_ids[-1]])

    async def sample_token(_logits: torch.Tensor, _history: tuple[int, ...]) -> int:
        return 99

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=sample_token, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(transition_token_ids=[7], max_new_tokens=3, on_token=emitted.append)
    )

    assert forwarded == [[7]]
    assert emitted == []
    assert result.termination_reason == "eos"
    assert result.generated_count == 0
    assert result.committed_count == 0
    assert result.sampled_count == 1


def test_decode_checks_cancellation_before_native_dispatch() -> None:
    forwarded: list[list[int]] = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.zeros((1,))

    def sample_token(_logits: torch.Tensor, _history: tuple[int, ...]) -> int:
        return 1

    async def stop_check() -> bool:
        return True

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=sample_token, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(transition_token_ids=[20, 21], max_new_tokens=4, stop_check=stop_check)
    )

    assert forwarded == []
    assert result.termination_reason == "cancelled"
    assert result.generated_count == 0
    assert result.committed_count == 0
    assert result.sampled_count == 0
    assert result.pending_token_id is None
    assert result.owner_action == "rollback_assistant_branch"


def test_budget_exhaustion_streams_last_token_and_marks_it_pending_for_owner() -> None:
    forwarded: list[list[int]] = []
    emitted = []
    dispatch_count_at_emit = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([token_ids[-1]])

    def sample_token(_logits: torch.Tensor, _history: tuple[int, ...]) -> int:
        return 42

    def on_token(token_id, hidden, timing) -> None:
        emitted.append((token_id, hidden, timing))
        dispatch_count_at_emit.append(len(forwarded))

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=sample_token, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(transition_token_ids=[7], max_new_tokens=1, on_token=on_token)
    )

    assert forwarded == [[7], [42]]
    assert [token_id for token_id, _, _ in emitted] == [42]
    assert dispatch_count_at_emit == [2]
    assert result.generated_count == 1
    assert result.committed_count == 1
    assert result.pending_token_id is None
    assert result.owner_action == "append_turn_eos"
    assert result.termination_reason == "budget_exhausted"


def test_cancellation_after_streaming_leaves_pending_token_and_requires_branch_rollback() -> None:
    forwarded: list[list[int]] = []
    emitted = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([token_ids[-1]])

    def sample_token(_logits: torch.Tensor, _history: tuple[int, ...]) -> int:
        return 42

    async def on_token(token_id, hidden, timing) -> None:
        emitted.append(token_id)

    async def stop_check() -> bool:
        return len(emitted) > 0

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=sample_token, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[7], max_new_tokens=3, stop_check=stop_check, on_token=on_token
        )
    )

    assert forwarded == [[7], [42]]
    assert emitted == [42]
    assert result.termination_reason == "cancelled"
    assert result.pending_token_id == 42
    assert result.committed_count == 1
    assert result.owner_action == "rollback_assistant_branch"


def test_shared_sampler_orders_raw_next_force_publish_then_next_dispatch():
    events = []

    class Tokenizer:
        eos_token_id = 99
        eot_id = None

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    sampler = ThinkerSampler(
        tokenizer=Tokenizer(),
        device="cpu",
        eos_token_ids={99},
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        on_generated_token=lambda index, token_id, text, hidden: (
            events.append(("generated", index, token_id, text, int(hidden.item())))
            or ((3, "structure") if token_id == 1 else None)
        ),
    )
    original_sample = sampler.sample_token

    def sample_token(logits, history):
        candidate = original_sample(logits, history)
        events.append(("sample", tuple(history), candidate[1]))
        return candidate

    sampler.sample_token = sample_token

    async def forward_tokens(token_ids):
        events.append(("forward", tuple(token_ids)))
        logits = torch.zeros((1, 100))
        if token_ids[-1] == 10:
            logits[0, 1] = 10.0
        elif token_ids[-1] == 1:
            logits[0, 2] = 10.0
        else:
            logits[0, 99] = 10.0
        return logits, torch.tensor([float(token_ids[-1])])

    def on_token(token_id, _hidden, _timing):
        events.append(("publish", token_id))

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sampler=sampler, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[10],
            max_new_tokens=5,
            on_token=on_token,
        )
    )

    assert events == [
        ("forward", (10,)),
        ("sample", (), 1),
        ("forward", (1,)),
        ("sample", (1,), 2),
        ("generated", 0, 1, "<1>", 1),
        ("publish", 1),
        ("forward", (3,)),
        ("sample", (1, 3), 99),
        ("generated", 1, 3, "<3>", 3),
        ("publish", 3),
    ]
    assert result.termination_reason == "eos"
    assert result.generated_count == result.committed_count == 2
    assert result.sampled_count == 3


def test_initial_forced_eos_still_obeys_thinker_min_new_tokens_contract():
    class Tokenizer:
        eos_token_id = 99
        eot_id = None

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    sampler = ThinkerSampler(
        tokenizer=Tokenizer(),
        device="cpu",
        eos_token_ids={99},
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=1,
        on_generation_start=lambda: 99,
    )

    async def forward_tokens(_token_ids):
        logits = torch.zeros((1, 100))
        logits[0, 99] = 10.0
        return logits, torch.zeros((1,))

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sampler=sampler, monotonic_ns=_Clock()
    )
    with pytest.raises(RuntimeError, match="ha restituito EOS prima della soglia"):
        asyncio.run(
            decoder.decode(transition_token_ids=[10], max_new_tokens=3)
        )


def test_cancellation_between_sampling_and_dispatch_reports_uncommitted_sample() -> None:
    forwarded: list[list[int]] = []
    cancelled = False
    emitted = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([token_ids[-1]])

    def sample_token(_logits: torch.Tensor, _history: tuple[int, ...]) -> int:
        nonlocal cancelled
        cancelled = True
        return 42

    def stop_check() -> bool:
        return cancelled

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=sample_token, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[7], max_new_tokens=3, stop_check=stop_check, on_token=lambda *args: emitted.append(args)
        )
    )

    assert forwarded == [[7]]
    assert emitted == []
    assert result.termination_reason == "cancelled"
    assert result.generated_count == 0
    assert result.committed_count == 0
    assert result.pending_token_id == 42
    assert result.owner_action == "rollback_assistant_branch"


def test_decoder_fork_uses_private_sampler_callbacks_history_and_rng():
    source_generator = torch.Generator(device="cpu").manual_seed(83)
    source_callback = lambda *_args: None

    class Sampler:
        def __init__(self, generator, callback):
            self.generator = generator
            self.on_generated_token_callback = callback
            self.on_candidate_token = callback
            self.state = object()
            self.runtime_metrics = {"count": 4, "nested": {"kept": True}}
            self.last_decoded_output = "already"
            self.forced_action_logprobs = [torch.tensor([1.0])]
            self.forced_action_overrides = 2
            self.forced_structure_logprobs = [torch.tensor([2.0])]
            self.forced_structure_overrides = 3
            self.action_logits_probe_emitted = True
            self.backchannel_trie = {(1,): {2}}
            self.temperature = 1.0

        def sample_token(self, logits, history):
            return torch.tensor([[1]]), 1, logits

    source_sampler = Sampler(source_generator, source_callback)
    source_rng_before = source_generator.get_state().clone()

    async def original_forward(_token_ids):
        return torch.zeros((1, 2)), torch.zeros((1,))

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=original_forward, sampler=source_sampler,
        monotonic_ns=_Clock(),
    )
    new_callbacks = lambda *_args: None
    forked_sampler_box = {}

    def sampler_factory(source, fork_generator, fork_id):
        assert source is source_sampler
        assert fork_id == "backchannel-7"
        forked = Sampler(fork_generator, new_callbacks)
        forked.state = object()
        forked.runtime_metrics = {}
        forked_sampler_box["value"] = forked
        return forked

    async def fork_forward(_token_ids):
        return torch.ones((1, 2)), torch.ones((1,))

    forked = decoder.fork(
        forward_tokens=fork_forward,
        sampler_factory=sampler_factory,
        fork_id="backchannel-7",
    )
    branch_sampler = forked_sampler_box["value"]

    assert forked.thinker is decoder.thinker
    assert forked.forward_tokens is fork_forward
    assert forked.sampler is branch_sampler and forked.sampler is not source_sampler
    assert forked.sample_token.__self__ is branch_sampler
    assert branch_sampler.generator is not source_generator
    assert torch.equal(source_generator.get_state(), source_rng_before)
    assert branch_sampler.on_generated_token_callback is new_callbacks
    assert branch_sampler.on_candidate_token is new_callbacks
    assert branch_sampler.state is not source_sampler.state
    assert branch_sampler.runtime_metrics == source_sampler.runtime_metrics
    assert branch_sampler.runtime_metrics is not source_sampler.runtime_metrics
    assert branch_sampler.last_decoded_output == source_sampler.last_decoded_output
    assert branch_sampler.forced_action_logprobs[0] is not source_sampler.forced_action_logprobs[0]
    assert branch_sampler.backchannel_trie == source_sampler.backchannel_trie
    assert branch_sampler.backchannel_trie is not source_sampler.backchannel_trie
    fork_rng_before = branch_sampler.generator.get_state().clone()
    torch.rand(1, generator=branch_sampler.generator)
    assert torch.equal(source_generator.get_state(), source_rng_before)
    assert not torch.equal(branch_sampler.generator.get_state(), fork_rng_before)


def test_decoder_fork_rejects_stochastic_sampler_without_private_rng():
    class StochasticSampler:
        temperature = 1.0
        generator = None

        def sample_token(self, logits, _history):
            return 1

    async def forward(_token_ids):
        return torch.zeros((1, 2)), torch.zeros((1,))

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward, sampler=StochasticSampler(),
        monotonic_ns=_Clock(),
    )
    with pytest.raises(RuntimeError, match="private generator"):
        decoder.fork(
            forward_tokens=forward,
            sampler_factory=lambda source, generator, _fork_id: StochasticSampler(),
            fork_id="branch-no-rng",
        )
