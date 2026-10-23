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

    assert forwarded == [[10, 11, 12], [42], [43]]
    assert sampled_from == [12, 42, 43]
    assert [token_id for token_id, _, _ in emitted] == [42, 43]
    assert dispatch_count_at_emit == [2, 3]
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


def test_anticipated_first_token_uses_transition_hidden_and_publishes_before_own_forward() -> None:
    events = []
    forwarded: list[list[int]] = []
    sample_ids = iter((42, 43, 99))
    committed = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        events.append(("forward", tuple(token_ids)))
        hidden = torch.tensor([1000 + token_ids[-1]], dtype=torch.float32)
        logits = torch.zeros((1, 100))
        logits[0, 1 if token_ids[-1] == 7 else 2 if token_ids[-1] == 42 else 99] = 10.0
        return logits, hidden

    def sample_token(_logits: torch.Tensor, _history: tuple[int, ...]) -> int:
        token_id = next(sample_ids)
        events.append(("sample", token_id))
        return token_id

    def on_token_packet(packet) -> None:
        events.append((
            "publish", packet.token_index, packet.token_id, packet.hidden_provenance,
            packet.kv_committed, int(packet.hidden.item()),
        ))

    def on_token_committed(packet) -> None:
        committed.append(packet)
        events.append((
            "commit", packet.token_index, packet.token_id, packet.hidden_provenance,
            packet.kv_committed, int(packet.hidden.item()),
        ))

    def first_token_ready(packet) -> None:
        events.append(("first_ready", packet.token_id, packet.kv_committed, packet.packet_published_ns))

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=sample_token, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[7],
            max_new_tokens=3,
            anticipate_first_token=True,
            on_token_packet=on_token_packet,
            on_token_committed=on_token_committed,
            first_token=first_token_ready,
        )
    )

    assert events.index(("publish", 0, 42, "transition", False, 1007)) < events.index(("forward", (42,)))
    first_ready = next(event for event in events if event[0] == "first_ready")
    assert first_ready[2] is False and isinstance(first_ready[3], int)
    assert events.index(first_ready) < events.index(("forward", (42,)))
    assert events.index(("forward", (42,))) < events.index(("commit", 0, 42, "transition", True, 1007))
    assert events.index(("commit", 0, 42, "transition", True, 1007)) < events.index(("sample", 43))
    assert ("publish", 1, 43, "token", True, 1043) in events
    assert events.index(("forward", (43,))) < events.index(("publish", 1, 43, "token", True, 1043))
    assert [token.hidden_provenance for token in result.generated_tokens] == [
        "transition",
        "token",
    ]
    assert [token.kv_committed for token in result.generated_tokens] == [True, True]
    assert [int(token.hidden.item()) for token in result.generated_tokens] == [1007, 1043]
    assert result.termination_reason == "eos"
    assert result.committed_count == 2
    assert [(token.token_index, token.hidden_provenance) for token in committed] == [(0, "transition")]


def test_anticipated_first_token_cancellation_after_publish_does_not_forward_it() -> None:
    forwarded: list[list[int]] = []
    published = []
    committed = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([100 + token_ids[-1]])

    def on_token_packet(packet) -> None:
        published.append(packet.token_id)

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=lambda *_args: 42, monotonic_ns=_Clock()
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[7],
            max_new_tokens=3,
            anticipate_first_token=True,
            stop_check=lambda: bool(published),
            on_token_packet=on_token_packet,
            on_token_committed=committed.append,
        )
    )

    assert published == [42]
    assert forwarded == [[7]]
    assert result.termination_reason == "cancelled"
    assert result.pending_token_id == 42
    assert result.committed_count == 0
    assert result.generated_tokens[0].kv_committed is False
    assert result.generated_tokens[0].hidden_provenance == "transition"
    assert result.generated_tokens[0].packet_published_ns is not None
    assert committed == []
    assert result.owner_action == "rollback_assistant_branch"


def test_anticipated_first_token_callback_error_prevents_its_native_forward() -> None:
    forwarded: list[list[int]] = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([token_ids[-1]])

    def fail_publish(_packet) -> None:
        raise RuntimeError("packet publish failed")

    decoder = UnifiedNativeDecoder(
        _thinker(), forward_tokens=forward_tokens, sample_token=lambda *_args: 42, monotonic_ns=_Clock()
    )
    with pytest.raises(RuntimeError, match="packet publish failed"):
        asyncio.run(
            decoder.decode(
                transition_token_ids=[7],
                max_new_tokens=2,
                anticipate_first_token=True,
                on_token_packet=fail_publish,
            )
        )

    assert forwarded == [[7]]


def test_anticipated_first_token_late_commit_is_identified_for_stale_generation_filtering() -> None:
    async def scenario():
        token_forward_started = asyncio.Event()
        finish_token_forward = asyncio.Event()
        stale_generation = False
        committed_callbacks = []
        accepted_commits = []

        async def forward_tokens(token_ids: list[int]):
            if token_ids == [42]:
                token_forward_started.set()
                await finish_token_forward.wait()
            logits = torch.zeros((1, 100))
            logits[0, 99 if token_ids[-1] == 42 else 1] = 10.0
            return logits, torch.tensor([100 + token_ids[-1]])

        def on_token_committed(packet) -> None:
            committed_callbacks.append(packet)
            if not stale_generation:
                accepted_commits.append(packet.token_index)

        decoder = UnifiedNativeDecoder(
            _thinker(),
            forward_tokens=forward_tokens,
            sample_token=lambda *_args: 42,
            monotonic_ns=_Clock(),
        )
        decode_task = asyncio.create_task(
            decoder.decode(
                transition_token_ids=[7],
                max_new_tokens=3,
                anticipate_first_token=True,
                on_token_packet=lambda _packet: None,
                on_token_committed=on_token_committed,
                stop_check=lambda: stale_generation,
            )
        )
        await token_forward_started.wait()
        stale_generation = True
        finish_token_forward.set()
        result = await decode_task
        return result, committed_callbacks, accepted_commits

    result, committed_callbacks, accepted_commits = asyncio.run(scenario())

    assert result.termination_reason == "cancelled"
    assert result.generated_tokens[0].kv_committed is True
    assert result.committed_count == 1
    assert result.pending_token_id == 42
    assert [(packet.token_index, packet.kv_committed) for packet in committed_callbacks] == [(0, True)]
    assert accepted_commits == []


def test_anticipated_first_token_respects_one_token_budget_and_eos_has_no_packet() -> None:
    forwarded: list[list[int]] = []
    published = []
    committed = []
    sample_ids = iter((42, 99))

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.zeros((1, 3)), torch.tensor([100 + token_ids[-1]])

    decoder = UnifiedNativeDecoder(
        _thinker(),
        forward_tokens=forward_tokens,
        sample_token=lambda *_args: next(sample_ids),
        monotonic_ns=_Clock(),
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[7],
            max_new_tokens=1,
            anticipate_first_token=True,
            on_token_packet=published.append,
            on_token_committed=committed.append,
        )
    )

    assert [packet.token_id for packet in published] == [42]
    assert forwarded == [[7], [42]]
    assert result.generated_count == result.committed_count == 1
    assert result.termination_reason == "budget_exhausted"
    assert len(committed) == 1
    assert committed[0].kv_committed is True
    assert committed[0].packet_published_ns == result.generated_tokens[0].packet_published_ns

    eos_published = []
    eos_decoder = UnifiedNativeDecoder(
        _thinker(),
        forward_tokens=forward_tokens,
        sample_token=lambda *_args: 99,
        monotonic_ns=_Clock(),
    )
    eos_result = asyncio.run(
        eos_decoder.decode(
            transition_token_ids=[8],
            max_new_tokens=2,
            anticipate_first_token=True,
            on_token_packet=eos_published.append,
        )
    )
    assert eos_published == []
    assert eos_result.generated_count == 0


def test_token_packet_text_uses_full_prefix_delta_contract() -> None:
    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens=True):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    sampler = SimpleNamespace(tokenizer=Tokenizer(), last_decoded_output="")
    forwarded = []
    sample_ids = iter((42, 43, 99))
    packets = []

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        logits = torch.zeros((1, 100))
        logits[0, 1 if token_ids[-1] == 7 else 2 if token_ids[-1] == 42 else 99] = 10.0
        return logits, torch.tensor([token_ids[-1]])

    decoder = UnifiedNativeDecoder(
        _thinker(),
        forward_tokens=forward_tokens,
        sample_token=lambda *_args: next(sample_ids),
        sampler=sampler,
        monotonic_ns=_Clock(),
    )
    result = asyncio.run(
        decoder.decode(
            transition_token_ids=[7],
            max_new_tokens=3,
            anticipate_first_token=True,
            on_token_packet=packets.append,
        )
    )

    assert [packet.token_text for packet in packets] == ["<42>", "<43>"]
    assert [token.token_text for token in result.generated_tokens] == ["<42>", "<43>"]


@pytest.mark.parametrize("transition_len", [5, 8])
def test_decode_forwards_transition_up_to_eight_tokens_in_one_native_call(transition_len: int) -> None:
    forwarded: list[list[int]] = []
    first_logits_events = []
    sample_ids = iter((42, 99))

    async def forward_tokens(token_ids: list[int]):
        forwarded.append(list(token_ids))
        return torch.tensor([[float(token_ids[-1]), 0.0]]), torch.tensor([token_ids[-1]])

    def sample_token(logits: torch.Tensor, _generated_ids: tuple[int, ...]):
        token_id = next(sample_ids)
        return torch.tensor([[token_id]]), token_id, logits

    decoder = UnifiedNativeDecoder(
        _thinker(),
        forward_tokens=forward_tokens,
        sample_token=sample_token,
        monotonic_ns=_Clock(),
    )
    result = asyncio.run(decoder.decode(
        transition_token_ids=list(range(10, 10 + transition_len)),
        max_new_tokens=2,
        first_logits_ready=first_logits_events.append,
    ))

    assert forwarded == [list(range(10, 10 + transition_len)), [42]]
    assert len(first_logits_events) == 1
    assert int(first_logits_events[0].final_hidden.item()) == 10 + transition_len - 1
    assert result.generated_count == 1
    assert result.transition_token_count == transition_len


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
