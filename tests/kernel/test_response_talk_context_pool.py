from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.response_talk_context import ResponseTalkContextPool
from duplex_kernel.server import _EngineSynthesisOwner, _InferenceExecutionGate
from duplex_kernel.subtalker_graph import SubtalkerGraph


class _Graph:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def test_pool_leases_two_distinct_graphs_and_cancellation_keeps_slot_owned() -> None:
    graphs = [_Graph(), _Graph()]
    pool = ResponseTalkContextPool(graphs)

    first = pool.acquire("session-a", "response-1", generation_id=1, timeout=0.1)
    second = pool.acquire("session-b", "response-2", generation_id=7, timeout=0.1)

    assert first.subtalker_graph is not second.subtalker_graph
    assert graphs[0].resets == graphs[1].resets == 1
    assert first.is_current() and second.is_current()

    first.cancel()
    assert first.is_cancelled()
    assert pool.active_count == 2
    with pytest.raises(TimeoutError):
        pool.acquire("session-c", "response-3", generation_id=9, timeout=0.01)

    pool.release(first)
    replacement = pool.acquire("session-c", "response-3", generation_id=9, timeout=0.1)
    assert replacement.slot_id == first.slot_id
    assert replacement.lease_id != first.lease_id
    assert not first.is_current()
    with pytest.raises(RuntimeError, match="stale or already released"):
        pool.release(first)
    pool.release(second)
    pool.release(replacement)
    assert pool.active_count == 0


def test_pool_keeps_code2wav_window_scratch_private_per_response() -> None:
    subtalker_graphs = [_Graph(), _Graph()]
    decoder_graphs = [_Graph(), _Graph()]
    pool = ResponseTalkContextPool(subtalker_graphs, code2wav_graphs=decoder_graphs)

    first = pool.acquire("session-a", "response-1", generation_id=1, timeout=0.1)
    second = pool.acquire("session-b", "response-2", generation_id=2, timeout=0.1)

    assert first.code2wav_graph is decoder_graphs[0]
    assert second.code2wav_graph is decoder_graphs[1]
    assert first.code2wav_graph is not second.code2wav_graph
    pool.release(first)
    pool.release(second)


def test_pool_release_waits_for_callback_completion() -> None:
    graph = _Graph()
    pool = ResponseTalkContextPool([graph])
    lease = pool.acquire("session-a", "response-1", generation_id=1, timeout=0.1)
    lease.callback_started()

    with pytest.raises(RuntimeError, match="callbacks still active"):
        pool.release(lease)

    lease.callback_finished()
    pool.release(lease)
    assert pool.active_count == 0


def test_concurrent_synthesis_owner_cancels_only_the_matching_response() -> None:
    owner = _EngineSynthesisOwner()
    cancelled = {"a": threading.Event(), "b": threading.Event()}
    owner.claim("session-a", "response-a", allow_concurrent=True, cancel=cancelled["a"].set)
    owner.claim("session-b", "response-b", allow_concurrent=True, cancel=cancelled["b"].set)

    assert owner.stop_if_owned("session-a", "response-a", lambda: pytest.fail("global stop used"))
    assert cancelled["a"].is_set()
    assert not cancelled["b"].is_set()
    with pytest.raises(RuntimeError, match="capacità synthesis parallela"):
        owner.claim("session-c", "response-c", allow_concurrent=True, cancel=lambda: None)
    owner.release("session-a", "response-a")
    owner.release("session-b", "response-b")


def test_legacy_execution_waits_for_pooled_responses_but_pool_peers_overlap() -> None:
    gate = _InferenceExecutionGate()
    first_pool_entered = threading.Event()
    release_pool = threading.Event()
    legacy_entered = threading.Event()

    def pooled() -> None:
        with gate.pooled():
            first_pool_entered.set()
            release_pool.wait(1)

    def legacy() -> None:
        with gate.legacy():
            legacy_entered.set()

    worker = threading.Thread(target=pooled)
    worker.start()
    assert first_pool_entered.wait(1)
    with gate.pooled():
        assert not legacy_entered.is_set()
    legacy_worker = threading.Thread(target=legacy)
    legacy_worker.start()
    assert not legacy_entered.wait(0.02)
    release_pool.set()
    worker.join(1)
    legacy_worker.join(1)
    assert legacy_entered.is_set()


def test_captured_subtalker_graph_reset_clears_slot_owned_cache_and_scratch(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    graph = SubtalkerGraph.__new__(SubtalkerGraph)
    graph.captured = True
    graph.device = "cpu"
    cache_key = torch.ones((1, 2))
    cache_value = torch.ones((1, 2))
    cache_length = torch.tensor([2])
    graph.static_cache = SimpleNamespace(layers=[SimpleNamespace(
        keys=cache_key,
        values=cache_value,
        cumulative_length=cache_length,
    )])
    scratch = [torch.ones((2,)) for _ in range(5)]
    (
        graph.static_prefill_inputs,
        graph.static_tok,
        graph.static_pos,
        graph.static_tokens,
        graph.static_logits,
    ) = scratch

    graph.reset()

    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in scratch)
    assert torch.count_nonzero(cache_key).item() == 0
    assert torch.count_nonzero(cache_value).item() == 0
    assert torch.count_nonzero(cache_length).item() == 0
