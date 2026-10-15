from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from duplex_kernel.listener import GemmaAudioTowerBackend


class _FakeStream:
    def wait_stream(self, other) -> None:
        pass


class _FakeSnapshot:
    def __init__(self, value) -> None:
        self.value = value

    def record_stream(self, stream) -> None:
        pass


class _FakeView:
    def __init__(self, source) -> None:
        self.source = source

    def clone(self):
        return _FakeSnapshot(self.source.value)


class _FakePooler:
    ndim = 3
    shape = (1, 4, 2)

    def __init__(self) -> None:
        self.value = None

    def __getitem__(self, index):
        return _FakeView(self)


class _FakeGraph:
    def __init__(self, backend, replay_entered, release_replay) -> None:
        self.backend = backend
        self.replay_entered = replay_entered
        self.release_replay = release_replay

    def replay(self) -> None:
        observed = self.backend._static_features.value
        if observed == "session-a":
            self.replay_entered.set()
            assert self.release_replay.wait(timeout=1.0)
        self.backend._graph_output.pooler_output.value = observed


class _FakeCuda:
    def __init__(self, stream) -> None:
        self._stream = stream

    def current_stream(self, device=None):
        return self._stream

    @staticmethod
    def stream(stream):
        return nullcontext()


def test_shared_graph_replay_snapshots_each_call_under_workspace_lease() -> None:
    backend = object.__new__(GemmaAudioTowerBackend)
    backend._audio_tower_workspace_lock = threading.RLock()
    backend.device = "cuda:0"
    backend.dtype = "bf16"
    backend.hidden_size = 2
    backend._static_features = SimpleNamespace(value=None)
    backend._graph_stream = _FakeStream()
    replay_entered = threading.Event()
    release_replay = threading.Event()
    backend._cuda_graph = _FakeGraph(backend, replay_entered, release_replay)
    backend._graph_output = SimpleNamespace(pooler_output=_FakePooler())
    fake_torch = SimpleNamespace(cuda=_FakeCuda(_FakeStream()))
    results = {}
    second_update = threading.Event()

    def worker(name: str) -> None:
        def update() -> None:
            backend._static_features.value = name
            if name == "session-b":
                second_update.set()

        snapshot, _ = backend._replay_shared_audio_tower(
            update,
            native_session_steady=True,
        )
        results[name] = snapshot

    with patch.dict("sys.modules", {"torch": fake_torch}):
        first = threading.Thread(target=worker, args=("session-a",))
        second = threading.Thread(target=worker, args=("session-b",))
        first.start()
        assert replay_entered.wait(timeout=1.0)
        second.start()
        time.sleep(0.03)
        assert not second_update.is_set()
        release_replay.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)
        assert not first.is_alive()
        assert not second.is_alive()

    # Later replays mutate the one shared output, while snapshots stay private.
    backend._graph_output.pooler_output.value = "overwritten"
    assert results["session-a"].value == "session-a"
    assert results["session-b"].value == "session-b"
    assert results["session-a"] is not results["session-b"]
