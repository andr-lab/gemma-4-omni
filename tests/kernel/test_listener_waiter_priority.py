from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import numpy as np

from duplex_kernel.listener import GemmaListener, _ListenerWaiterPriority


def test_listener_waiting_signal_stays_set_until_last_concurrent_step_finishes() -> None:
    waiting = threading.Event()
    priority = _ListenerWaiterPriority(waiting)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    def step(entered: threading.Event, release: threading.Event) -> None:
        with priority.registered():
            entered.set()
            assert release.wait(2.0)

    first = threading.Thread(target=step, args=(first_entered, release_first))
    second = threading.Thread(target=step, args=(second_entered, release_second))
    first.start()
    assert first_entered.wait(2.0)
    second.start()
    assert second_entered.wait(2.0)
    assert waiting.is_set()
    assert priority.waiter_count == 2

    release_first.set()
    first.join(2.0)
    assert not first.is_alive()
    assert waiting.is_set()
    assert priority.waiter_count == 1

    release_second.set()
    second.join(2.0)
    assert not second.is_alive()
    assert not waiting.is_set()
    assert priority.waiter_count == 0


def test_listener_waiting_signal_clears_on_step_exception() -> None:
    waiting = threading.Event()
    priority = _ListenerWaiterPriority(waiting)

    with pytest.raises(RuntimeError, match="fixture failure"):
        with priority.registered():
            assert waiting.is_set()
            raise RuntimeError("fixture failure")

    assert not waiting.is_set()
    assert priority.waiter_count == 0


def test_gemma_listener_releases_registration_when_step_validation_raises() -> None:
    waiting = threading.Event()
    priority = _ListenerWaiterPriority(waiting)
    listener = GemmaListener(backend=SimpleNamespace(
        _waiting_for_lm=waiting,
        _listener_waiter_priority=priority,
        _tick_started_event=None,
    ))

    with pytest.raises(TypeError, match="GemmaListenerState"):
        listener.step(np.zeros(listener.frame_samples, dtype=np.float32), None, object())

    assert not waiting.is_set()
    assert priority.waiter_count == 0
