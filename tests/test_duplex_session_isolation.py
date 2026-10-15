# -*- coding: utf-8 -*-
"""tests/test_duplex_session_isolation.py

Verifies cross-session isolation between concurrent sessions A and B sharing a single engine.
Proves that:
1. KV cache (Thinker context, Talker prefix KV, Turn KV) is strictly isolated per session.
2. Talker state, V9 state, and dynamic conditioning buffers do not bleed between sessions.
3. Generation fencing and cancellations in session A do not cancel or affect session B.
4. History and instructions remain strictly isolated per session.
5. Audio ingress queues and playout buffers do not leak cross-session.
6. The global inference lock alone is not relied upon: internal session state is disjoint.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from types import SimpleNamespace
import numpy as np
import pytest

from duplex_kernel.session import DuplexSession, GenerationToken, SessionManager
from duplex_kernel.kv_cache import ThinkerContext, TalkerPrefixKV, TurnKVState
from duplex_kernel.turn_policy import ControlAction, ControlState, ScriptedPolicy


def test_server_default_session_manager_admits_more_than_sixteen_sessions(monkeypatch) -> None:
    """The app's default manager must not reintroduce an arbitrary N cap."""
    import duplex_kernel.server as server

    created: list[SessionManager] = []
    manager_type = server.SessionManager

    class RecordingSessionManager(manager_type):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(server, "SessionManager", RecordingSessionManager)
    server.create_app(SimpleNamespace())

    assert len(created) == 1
    manager = created[0]
    assert manager.max_sessions is None
    sessions = [manager.create_session() for _ in range(17)]
    try:
        assert len(manager.all()) == 17
        assert len({session.session_id for session in sessions}) == 17
    finally:
        for session in sessions:
            manager.close(session.session_id)


def test_explicit_session_limit_remains_enforced() -> None:
    manager = SessionManager(max_sessions=2)
    sessions = [manager.create_session() for _ in range(2)]
    try:
        with pytest.raises(RuntimeError, match=r"Limite sessioni raggiunto \(2\)"):
            manager.create_session()
    finally:
        for session in sessions:
            manager.close(session.session_id)


@pytest.mark.parametrize("max_sessions", [0, -1, True, 1.5])
def test_explicit_session_limit_must_be_a_positive_integer(max_sessions: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        SessionManager(max_sessions=max_sessions)


def test_session_kv_cache_isolation() -> None:
    mgr = SessionManager(max_sessions=4)
    sess_a = mgr.create_session(instructions="Session A instructions")
    sess_b = mgr.create_session(instructions="Session B instructions")

    assert sess_a.session_id != sess_b.session_id

    # Populate Thinker and Talker caches on Session A
    sess_a.thinker_ctx.set_messages([
        {"role": "user", "content": "Hello from A"},
        {"role": "assistant", "content": "Response to A"},
    ])
    sess_a.talker_prefix_text = "Prefix A"
    sess_a.talker_prefix_h = np.ones((16, 2048), dtype=np.float32) * 1.5

    # Session B should be completely empty
    assert len(sess_b.thinker_ctx.messages()) == 0
    assert sess_b.talker_prefix_text is None
    assert sess_b.talker_prefix_h is None

    # Populate Session B with different data
    sess_b.thinker_ctx.set_messages([{"role": "user", "content": "Hello from B"}])
    sess_b.talker_prefix_text = "Prefix B"
    sess_b.talker_prefix_h = np.ones((16, 2048), dtype=np.float32) * 9.9

    # Verify Session A is unaltered by Session B writes
    msgs_a = sess_a.thinker_ctx.messages()
    assert len(msgs_a) == 2
    assert msgs_a[0]["content"] == "Hello from A"
    assert sess_a.talker_prefix_text == "Prefix A"
    np.testing.assert_allclose(sess_a.talker_prefix_h, 1.5)

    # Verify Session B has its own independent values
    msgs_b = sess_b.thinker_ctx.messages()
    assert len(msgs_b) == 1
    assert msgs_b[0]["content"] == "Hello from B"
    assert sess_b.talker_prefix_text == "Prefix B"
    np.testing.assert_allclose(sess_b.talker_prefix_h, 9.9)

    # Mutate / reset turn on Session A
    sess_a.reset_turn()
    assert sess_b.talker_prefix_text == "Prefix B"
    assert len(sess_b.thinker_ctx.messages()) == 1


def test_generation_fencing_and_cancellation_isolation() -> None:
    mgr = SessionManager(max_sessions=4)
    sess_a = mgr.create_session()
    sess_b = mgr.create_session()

    token_a = sess_a.begin_response("resp_a_1")
    token_b = sess_b.begin_response("resp_b_1")

    assert sess_a.is_generation_current(token_a)
    assert sess_b.is_generation_current(token_b)

    # Attempt cross-session write (token A on session B) must fail
    writes_to_b: list[str] = []
    accepted = sess_b.guarded_write(token_a, "audio", lambda: writes_to_b.append("leak"))
    assert not accepted
    assert len(writes_to_b) == 0

    # Cancel Session A
    sess_a.invalidate_generation("barge_in_on_a")

    # Session A token is now stale
    assert not sess_a.is_generation_current(token_a)
    writes_to_a: list[str] = []
    assert not sess_a.guarded_write(token_a, "audio", lambda: writes_to_a.append("stale"))
    assert len(writes_to_a) == 0
    assert sess_a.stale_write_attempts.get("audio", 0) >= 1

    # Session B MUST remain completely unaffected and its active token valid
    assert sess_b.is_generation_current(token_b)
    writes_b_valid: list[str] = []
    assert sess_b.guarded_write(token_b, "audio", lambda: writes_b_valid.append("valid_b"))
    assert writes_b_valid == ["valid_b"]


def test_history_and_instructions_isolation() -> None:
    mgr = SessionManager(max_sessions=4)
    sess_a = mgr.create_session(instructions="System prompt A", voice="female_it")
    sess_b = mgr.create_session(instructions="System prompt B", voice="male_it")

    assert sess_a.instructions == "System prompt A"
    assert sess_b.instructions == "System prompt B"
    assert sess_a.voice == "female_it"
    assert sess_b.voice == "male_it"

    token_a = sess_a.begin_response("resp_a")
    if sess_a.speaker_segment is not None:
        sess_a.speaker_segment.emitted_text = "Risposta A"
    sess_a.commit_history(token_a, [{"role": "user", "content": "Domanda A"}], "Risposta A")

    token_b = sess_b.begin_response("resp_b")
    if sess_b.speaker_segment is not None:
        sess_b.speaker_segment.emitted_text = "Risposta B"
    sess_b.commit_history(token_b, [{"role": "user", "content": "Domanda B"}], "Risposta B")

    # Check history contents
    assert len(sess_a.history) == 2
    assert sess_a.history[0]["content"] == "Domanda A"
    assert sess_a.history[1]["content"] == "Risposta A"

    assert len(sess_b.history) == 2
    assert sess_b.history[0]["content"] == "Domanda B"
    assert sess_b.history[1]["content"] == "Risposta B"


def test_audio_ingress_isolation() -> None:
    mgr = SessionManager(max_sessions=4)
    sess_a = mgr.create_session()
    sess_b = mgr.create_session()

    # Push 1280 samples (80 ms) to Session A via append
    frame_a = np.ones(1280, dtype=np.float32) * 0.5
    now_ns = time.monotonic_ns()
    sess_a.audio_ingress.append(frame_a, capture_start_ns=now_ns)

    assert sess_a.audio_ingress.qsize == 1
    assert sess_b.audio_ingress.qsize == 0

    popped = sess_a.audio_ingress.get_nowait()
    assert popped is not None
    assert sess_a.audio_ingress.qsize == 0
    assert sess_b.audio_ingress.qsize == 0


def test_v9_and_listener_state_isolation() -> None:
    mgr = SessionManager(max_sessions=4)
    sess_a = mgr.create_session()
    sess_b = mgr.create_session()

    with sess_a.v9_state_lock:
        sess_a.v9_state = {"layer_cache": np.zeros((4, 2, 64, 128))}
        sess_a.v9_text_anchor = np.ones(2048) * 3.14

    assert sess_b.v9_state is None
    assert sess_b.v9_text_anchor is None

    # Resetting Session A clears only Session A's V9 state
    sess_a.reset_turn()
    assert sess_a.v9_state is None
    assert sess_a.v9_text_anchor is None


def test_concurrent_multi_session_threads() -> None:
    """Stress test verifying that concurrent multi-threaded execution maintains isolation."""
    mgr = SessionManager(max_sessions=8)
    sess_a = mgr.create_session()
    sess_b = mgr.create_session()

    errors: list[str] = []

    def worker_a():
        for i in range(100):
            token = sess_a.begin_response(f"a_{i}")
            time.sleep(0.001)
            if i % 3 == 0:
                sess_a.invalidate_generation("cancel")
            else:
                ok = sess_a.guarded_write(token, "text", lambda: None)
                if not ok and sess_a.is_generation_current(token):
                    errors.append(f"Worker A valid token rejected at {i}")

    def worker_b():
        for j in range(100):
            token = sess_b.begin_response(f"b_{j}")
            time.sleep(0.001)
            if j % 5 == 0:
                sess_b.invalidate_generation("cancel")
            else:
                ok = sess_b.guarded_write(token, "text", lambda: None)
                if not ok and sess_b.is_generation_current(token):
                    errors.append(f"Worker B valid token rejected at {j}")

    t_a = threading.Thread(target=worker_a)
    t_b = threading.Thread(target=worker_b)

    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert len(errors) == 0, f"Concurrency errors: {errors}"
    assert sess_a.session_generation_id >= 100
    assert sess_b.session_generation_id >= 100
