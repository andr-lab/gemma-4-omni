from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from duplex_kernel.server import (
    POLICY_FRAME_SAMPLES,
    _native_control6_decision_has_valid_input,
    _native_control6_listener_output_is_fresh_for_frame,
    _native_control6_listener_source_frame_id,
    _native_control6_policy_hidden_for_tick,
    _silence_skip_is_true_idle,
    _silence_skip_transition_is_still_idle,
)
from duplex_kernel.session import DuplexSession


@pytest.mark.parametrize(
    ("pcm", "speaker_active", "expected_no_input"),
    [
        (np.zeros(POLICY_FRAME_SAMPLES, dtype=np.float32), True, True),
        (np.zeros(POLICY_FRAME_SAMPLES, dtype=np.float32), False, False),
        (np.full(POLICY_FRAME_SAMPLES, 0.01, dtype=np.float32), True, False),
        (np.zeros(POLICY_FRAME_SAMPLES - 1, dtype=np.float32), True, False),
    ],
)
def test_exact_zero_active_frame_is_no_input_without_hiding_listener_conditioning(
    pcm: np.ndarray,
    speaker_active: bool,
    expected_no_input: bool,
) -> None:
    listener_hidden = torch.ones((1, 2048), dtype=torch.float32)

    policy_hidden, no_input_tick = _native_control6_policy_hidden_for_tick(
        user_pcm=pcm,
        speaker_active=speaker_active,
        listener_hidden=listener_hidden,
    )

    assert no_input_tick is expected_no_input
    assert policy_hidden is (None if expected_no_input else listener_hidden)


def test_native_no_input_decision_only_allows_legal_continue_while_speaking() -> None:
    assert _native_control6_decision_has_valid_input(
        fresh_hidden=False,
        no_input_tick=True,
        speaker_active=True,
        action="CONTINUE",
        legal_actions=("LISTEN", "CONTINUE"),
    )


def test_stale_silence_skipped_listener_output_keeps_its_old_source_frame() -> None:
    output = type("ListenerOutput", (), {})()
    output.frame_id = 12
    output.source_frame_id = 11
    output.fresh = False
    output.silence_skipped = True

    source_frame_id = _native_control6_listener_source_frame_id(output)

    assert source_frame_id == 11
    assert not _native_control6_listener_output_is_fresh_for_frame(
        output,
        12,
        source_frame_id=source_frame_id,
    )
    assert not _native_control6_decision_has_valid_input(
        fresh_hidden=False,
        no_input_tick=False,
        speaker_active=False,
        action="REPLAN",
        legal_actions=("LISTEN", "REPLAN"),
    )


@pytest.mark.parametrize("source_frame_id", ["12", True, -1])
def test_invalid_listener_source_frame_fails_closed(source_frame_id) -> None:
    output = type("ListenerOutput", (), {})()
    output.frame_id = 12
    output.source_frame_id = source_frame_id
    output.fresh = True

    resolved = _native_control6_listener_source_frame_id(output)

    assert resolved is None
    assert not _native_control6_listener_output_is_fresh_for_frame(
        output,
        12,
        source_frame_id=resolved,
    )


def test_missing_listener_source_frame_uses_valid_output_frame_for_legacy_rows() -> None:
    output = type("ListenerOutput", (), {})()
    output.frame_id = 12
    output.source_frame_id = None
    output.fresh = True

    source_frame_id = _native_control6_listener_source_frame_id(output)

    assert source_frame_id == 12
    assert _native_control6_listener_output_is_fresh_for_frame(
        output,
        12,
        source_frame_id=source_frame_id,
    )


def test_current_fresh_listener_output_requires_matching_source_and_output_frames() -> None:
    output = type("ListenerOutput", (), {})()
    output.frame_id = 12
    output.source_frame_id = 12
    output.fresh = True

    source_frame_id = _native_control6_listener_source_frame_id(output)

    assert _native_control6_listener_output_is_fresh_for_frame(
        output,
        12,
        source_frame_id=source_frame_id,
    )
    output.frame_id = 11
    assert not _native_control6_listener_output_is_fresh_for_frame(
        output,
        12,
        source_frame_id=source_frame_id,
    )
    assert not _native_control6_decision_has_valid_input(
        fresh_hidden=False,
        no_input_tick=True,
        speaker_active=True,
        action="REPLAN",
        legal_actions=("LISTEN", "CONTINUE", "REPLAN"),
    )
    assert not _native_control6_decision_has_valid_input(
        fresh_hidden=False,
        no_input_tick=True,
        speaker_active=False,
        action="CONTINUE",
        legal_actions=("LISTEN",),
    )
    assert _native_control6_decision_has_valid_input(
        fresh_hidden=True,
        no_input_tick=False,
        speaker_active=False,
        action="SPEAK",
        legal_actions=("SPEAK",),
    )


def test_silence_skip_is_not_idle_during_response_generation() -> None:
    assert not _silence_skip_is_true_idle(
        candidate=True,
        query_coherent=True,
        control_phase_idle=False,
    )


def test_silence_skip_rejects_a_response_boundary_that_started_and_ended_after_snapshot() -> None:
    assert not _silence_skip_transition_is_still_idle(
        snapshot_generation_id=7,
        current_generation_id=8,
        active_response=False,
        response_pending=False,
        control_state="LISTENING",
    )
    assert _silence_skip_transition_is_still_idle(
        snapshot_generation_id=8,
        current_generation_id=8,
        active_response=False,
        response_pending=False,
        control_state="LISTENING",
    )


def test_begin_response_waits_for_an_inflight_silence_skip_transition_lock() -> None:
    session = DuplexSession("silence-skip-transition")
    held_lock = threading.Lock()
    held_lock.acquire()
    acquire_contended = threading.Event()

    class AcquireObservedLock:
        def __enter__(self):
            if held_lock.acquire(blocking=False):
                return self
            acquire_contended.set()
            held_lock.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            held_lock.release()

        def acquire(self):
            if held_lock.acquire(blocking=False):
                return True
            acquire_contended.set()
            return held_lock.acquire()

        def release(self):
            held_lock.release()

    session._silence_skip_transition_lock = AcquireObservedLock()
    completed = threading.Event()

    def begin() -> None:
        session.begin_response("resp_test")
        completed.set()

    worker = threading.Thread(target=begin)
    worker.start()
    try:
        assert acquire_contended.wait(timeout=2.0)
        assert not completed.wait(timeout=0.05)
    finally:
        held_lock.release()
        worker.join(timeout=2.0)
    assert completed.is_set()
    assert not worker.is_alive()
