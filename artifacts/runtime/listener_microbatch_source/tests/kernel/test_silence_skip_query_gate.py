from __future__ import annotations

from types import SimpleNamespace

import torch

from duplex_kernel.server import (
    _silence_skip_query_is_coherent,
    _silence_skip_is_true_idle,
)


def test_pending_turn_and_response_keep_zero_frames_on_the_listener_path() -> None:
    common = {"candidate": True, "query_coherent": True}
    assert not _silence_skip_is_true_idle(**common, unanswered_user_turn=True)
    assert not _silence_skip_is_true_idle(**common, active_response=True)
    assert not _silence_skip_is_true_idle(**common, response_pending=True)
    assert not _silence_skip_is_true_idle(**common, pending_plan=True)
    assert _silence_skip_is_true_idle(**common)


def test_silence_skip_falls_back_when_session_query_was_cleared() -> None:
    session = SimpleNamespace(
        listener_audio_queries=None,
        listener_audio_frame_id=None,
        listener_hidden=None,
        listener_hidden_frame_id=None,
        listener_state=SimpleNamespace(last_hidden_source_frame_id=None),
    )

    assert _silence_skip_query_is_coherent(session, frame_id=12) == (
        False,
        "query_unavailable",
    )


def test_silence_skip_requires_matching_older_query_hidden_and_state() -> None:
    session = SimpleNamespace(
        listener_audio_queries=torch.ones((1, 2, 3)),
        listener_audio_frame_id=11,
        listener_hidden=torch.ones((1, 4)),
        listener_hidden_frame_id=11,
        listener_state=SimpleNamespace(last_hidden_source_frame_id=11),
    )

    assert _silence_skip_query_is_coherent(session, frame_id=12) == (True, None)
    assert _silence_skip_query_is_coherent(session, frame_id=11) == (
        False,
        "query_origin_not_older",
    )
    session.listener_hidden_frame_id = 10
    assert _silence_skip_query_is_coherent(session, frame_id=12) == (
        False,
        "query_hidden_frame_mismatch",
    )
