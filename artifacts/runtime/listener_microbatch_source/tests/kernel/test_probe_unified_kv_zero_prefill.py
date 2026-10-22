from types import SimpleNamespace

from tools.probe_unified_kv_zero_prefill import (
    LOGICAL_Q_CAPACITY,
    _cursor_and_q_check,
    _preserve_unified_kv_telemetry,
    _silence_response_check,
    analyze_events,
)


def _row(
    frame_id: int, q_len: int, before: int, after: int, *,
    reason: str | None, queued: bool,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "logical_q_len": q_len,
        "ephemeral_reason": reason,
        "queue_audio_for_next_turn": queued,
        "kv_rollback_tokens": 2 - q_len,
        "kv_rollback_enqueue_ms": 0.05,
        "listener_cache_position_before": before,
        "listener_cache_position_after": after,
        "listener_valid_history_tokens_before": before,
        "listener_valid_history_tokens_after": after,
        "native_cache_current_position_after": after,
        "native_cache_valid_kv_len_after": after,
    }


def test_smoke2_cursor_rows_allow_assistant_q0_and_reset_q2_continuity() -> None:
    # These cursor values are copied from the small smoke2 samples: pre-speech
    # Q0 held 190, active-user Q2 committed 190->192, assistant Q0 held 314 and
    # 320 across text commits, then post-cancel Q2 committed 332->334.
    rows = [
        _row(0, 0, 190, 190, reason="prespeech_silence", queued=False),
        _row(4, 2, 190, 192, reason=None, queued=False),
        _row(65, 0, 314, 314, reason="assistant_turn_open", queued=True),
        _row(66, 0, 320, 320, reason="assistant_turn_open", queued=True),
        _row(75, 2, 332, 334, reason=None, queued=False),
    ]

    prespeech, prespeech_errors = _cursor_and_q_check(rows[:1], 0)
    speech, speech_errors = _cursor_and_q_check(
        rows[1:], LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )

    assert prespeech["status"] == "verified", (prespeech, prespeech_errors)
    assert speech["status"] == "verified", (speech, speech_errors)
    assert speech["phase_counts"] == {"user_q2": 2, "assistant_q0": 2}


def test_event_analyzer_accepts_ephemeral_assistant_rows_and_checks_both_speakers() -> None:
    cursor_rows = {
        0: _row(0, 0, 190, 190, reason="prespeech_silence", queued=False),
        1: _row(1, 0, 190, 190, reason="prespeech_silence", queued=False),
        2: _row(2, 2, 190, 192, reason=None, queued=False),
        3: _row(3, 2, 192, 194, reason=None, queued=False),
        4: _row(4, 0, 314, 314, reason="assistant_turn_open", queued=True),
        5: _row(5, 0, 320, 320, reason="assistant_turn_open", queued=True),
        6: _row(6, 2, 332, 334, reason=None, queued=False),
    }
    events: list[dict[str, object]] = []
    for frame_id, consumed in cursor_rows.items():
        events.extend([
            {
                "type": "probe.input_frame_sent", "speaker": "A", "frame_id": frame_id,
                "source_response_id": "seed", "sent_at_ns": frame_id * 80_000_000,
                "at_ms": frame_id * 80.0, "rms": 0.0 if frame_id < 2 else 0.1,
            },
            {"type": "input_audio_buffer.consumed", "speaker": "A", **consumed},
            {
                "type": "probe.input_frame_ack", "speaker": "A",
                "client_frame_id": frame_id, "ack_frame_id": frame_id,
            },
        ])

    result = analyze_events(events, silence_frames=2, speech_frames=5)
    assert result["cache"]["status"] == "verified", result
    assert result["silence_response"]["status"] == "verified", result
    assert result["cache"]["speech"]["phase_counts"] == {
        "user_q2": 3, "assistant_q0": 2,
    }

    events.append({
        "type": "response.created", "speaker": "B", "response_id": "peer",
        "at_ms": 120.0,
    })
    response_during_silence = analyze_events(events, silence_frames=2, speech_frames=5)
    assert response_during_silence["silence_response"]["status"] == "failed"


def test_assistant_q0_must_not_change_its_own_cursor() -> None:
    row = _row(65, 0, 314, 316, reason="assistant_turn_open", queued=True)
    result, errors = _cursor_and_q_check(
        [row], LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )
    assert result["status"] == "failed"
    assert any("is not Q=0" in error for error in errors)


def test_missing_phase_telemetry_is_not_measurable() -> None:
    row = _row(65, 0, 314, 314, reason="assistant_turn_open", queued=True)
    row.pop("ephemeral_reason")
    result, errors = _cursor_and_q_check(
        [row], LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )
    assert result["status"] == "not_measurable"
    assert result["unresolved_phase_frame_ids"] == [65]


def test_user_q2_does_not_require_ephemeral_reason_field() -> None:
    row = _row(4, 2, 190, 192, reason=None, queued=False)
    row.pop("ephemeral_reason")
    result, errors = _cursor_and_q_check([row], LOGICAL_Q_CAPACITY)
    assert result["status"] == "verified", (result, errors)


def test_silence_check_catches_response_from_either_speaker() -> None:
    result, errors = _silence_response_check(
        [
            {
                "type": "response.created",
                "speaker": "B",
                "response_id": "peer-response",
                "at_ms": 420.0,
                # Server clock value is deliberately unrelated to at_ms.
                "response_created_at_monotonic_ns": 99_000_000_000,
            },
        ],
        silence_start_at_ms=400.0,
        speech_start_at_ms=460.0,
    )
    assert result["status"] == "failed"
    assert result["responses_in_silence"][0]["speaker"] == "B"
    assert errors


def test_silence_check_does_not_claim_success_without_client_timing() -> None:
    result, errors = _silence_response_check(
        [{"type": "response.created", "speaker": "B"}],
        silence_start_at_ms=400.0,
        speech_start_at_ms=460.0,
    )
    assert result["status"] == "not_measurable"
    assert errors


def test_probe_preserves_unified_native_profile_from_thinker_metrics_event() -> None:
    client = SimpleNamespace(
        _project_server_event_for_log=lambda event: {"type": event.get("type")},
        _project_input_ack_fields=lambda event: {"frame_id": event.get("frame_id")},
    )
    old_project, old_ack = _preserve_unified_kv_telemetry(client)
    profile = {"append_steps": [{"native_step_started_ns": 10}]}
    metrics = {"sampled_ids": [11, 12]}
    projected = client._project_server_event_for_log({
        "type": "response.thinker_metrics",
        "unified_native_profile": profile,
        "thinker_runtime_metrics": metrics,
    })
    ack = client._project_input_ack_fields({"frame_id": 7})
    assert projected["unified_native_profile"] == profile
    assert projected["thinker_runtime_metrics"] == metrics
    assert ack == {"frame_id": 7}
    client._project_server_event_for_log = old_project
    client._project_input_ack_fields = old_ack
