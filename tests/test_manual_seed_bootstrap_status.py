import pytest

from client.duplex_stream_e2e import (
    _manual_seed_bootstrap_response_succeeded,
    _manual_seed_bootstrap_terminal_status,
)


def _audible_speak_response() -> dict[str, object]:
    return {
        "action": "SPEAK",
        "response_action": "SPEAK",
        "audio_samples": 407040,
    }


def test_raw_normal_response_done_without_status_is_completed_audible_speak():
    # Shape captured from the live server: successful response.done omits status.
    event = {
        "type": "response.done",
        "response_id": "resp_4ccecdc46432",
        "response_done_at_monotonic_ns": 4860012921576501,
        "runtime_metrics": {
            "audio_response_protocol": {
                "status": "controller_action_fixed",
                "selected_action": "SPEAK",
            },
            "talker": {
                "generated_pcm_samples": 407040,
                "active_pcm_samples": 407040,
                "terminated_by": "eos",
            },
        },
    }

    assert "status" not in event
    assert _manual_seed_bootstrap_terminal_status("response.done", event) == "completed"
    assert _manual_seed_bootstrap_response_succeeded(
        "response.done", event, _audible_speak_response(),
    )


@pytest.mark.parametrize(
    ("event_kind", "status", "reason", "terminated_by"),
    [
        ("response.done", "no_speech", None, None),
        ("response.done", "deferred_until_endpoint", None, None),
        ("response.done", "failed", "inference_error", None),
        ("response.cancelled", "cancelled", None, None),
        ("response.done", "cancelled", None, None),
        ("response.done", None, "generation_aborted", None),
        ("response.done", None, None, "cancelled"),
    ],
)
def test_terminal_failure_payloads_never_enable_bootstrap(
    event_kind: str,
    status: str | None,
    reason: str | None,
    terminated_by: str | None,
):
    event: dict[str, object] = {"type": event_kind}
    if status is not None:
        event["status"] = status
    if reason is not None:
        event["reason"] = reason
    metrics: dict[str, object] = {}
    if terminated_by is not None:
        metrics["talker"] = {"terminated_by": terminated_by}
    if metrics:
        event["runtime_metrics"] = metrics

    normalized = _manual_seed_bootstrap_terminal_status(event_kind, event)
    if status is not None or reason is not None or event_kind == "response.cancelled":
        assert normalized != "completed"
    assert not _manual_seed_bootstrap_response_succeeded(
        event_kind, event, _audible_speak_response(),
    )


def test_non_eos_talker_termination_rejects_other_terminal_reason():
    event = {
        "type": "response.done",
        "runtime_metrics": {"talker": {"terminated_by": "max_frames"}},
    }

    assert not _manual_seed_bootstrap_response_succeeded(
        "response.done", event, _audible_speak_response(),
    )


def test_raw_normal_done_still_requires_speak_and_received_pcm():
    event = {"type": "response.done"}

    assert not _manual_seed_bootstrap_response_succeeded(
        "response.done", event, {"action": "BACKCHANNEL", "audio_samples": 407040},
    )
    assert not _manual_seed_bootstrap_response_succeeded(
        "response.done", event, {"action": "SPEAK", "audio_samples": 0},
    )
