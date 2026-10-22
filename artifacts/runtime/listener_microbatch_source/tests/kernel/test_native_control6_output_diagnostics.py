import pytest
import torch
from types import SimpleNamespace

from duplex_kernel.native_control6_diagnostics import (
    attach_native_control6_zero_output_diagnostic,
    native_control6_row_statistics,
    native_control6_validate_batch_output_health,
    native_control6_validate_listener_output_health,
    native_control6_verify_output_health_if_zero,
    native_control6_verify_scored_output_health,
)


def test_row_statistics_are_per_query_and_report_nonfinite_fraction():
    value = torch.tensor(
        [[[3.0, 4.0], [0.0, 0.0], [float("nan"), 2.0], [-2.0, 2.0]]],
        dtype=torch.float32,
    )
    result = native_control6_row_statistics(value)
    assert result is not None
    assert result["shape"] == [4, 2]
    assert result["rows"][0] == pytest.approx(
        {"rms": (12.5) ** 0.5, "maxabs": 4.0, "isfinite_fraction": 1.0}
    )
    assert result["rows"][1] == {"rms": 0.0, "maxabs": 0.0, "isfinite_fraction": 1.0}
    assert result["rows"][2] == {"rms": 2.0, "maxabs": 2.0, "isfinite_fraction": 0.5}


def test_output_health_is_only_read_for_exact_zero_hidden():
    class Health:
        def __init__(self, error=None):
            self.calls = 0
            self.error = error

        def verify_output_health(self):
            self.calls += 1
            if self.error:
                raise RuntimeError(self.error)
            return True

    health = Health()
    skipped = native_control6_verify_output_health_if_zero(
        health, hidden_rms=1e-8, hidden_maxabs=0.0
    )
    assert skipped == {"checked": False, "status": "not_zero_hidden"}
    assert health.calls == 0

    checked = native_control6_verify_output_health_if_zero(
        health, hidden_rms=0.0, hidden_maxabs=0.0
    )
    assert checked == {"checked": True, "status": "healthy"}
    assert health.calls == 1

    fault = Health("output nativo non finito; contesto marcato poisoned")
    result = native_control6_verify_output_health_if_zero(
        fault, hidden_rms=0.0, hidden_maxabs=0.0
    )
    assert result["checked"] is True
    assert result["status"] == "nonfinite_guard"
    assert fault.calls == 1


def test_scored_output_health_is_checked_once_and_fails_closed():
    class Health:
        def __init__(self, error=None):
            self.calls = 0
            self.error = error

        def verify_output_health(self):
            self.calls += 1
            if self.error:
                raise RuntimeError(self.error)
            return True

    health = Health()
    assert native_control6_verify_scored_output_health(
        health, native_policy=True, fresh_hidden=True
    ) == {"checked": True, "status": "healthy"}
    assert health.calls == 1
    assert native_control6_verify_scored_output_health(
        health, native_policy=True, fresh_hidden=False
    ) == {"checked": False, "status": "not_scored"}
    assert health.calls == 1

    fault = Health("output nativo non finito; contesto marcato poisoned")
    result = native_control6_verify_scored_output_health(
        fault, native_policy=True, fresh_hidden=True
    )
    assert result["checked"] is True
    assert result["status"] == "nonfinite_guard"
    assert fault.calls == 1


def test_batch_output_health_requires_matching_row_provenance_and_reports_batch_scope():
    row = SimpleNamespace(
        session_id="session-a",
        session_generation=7,
        state_epoch_id="epoch-a",
        frame_id=12,
        listener_state_version=12,
    )
    snapshot = SimpleNamespace(
        generation=4,
        row=row,
        checked=True,
        nonfinite=False,
        status="healthy",
    )
    expected = dict(
        session_id="session-a",
        generation_id=7,
        state_epoch_id="epoch-a",
        frame_id=12,
        state_version=12,
    )
    result = native_control6_validate_batch_output_health(snapshot, **expected)
    assert result == {
        "checked": True,
        "status": "healthy",
        "batch_health_scope": "all_rows",
        "listener_batch_output_generation": 4,
        "listener_generation_id": 7,
        "listener_state_epoch_id": "epoch-a",
        "listener_frame_id": 12,
        "listener_state_version": 12,
    }

    row.frame_id = 11
    mismatch = native_control6_validate_batch_output_health(snapshot, **expected)
    assert mismatch["checked"] is False
    assert mismatch["status"] == "health_provenance_mismatch"

    row.frame_id = 12
    snapshot.nonfinite = True
    snapshot.status = "nonfinite_guard"
    failed = native_control6_validate_batch_output_health(snapshot, **expected)
    assert failed["checked"] is True
    assert failed["status"] == "nonfinite_guard"
    assert failed["batch_health_scope"] == "all_rows"


def test_listener_output_health_bridge_uses_current_frame_and_pre_step_version():
    row = SimpleNamespace(
        session_id="session-a",
        session_generation=7,
        state_epoch_id="epoch-a",
        frame_id=12,
        listener_state_version=12,
    )
    health = SimpleNamespace(
        generation=4,
        row=row,
        checked=True,
        nonfinite=False,
        status="healthy",
    )
    listener_output = SimpleNamespace(
        frame_id=12,
        listener_state_version=13,
        native_output_health=health,
    )

    result = native_control6_validate_listener_output_health(
        listener_output,
        session_id="session-a",
        generation_id=7,
        state_epoch_id="epoch-a",
        frame_id=12,
    )
    assert result["status"] == "healthy"
    assert result["listener_state_version"] == 12

    listener_output.frame_id = 11
    stale = native_control6_validate_listener_output_health(
        listener_output,
        session_id="session-a",
        generation_id=7,
        state_epoch_id="epoch-a",
        frame_id=12,
    )
    assert stale["checked"] is False
    assert stale["status"] == "health_provenance_mismatch"


def test_decision_trace_hook_records_rows_and_both_guard_outcomes():
    class Health:
        def __init__(self, error=None):
            self.calls = 0
            self.error = error

        def verify_output_health(self):
            self.calls += 1
            if self.error:
                raise RuntimeError(self.error)
            return True

    audio_soft = torch.ones(1, 4, 3)
    q4_output = torch.stack((torch.zeros(3), torch.ones(3), torch.ones(3) * 2, torch.ones(3) * 3)).unsqueeze(0)
    trace = {"native_hidden_rms": 0.0, "native_hidden_maxabs": 0.0}
    healthy = Health()
    attach_native_control6_zero_output_diagnostic(
        trace,
        enabled=True,
        native_policy=True,
        head_scored=True,
        native_listener=healthy,
        audio_soft=audio_soft,
        q4_output=q4_output,
        prior_health_result={"checked": True, "status": "healthy"},
    )
    assert healthy.calls == 0
    assert trace["native_control6_output_guard_status"] == "healthy"
    assert trace["native_control6_input_rows"]["rows"][0]["rms"] == 1.0
    assert trace["native_control6_output_rows"]["rows"][0]["maxabs"] == 0.0

    faulty_trace = {"native_hidden_rms": 0.0, "native_hidden_maxabs": 0.0}
    faulty = Health("output nativo non finito; contesto marcato poisoned")
    attach_native_control6_zero_output_diagnostic(
        faulty_trace,
        enabled=True,
        native_policy=True,
        head_scored=True,
        native_listener=faulty,
        audio_soft=audio_soft,
        q4_output=q4_output,
    )
    assert faulty.calls == 1
    assert faulty_trace["native_control6_output_guard_status"] == "nonfinite_guard"
    assert faulty_trace["native_control6_input_rows"]["rows"][0]["rms"] == 1.0
    assert faulty_trace["native_control6_output_rows"]["rows"][0]["maxabs"] == 0.0

    skipped_trace = {"native_hidden_rms": 1e-8, "native_hidden_maxabs": 1.0}
    skipped = Health()
    attach_native_control6_zero_output_diagnostic(
        skipped_trace,
        enabled=True,
        native_policy=True,
        head_scored=True,
        native_listener=skipped,
        audio_soft=audio_soft,
        q4_output=q4_output,
    )
    assert skipped.calls == 0
    assert skipped_trace["native_control6_output_guard_status"] == "not_zero_hidden"
