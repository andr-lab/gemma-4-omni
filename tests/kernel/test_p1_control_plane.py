from pathlib import Path

from tools.kernel_gates.p1_checks import run_checks


def test_p1_control_plane_gates():
    result = run_checks(Path(__file__).resolve().parents[2])
    assert result["passed"], result
    assert result["cancellation_races"] == 1000
    assert result["illegal_state_transitions"] == 0
    assert result["uncaught_exceptions"] == 0
    assert result["history_runtime_partial_interruptions"] == 1
    assert result["history_runtime_passed"] is True
    assert result["stale_write_attempts_blocked"] == 9000
