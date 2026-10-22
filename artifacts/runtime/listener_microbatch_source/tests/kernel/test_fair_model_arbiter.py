from __future__ import annotations

import pytest

from duplex_kernel.engine import ListenerBacklogAdmission
from duplex_kernel.server import _wait_for_listener_backlog_admission


def test_listener_backlog_wait_is_bounded_to_one_admission_window() -> None:
    now = [0.0]

    admitted = _wait_for_listener_backlog_admission(
        lambda: 3 if now[0] < 0.240 else 1,
        lambda: True,
        max_wait_s=0.160,
        poll_interval_s=0.040,
        now=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert admitted is ListenerBacklogAdmission.ADMITTED
    assert now[0] == pytest.approx(0.160)
