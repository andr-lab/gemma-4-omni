from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.server import _direct_audio_snapshot_event_fields, _snapshot_direct_policy_audio
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction


def test_ready_dispatch_rejection_preserves_bounded_snapshot_cause_fields() -> None:
    session = DuplexSession("ready-dispatch-snapshot-diagnostics")
    pcm = np.ones(1280, dtype=np.float32)
    for frame_id in (230, 231, 233):
        session.append_policy_audio(frame_id, pcm)
    session.take_policy_audio_through_frame(230)

    with pytest.raises(ValueError, match="snapshot PCM e span causale non allineati") as caught:
        _snapshot_direct_policy_audio(
            session,
            action=ControlAction.REPLAN,
            first_frame_id=230,
            last_frame_id=233,
            user_active=True,
        )

    fields = _direct_audio_snapshot_event_fields(caught.value)
    assert fields["direct_audio_snapshot_onset_frame_id"] == 230
    assert fields["direct_audio_snapshot_consumed_through_frame_id"] == 230
    assert fields["direct_audio_snapshot_retained_frame_count"] == 2
    assert fields["direct_audio_snapshot_missing_frame_ids"] == [232]
    assert fields["direct_audio_snapshot_missing_frame_ids_truncated"] is False
    assert not any(isinstance(value, np.ndarray) for value in fields.values())


def test_ready_dispatch_snapshot_missing_ids_are_bounded() -> None:
    session = DuplexSession("ready-dispatch-snapshot-diagnostics-cap")
    session.append_policy_audio(100, np.ones(1280, dtype=np.float32))

    with pytest.raises(ValueError) as caught:
        _snapshot_direct_policy_audio(
            session,
            action=ControlAction.BACKCHANNEL,
            first_frame_id=100,
            last_frame_id=140,
            user_active=True,
        )

    fields = _direct_audio_snapshot_event_fields(caught.value)
    assert fields["direct_audio_snapshot_missing_frame_count"] == 40
    assert len(fields["direct_audio_snapshot_missing_frame_ids"]) == 16
    assert fields["direct_audio_snapshot_missing_frame_ids_truncated"] is True
