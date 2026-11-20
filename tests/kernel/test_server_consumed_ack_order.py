from __future__ import annotations

from duplex_kernel.server import _queue_consumed_ack_before_deferred_work


def test_consumed_ack_is_queued_before_deferred_listener_publication():
    events = []
    consumed = {"type": "input_audio_buffer.consumed", "frame_id": 7}

    class EventWriterQueue:
        def put_nowait(self, event):
            events.append(("queued", dict(event)))
            return True

    def publish_conditioning_row():
        events.append(("conditioning", None))

    _queue_consumed_ack_before_deferred_work(
        EventWriterQueue(),
        consumed,
        publish_conditioning_row,
    )

    assert events == [
        ("queued", consumed),
        ("conditioning", None),
    ]
