import duplex_kernel.engine as engine
from duplex_kernel.engine import _summarize_cuda_event_pairs
from duplex_kernel.engine import _completed_cuda_event_elapsed_ms


class _CompletedEvent:
    def __init__(self, duration_ms: float, complete: bool = True) -> None:
        self.duration_ms = duration_ms
        self.complete = complete
        self.queried = False

    def query(self) -> bool:
        self.queried = True
        return self.complete

    def elapsed_time(self, end_event: "_CompletedEvent") -> float:
        assert end_event is self
        return self.duration_ms


class _IncompleteTimingEvent(_CompletedEvent):
    def elapsed_time(self, end_event: "_CompletedEvent") -> float:
        raise RuntimeError("Both events must be completed before calculating elapsed time.")


def test_cuda_timing_summary_uses_only_completed_events_without_synchronizing() -> None:
    complete = [_CompletedEvent(value) for value in (1.0, 2.0, 3.0, 4.0)]
    pending = _CompletedEvent(99.0, complete=False)

    summary = _summarize_cuda_event_pairs(
        [(event, event) for event in complete] + [(pending, pending)]
    )

    assert summary == {
        "sample_count": 4,
        "pending_count": 1,
        "p50_ms": 2.5,
        "p95_ms": 3.85,
        "max_ms": 4.0,
    }
    assert all(event.queried for event in [*complete, pending])


def test_single_cuda_timing_skips_incomplete_events_without_failing_inference() -> None:
    start = _CompletedEvent(0.0)
    pending_end = _IncompleteTimingEvent(0.0, complete=False)

    assert _completed_cuda_event_elapsed_ms(start, pending_end) is None
    assert start.queried and pending_end.queried

    complete_start = _CompletedEvent(0.0)
    complete_end = _CompletedEvent(1.25)
    complete_end.elapsed_time = lambda _end: 1.25
    assert _completed_cuda_event_elapsed_ms(complete_start, complete_end) == 1.25


def test_talker_cuda_timing_claim_is_one_shot(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_TALKER_CUDA_TIMING_PROBE_CLAIMED", False)

    assert engine._claim_talker_cuda_timing_probe() is True
    assert engine._claim_talker_cuda_timing_probe() is False
