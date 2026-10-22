from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from duplex_kernel.engine import PipelineEngine


def test_native_asr_yields_model_lock_between_q1_steps_and_runs_callbacks_outside_lock() -> None:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = "cpu"
    engine._compiled_thinker = None
    engine._thinker_lock = threading.Lock()
    waiting_for_lm = threading.Event()
    engine.listener = SimpleNamespace(backend=SimpleNamespace(_waiting_for_lm=waiting_for_lm))

    timeline: list[str] = []
    callback_lock_states: list[bool] = []
    listener_acquired = threading.Event()
    listener_thread: threading.Thread | None = None

    def stop_check() -> bool:
        callback_lock_states.append(engine._thinker_lock.locked())
        return False

    class _Wrapper:
        def transcribe_audio(self, _pcm: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal listener_thread
            decode_step = kwargs["decode_step"]

            def first_q1() -> str:
                nonlocal listener_thread
                timeline.append("q1-1")

                def listener_step() -> None:
                    waiting_for_lm.set()
                    with engine._thinker_lock:
                        timeline.append("listener")
                        listener_acquired.set()
                    waiting_for_lm.clear()

                listener_thread = threading.Thread(target=listener_step, daemon=True)
                listener_thread.start()
                return "first"

            assert decode_step(first_q1) == "first"
            assert listener_acquired.wait(timeout=1.0), "listener did not acquire between Q1 forwards"
            assert not stop_check()

            def second_q1() -> str:
                timeline.append("q1-2")
                return "second"

            assert decode_step(second_q1) == "second"
            assert not stop_check()
            return {"phase_timings_ms": {}, "cache_info": {}}

    engine._thinker_wrapper = _Wrapper()
    result = engine.transcribe_audio([0.0], max_new_tokens=2, stop_check=stop_check)
    if listener_thread is not None:
        listener_thread.join(timeout=1.0)

    assert result["cache_info"] == {}
    assert timeline == ["q1-1", "listener", "q1-2"]
    assert callback_lock_states and not any(callback_lock_states)
