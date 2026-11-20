from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from duplex_kernel.unified_native_decoder import NativeDecodeSessionStatus
from duplex_kernel.server_response_runtime import _resume_adopted_controller
from duplex_kernel.server_unified_thinker import (
    _controller_handoff_is_valid,
    _run_controller_to_completion,
)


def test_controller_completion_caches_original_failure_without_advancing_again():
    failure = RuntimeError("original native step failure")
    calls = 0
    controller = {
        "done": False,
        "final_result": None,
        "fatal_error": None,
        "promoted": False,
    }

    def fail_advance(_ready_when):
        nonlocal calls
        calls += 1
        raise failure

    with pytest.raises(RuntimeError) as first:
        _run_controller_to_completion(
            controller,
            advance_controller=fail_advance,
            finish_decoded=lambda _result: pytest.fail("failed step cannot be finalized"),
        )

    assert first.value is failure
    assert controller["done"] is True

    with pytest.raises(RuntimeError) as repeated:
        _run_controller_to_completion(
            controller,
            advance_controller=fail_advance,
            finish_decoded=lambda _result: pytest.fail("failed step cannot be finalized"),
        )

    assert repeated.value is failure
    assert calls == 1


def test_controller_completion_caches_cancelled_result_and_finalizes_once():
    calls = 0
    finish_calls = 0
    controller = {
        "done": False,
        "final_result": None,
        "fatal_error": None,
        "promoted": False,
    }
    step = SimpleNamespace(
        status=NativeDecodeSessionStatus.CANCELLED,
        result=SimpleNamespace(termination_reason="cancelled"),
    )

    def advance(_ready_when):
        nonlocal calls
        calls += 1
        return step

    def finish(result):
        nonlocal finish_calls
        finish_calls += 1
        return {"terminal_reason": result.termination_reason, "cancelled": True}

    first = _run_controller_to_completion(
        controller,
        advance_controller=advance,
        finish_decoded=finish,
    )
    second = _run_controller_to_completion(
        controller,
        advance_controller=advance,
        finish_decoded=finish,
    )

    assert first == second == {"terminal_reason": "cancelled", "cancelled": True}
    assert calls == finish_calls == 1


def test_runtime_resumes_adopted_controller_or_uses_cached_adoption():
    cached = {"text": "adopted", "terminal_reason": "eos"}
    failure = RuntimeError("adopted continuation failed")
    controller = {
        "run_to_completion": lambda: (_ for _ in ()).throw(failure),
        "promoted": True,
        "revision_id": "r2",
        "session": SimpleNamespace(status=None, fatal_error=None, _failed=False, _closed=False),
    }
    branch = {
        "revision_id": "r2",
        "_streaming_revision_id": "r2",
        "_adopted_native_session_controller": controller,
        "_last_adopted_think": cached,
    }

    result = _resume_adopted_controller(controller, branch)

    assert result == {**cached, "revision_handoff": True}
    assert result is not cached

    with pytest.raises(RuntimeError) as propagated:
        _resume_adopted_controller(controller, {**branch, "_last_adopted_think": None})
    assert propagated.value is failure


def test_runtime_rejects_stale_or_failed_adoption_even_with_cached_think():
    controller = {
        "run_to_completion": lambda: pytest.fail("invalid controller must not resume"),
        "promoted": True,
        "revision_id": "r1",
        "fatal_error": None,
        "session": SimpleNamespace(status=None, fatal_error=None, _failed=False, _closed=False),
    }
    stale_branch = {
        "revision_id": "r2",
        "_streaming_revision_id": "r2",
        "_adopted_native_session_controller": controller,
        "_last_adopted_think": {"text": "stale"},
    }

    with pytest.raises(RuntimeError, match="not current and healthy"):
        _resume_adopted_controller(controller, stale_branch)

    failed = RuntimeError("adopted decode failed")
    controller["revision_id"] = "r2"
    controller["fatal_error"] = failed
    failed_branch = {
        **stale_branch,
        "revision_id": "r2",
        "_adopted_native_session_controller": controller,
    }
    with pytest.raises(RuntimeError) as propagated:
        _resume_adopted_controller(controller, failed_branch)
    assert propagated.value is failed


def test_revision_handoff_requires_a_promoted_controller_for_the_current_revision():
    controller = {"promoted": True, "revision_id": "r2"}
    target = {
        "revision_id": "r2",
        "_streaming_revision_id": "r2",
        "_adopted_native_session_controller": controller,
    }
    controller["handoff_state"] = {"target_branch": target}

    assert _controller_handoff_is_valid(controller)
    target["revision_id"] = "r3"
    assert not _controller_handoff_is_valid(controller)


def test_concurrent_controller_completion_has_one_owner_and_one_finalization():
    calls = 0
    finish_calls = 0
    errors: list[BaseException] = []
    results: list[dict[str, object]] = []
    entered = threading.Event()
    release = threading.Event()
    controller = {
        "done": False,
        "final_result": None,
        "fatal_error": None,
        "promoted": False,
    }
    step = SimpleNamespace(
        status=NativeDecodeSessionStatus.DONE,
        result=SimpleNamespace(termination_reason="eos"),
    )

    def advance(_ready_when):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2.0)
        return step

    def finish(result):
        nonlocal finish_calls
        finish_calls += 1
        return {"terminal_reason": result.termination_reason}

    def run():
        try:
            results.append(_run_controller_to_completion(
                controller,
                advance_controller=advance,
                finish_decoded=finish,
            ))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    time.sleep(0.02)
    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert results == [{"terminal_reason": "eos"}, {"terminal_reason": "eos"}]
    assert calls == finish_calls == 1
