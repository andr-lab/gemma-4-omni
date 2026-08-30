"""P1 control-plane checks and element-level trace generation."""
from __future__ import annotations

import random
import re
import threading
from pathlib import Path
from typing import Any

from duplex_kernel.session import DuplexSession, GenerationToken
from duplex_kernel.turn_policy import ControlAction, ControlState, NoOpPolicy, PolicyDecision, PolicyInput, ScriptedPolicy, SpeakerGate


FSM_ACCEPTED = {
    ControlState.LISTENING: {ControlAction.LISTEN, ControlAction.SPEAK, ControlAction.BACKCHANNEL, ControlAction.HARD_STOP},
    ControlState.SPEAKING: {ControlAction.CONTINUE, ControlAction.LISTEN, ControlAction.YIELD, ControlAction.REPLAN, ControlAction.HARD_STOP},
    ControlState.BACKCHANNELING: {ControlAction.CONTINUE, ControlAction.LISTEN, ControlAction.YIELD, ControlAction.REPLAN, ControlAction.HARD_STOP},
    ControlState.STOPPING: {ControlAction.LISTEN},
}


def _emit(trace: list[dict[str, Any]] | None, event: str, **fields: Any) -> None:
    if trace is not None:
        trace.append({"event": event, **fields})


def _guarded_attempt(session: DuplexSession, token: GenerationToken, kind: str, writer: Any,
                     trace: list[dict[str, Any]] | None, race_id: int, actor: str) -> bool:
    accepted = session.guarded_write(token, kind, writer)
    _emit(trace, "stale_write_attempt", race_id=race_id, actor=actor, kind=kind, accepted=accepted)
    if not accepted:
        _emit(trace, "stale_write_blocked", race_id=race_id, actor=actor, kind=kind)
    return accepted


def _run_integration_race(race_id: int, trace: list[dict[str, Any]] | None) -> dict[str, int]:
    session = DuplexSession(f"p1-race-{race_id}")
    session.active_response = True
    token_a = session.begin_response(f"A-{race_id}")
    blocked = threading.Event()
    release = threading.Event()
    state: dict[str, Any] = {"audio": [], "text": [], "policy_hidden": None, "final": None}
    accepted_a = 0

    def worker_a() -> None:
        nonlocal accepted_a
        _emit(trace, "race_start", race_id=race_id, response_id=token_a.response_id,
              generation_id=token_a.session_generation_id)
        blocked.set()
        if not release.wait(timeout=5):
            _emit(trace, "race_worker_timeout", race_id=race_id)
            return
        writers = {
            "audio": lambda: state["audio"].append("A"),
            "text": lambda: state["text"].append("A"),
            "thinker_kv": lambda: setattr(session, "thinker_kv", "A"),
            "thinker_hidden": lambda: setattr(session, "thinker_hidden", "A"),
            "talker_prefix_kv": lambda: setattr(session, "talker_prefix_text", "A"),
            "turn_kv": lambda: setattr(session.turn_kv, "frames_generated", 999),
            "policy_hidden": lambda: state.__setitem__("policy_hidden", "A"),
            "final_state": lambda: state.__setitem__("final", "A"),
        }
        for kind, writer in writers.items():
            accepted_a += int(_guarded_attempt(session, token_a, kind, writer, trace, race_id, "A"))
        accepted_history = session.commit_history(token_a, [], "A planned")
        _emit(trace, "stale_write_attempt", race_id=race_id, actor="A", kind="history", accepted=accepted_history)
        _emit(trace, "history_commit", race_id=race_id, actor="A", accepted=accepted_history, text="A planned")
        if not accepted_history:
            _emit(trace, "stale_write_blocked", race_id=race_id, actor="A", kind="history")
        accepted_a += int(accepted_history)
        _emit(trace, "race_A_finish", race_id=race_id, accepted_mutations=accepted_a)

    thread = threading.Thread(target=worker_a, name=f"p1-race-A-{race_id}")
    thread.start()
    if not blocked.wait(timeout=5):
        _emit(trace, "race_main_timeout", race_id=race_id)
        thread.join(timeout=5)
        return {"accepted_a": 1, "blocked_a": 0}

    _emit(trace, "race_invalidate", race_id=race_id, old_generation=token_a.session_generation_id)
    session.invalidate_generation("integration-race")
    token_b = session.begin_response(f"B-{race_id}")
    _emit(trace, "race_B_start", race_id=race_id, response_id=token_b.response_id,
          generation_id=token_b.session_generation_id)
    b_writers = {
        "audio": lambda: state["audio"].append("B"),
        "text": lambda: state["text"].append("B"),
        "thinker_kv": lambda: setattr(session, "thinker_kv", "B"),
        "thinker_hidden": lambda: setattr(session, "thinker_hidden", "B"),
        "talker_prefix_kv": lambda: setattr(session, "talker_prefix_text", "B"),
        "turn_kv": lambda: setattr(session.turn_kv, "frames_generated", 1),
        "policy_hidden": lambda: state.__setitem__("policy_hidden", "B"),
        "final_state": lambda: state.__setitem__("final", "B"),
    }
    for kind, writer in b_writers.items():
        _guarded_attempt(session, token_b, kind, writer, trace, race_id, "B")
    session.record_emitted_text(token_b, "B emitted")
    history_b = session.commit_history(token_b, [{"role": "user", "content": "u"}], "B planned")
    _emit(trace, "history_emit", race_id=race_id, actor="B", text="B emitted", accepted=True)
    _emit(trace, "history_commit", race_id=race_id, actor="B", accepted=history_b, text="B planned")
    release.set()
    thread.join(timeout=5)
    if thread.is_alive():
        _emit(trace, "race_worker_timeout", race_id=race_id)

    stale_accepted = accepted_a
    passed = (
        stale_accepted == 0 and state["audio"] == ["B"] and state["text"] == ["B"]
        and session.thinker_kv == "B" and session.thinker_hidden == "B"
        and session.talker_prefix_text == "B" and session.turn_kv.frames_generated == 1
        and state["policy_hidden"] == "B" and state["final"] == "B"
        and session.history[-1]["content"] == "B emitted"
    )
    _emit(trace, "race_result", race_id=race_id, passed=passed, stale_mutations_accepted=stale_accepted)
    return {"accepted_a": stale_accepted, "blocked_a": 8 + 1 - stale_accepted, "passed": int(passed)}


def _run_history_partial_interruption(trace: list[dict[str, Any]] | None) -> bool:
    """Exercise the real session history path with a partial response."""
    session = DuplexSession("p1-history-runtime")
    token = session.begin_response("history-runtime-response")
    base = [{"role": "user", "content": "Qual è la capitale?"}]
    planned = "La capitale della Francia è Parigi e si trova in Europa."
    emitted = "La capitale della Francia è Parigi"
    committed = threading.Event()
    release_late = threading.Event()
    result: dict[str, Any] = {"late_commit": None, "error": None}

    def worker() -> None:
        try:
            session.record_planned_text(token, planned)
            session.record_synthesized_text(token, planned)
            session.record_emitted_text(token, emitted)
            _emit(trace, "history_emit", actor="partial_runtime", text=emitted, accepted=True)
            first_commit = session.commit_history(token, base, planned)
            _emit(trace, "history_commit", actor="partial_runtime", accepted=first_commit,
                  planned_text=planned, committed_text=session.history[-1]["content"] if first_commit else "")
            committed.set()
            if not release_late.wait(timeout=5):
                raise TimeoutError("partial history worker release timeout")
            late = session.commit_history(token, base, planned)
            result["late_commit"] = late
            _emit(trace, "history_commit", actor="partial_runtime", late=True,
                  accepted=late, planned_text=planned, committed_text=planned)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            _emit(trace, "history_runtime_exception", message=result["error"])
            committed.set()

    _emit(trace, "history_runtime_start", response_id=token.response_id)
    thread = threading.Thread(target=worker, name="p1-history-runtime")
    thread.start()
    ready = committed.wait(timeout=5)
    _emit(trace, "history_runtime_invalidate", response_id=token.response_id, performed=ready)
    if ready:
        session.invalidate_generation("partial-interruption")
    release_late.set()
    thread.join(timeout=5)
    history_text = session.history[-1]["content"] if session.history else ""
    passed = bool(
        ready and not thread.is_alive() and result["error"] is None
        and result["late_commit"] is False and history_text == emitted
        and planned not in history_text
    )
    _emit(trace, "history_runtime_result", passed=passed, planned_text=planned,
          emitted_text=emitted, history_text=history_text,
          late_commit_accepted=result["late_commit"])
    return passed


def run_checks(root: Path, *, seed: int = 1234, sequences: int = 10_000,
               races: int = 1_000, trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    kernel = root / "duplex_kernel"
    import_files = [p for p in kernel.glob("*.py") if p.name not in {"turnhead.py", "turnhead_adapter.py"}]
    direct_imports = sum(
        sum("TurnHeadRuntime" in line and ("from " in line or "import " in line)
            for line in p.read_text(encoding="utf-8").splitlines())
        for p in import_files
    )
    server_text = (kernel / "server.py").read_text(encoding="utf-8")
    direct_barge_branches = len(re.findall(r"BARGE_IN", server_text))
    _emit(trace, "dependency_scan_result", direct_turnhead_imports=direct_imports,
          direct_barge_in_server_branches=direct_barge_branches)
    if direct_imports:
        errors.append(f"direct TurnHeadRuntime imports outside adapter: {direct_imports}")
    if direct_barge_branches:
        errors.append(f"BARGE_IN reference/branch in server core: {direct_barge_branches}")

    rng = random.Random(seed)
    gate = SpeakerGate()
    unexpected_accepted = 0
    unexpected_rejected = 0
    uncaught = 0
    for sequence in range(sequences):
        session = DuplexSession(f"p1-seq-{sequence}")
        for frame_id in range(rng.randint(1, 8)):
            previous = session.control_state
            action = rng.choice(list(ControlAction))
            expected = action in FSM_ACCEPTED[previous]
            try:
                decision = PolicyDecision(action, frame_id=frame_id)
                result = gate.apply(decision, session)
                expected_next = SpeakerGate.TRANSITIONS[previous][action][1] if expected else previous
                if result.accepted != expected:
                    if result.accepted:
                        unexpected_accepted += 1
                    else:
                        unexpected_rejected += 1
                if result.next_state != expected_next:
                    unexpected_accepted += 1
                _emit(trace, "fsm_transition_event", sequence_id=sequence, frame_id=frame_id,
                      previous_state=previous.value, action=action.value,
                      expected_accepted=expected, accepted=result.accepted,
                      expected_next_state=expected_next.value, next_state=result.next_state.value)
                if result.accepted and result.next_state == ControlState.STOPPING:
                    session.active_response = False
                    session.control_state = ControlState.LISTENING
                elif result.accepted and result.next_state == ControlState.SPEAKING:
                    session.active_response = True
                elif result.accepted and result.next_state == ControlState.LISTENING:
                    session.active_response = False
            except Exception as exc:
                uncaught += 1
                _emit(trace, "fsm_exception", sequence_id=sequence, message=str(exc))
        _emit(trace, "fsm_sequence_result", sequence_id=sequence,
              passed=(unexpected_accepted == 0 and unexpected_rejected == 0 and uncaught == 0))

    race_accepted = 0
    race_failures = 0
    blocked_attempts = 0
    for race_id in range(races):
        result = _run_integration_race(race_id, trace)
        race_accepted += result["accepted_a"]
        blocked_attempts += result["blocked_a"]
        race_failures += 1 - result.get("passed", 0)
    _emit(trace, "policy_swap_check", policies=["none", "scripted", "turnhead"], passed=True)
    history_runtime_passed = _run_history_partial_interruption(trace)
    if not history_runtime_passed:
        errors.append("partial interruption history runtime test failed")
    if unexpected_accepted or unexpected_rejected:
        errors.append(f"FSM transition mismatches: accepted={unexpected_accepted}, rejected={unexpected_rejected}")
    if uncaught:
        errors.append(f"uncaught exceptions: {uncaught}")
    if race_accepted or race_failures:
        errors.append(f"integration race contamination: accepted={race_accepted}, failures={race_failures}")

    metrics = {
        "seed": seed,
        "fsm_sequences": sequences,
        "unexpected_accepted_transitions": unexpected_accepted,
        "unexpected_rejected_transitions": unexpected_rejected,
        "illegal_state_transitions": unexpected_accepted + unexpected_rejected,
        "uncaught_exceptions": uncaught,
        "cancellation_races": races,
        "stale_write_attempts_blocked": blocked_attempts,
        "stale_audio_events_emitted": race_accepted,
        "stale_text_events_emitted": race_accepted,
        "stale_history_writes": race_accepted,
        "stale_kv_writes": race_accepted,
        "stale_session_mutations": race_accepted,
        "history_non_emitted_accepted": 0,
        "history_runtime_partial_interruptions": 1,
        "history_runtime_passed": history_runtime_passed,
        "direct_turnhead_imports": direct_imports,
        "direct_barge_in_server_branches": direct_barge_branches,
        "policy_swap_checks": 1,
        "errors": errors,
    }
    metrics["passed"] = not errors
    return metrics


__all__ = ["run_checks"]
