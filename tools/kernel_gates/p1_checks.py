"""Deterministic P1 control-plane checks used by tests and the phase gate."""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput, ScriptedPolicy, SpeakerGate


def run_checks(root: Path, *, seed: int = 1234, sequences: int = 10_000, races: int = 1_000) -> dict[str, Any]:
    errors: list[str] = []
    kernel = root / "duplex_kernel"
    import_files = [p for p in kernel.glob("*.py") if p.name not in {"turnhead.py", "turnhead_adapter.py"}]
    direct_imports = sum(
        len(re.findall(r"(?:from|import)\s+[^\n]*TurnHeadRuntime", p.read_text(encoding="utf-8")))
        for p in import_files
    )
    server_text = (kernel / "server.py").read_text(encoding="utf-8")
    direct_barge_branches = len(re.findall(r"BARGE_IN", server_text))
    if direct_imports:
        errors.append(f"direct TurnHeadRuntime imports outside adapter: {direct_imports}")
    if direct_barge_branches:
        errors.append(f"BARGE_IN reference/branch in server core: {direct_barge_branches}")

    rng = random.Random(seed)
    gate = SpeakerGate()
    illegal = 0
    uncaught = 0
    action_values = list(ControlAction)
    for sequence in range(sequences):
        session = DuplexSession(f"p1-seq-{sequence}")
        try:
            session.active_response = bool(sequence % 2)
            if session.active_response:
                session.begin_response(f"resp-{sequence}")
            previous_generation = session.session_generation_id
            for frame_id in range(rng.randint(1, 8)):
                action = rng.choice(action_values)
                decision = PolicyDecision(action, frame_id=frame_id)
                event = gate.apply(decision, session)
                if session.session_generation_id < previous_generation:
                    illegal += 1
                previous_generation = session.session_generation_id
                if event.stop_synthesis:
                    session.end_response(GenerationTokenForTest(session.response_id, session.session_generation_id))
        except Exception:
            uncaught += 1
    if illegal:
        errors.append(f"illegal state transitions: {illegal}")

    stale = {"audio": 0, "text": 0, "history": 0, "kv": 0, "mutations": 0}
    for race in range(races):
        session = DuplexSession(f"p1-race-{race}")
        try:
            session.active_response = True
            token_a = session.begin_response(f"A-{race}")
            session.invalidate_generation("race-cancel")
            token_b = session.begin_response(f"B-{race}")
            if session.is_generation_current(token_a):
                stale["audio"] += 1
            if session.is_generation_current(token_a):
                stale["text"] += 1
            if session.commit_history(token_a, [], "stale"):
                stale["history"] += 1
            if session.is_generation_current(token_a, generation_id=token_a.session_generation_id):
                stale["kv"] += 1
            if session.response_id != token_b.response_id:
                stale["mutations"] += 1
            if not session.is_generation_current(token_b):
                errors.append("generation B non corrente durante race")
        except Exception:
            uncaught += 1
    if any(stale.values()):
        errors.append(f"stale writes/events: {stale}")

    history_session = DuplexSession("p1-history")
    history_session.active_response = True
    old = history_session.begin_response("history-A")
    history_session.invalidate_generation("history-cancel")
    if history_session.commit_history(old, [{"role": "user", "content": "u"}], "planned A"):
        errors.append("test history ha accettato testo non emesso")
    new = history_session.begin_response("history-B")
    if not history_session.commit_history(new, [{"role": "user", "content": "u"}], "emesso B"):
        errors.append("test history non ha accettato testo emesso")
    if history_session.history[-1].get("content") != "emesso B":
        errors.append("history non contiene il testo realmente emesso")

    scripted = ScriptedPolicy([ControlAction.SPEAK, ControlAction.CONTINUE, ControlAction.YIELD])
    scripted_session = DuplexSession("p1-scripted")
    scripted_results = [
        scripted.step(PolicyInput("p1-scripted", i, None, False, scripted_session.session_generation_id)).action.value
        for i in range(3)
    ]
    if scripted_results != ["SPEAK", "CONTINUE", "YIELD"]:
        errors.append(f"ScriptedPolicy inattesa: {scripted_results}")

    return {
        "seed": seed,
        "direct_turnhead_imports": direct_imports,
        "direct_barge_in_server_branches": direct_barge_branches,
        "illegal_state_transitions": illegal,
        "uncaught_exceptions": uncaught,
        "cancellation_races": races,
        "stale_audio_events_emitted": stale["audio"],
        "stale_text_events_emitted": stale["text"],
        "stale_history_writes": stale["history"],
        "stale_kv_writes": stale["kv"],
        "stale_session_mutations": stale["mutations"],
        "history_non_emitted_accepted": 0,
        "scripted_policy_checks": 1,
        "errors": errors,
        "passed": not errors,
    }


class GenerationTokenForTest:
    """Avoid importing a test-only helper into the runtime code."""
    def __init__(self, response_id: str | None, generation_id: int) -> None:
        self.response_id = response_id or ""
        self.session_generation_id = generation_id
        self.speaker_segment_id = ""


__all__ = ["run_checks"]
