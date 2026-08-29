"""Policy/control-plane contracts for the pre-QLoRA kernel.

The speaker/listener core depends only on these actions.  Model-specific turn
recognition belongs in ``turnhead_adapter.py`` and can later be replaced by a
GemmaPolicy without changing the server or speaker loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Protocol

import numpy as np


class ControlAction(str, Enum):
    LISTEN = "LISTEN"
    SPEAK = "SPEAK"
    CONTINUE = "CONTINUE"
    YIELD = "YIELD"
    BACKCHANNEL = "BACKCHANNEL"
    REPLAN = "REPLAN"
    HARD_STOP = "HARD_STOP"


@dataclass(frozen=True)
class PolicyInput:
    session_id: str
    frame_id: int
    user_pcm: np.ndarray | None
    speaker_active: bool
    generation_id: int
    listener_hidden: Any | None = None
    speaker_segment_id: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    action: ControlAction
    frame_id: int
    confidence: float = 1.0
    reason: str = ""


@dataclass(frozen=True)
class GateEvent:
    action: ControlAction
    accepted: bool = True
    stop_synthesis: bool = False
    flush_audio_buffer: bool = False
    trigger_backchannel: bool = False
    state_changed: bool = False
    stale_generation_id: int | None = None


class TurnPolicy(Protocol):
    def step(self, state: PolicyInput) -> PolicyDecision:
        ...

    def update_hidden(self, hidden: Any | None) -> None:
        ...

    def reset(self) -> None:
        ...


class NoOpPolicy:
    """Deterministic policy used when no learned turn policy is configured."""

    def step(self, state: PolicyInput) -> PolicyDecision:
        action = ControlAction.CONTINUE if state.speaker_active else ControlAction.LISTEN
        return PolicyDecision(action=action, frame_id=state.frame_id, reason="noop")

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        pass


class ScriptedPolicy:
    """Finite/repeating policy for kernel semantics and race tests."""

    def __init__(self, actions: Iterable[ControlAction | str], *, repeat_last: bool = True) -> None:
        self._actions = [a if isinstance(a, ControlAction) else ControlAction(str(a)) for a in actions]
        if not self._actions:
            raise ValueError("ScriptedPolicy richiede almeno un'azione")
        self._repeat_last = repeat_last
        self._index = 0

    def step(self, state: PolicyInput) -> PolicyDecision:
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
        elif self._repeat_last:
            action = self._actions[-1]
        else:
            action = ControlAction.LISTEN
        return PolicyDecision(action=action, frame_id=state.frame_id, reason="scripted")

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        self._index = 0


class CallablePolicy:
    """Small injection adapter useful for embedding a GemmaPolicy later."""

    def __init__(self, fn: Callable[[PolicyInput], PolicyDecision]) -> None:
        self._fn = fn

    def step(self, state: PolicyInput) -> PolicyDecision:
        decision = self._fn(state)
        if not isinstance(decision, PolicyDecision):
            raise TypeError("la policy callable deve restituire PolicyDecision")
        return decision

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        pass


class SpeakerGate:
    """Applies control actions while fencing stopped generations."""

    def apply(self, decision: PolicyDecision, session: Any) -> GateEvent:
        action = decision.action
        if not isinstance(action, ControlAction):
            raise TypeError("SpeakerGate riceve solo ControlAction")
        active = bool(getattr(session, "active_response", False))
        if action in {ControlAction.LISTEN, ControlAction.YIELD, ControlAction.REPLAN, ControlAction.HARD_STOP}:
            if not active:
                return GateEvent(action=action, accepted=True, trigger_backchannel=False)
            old_generation = int(getattr(session, "session_generation_id", 0))
            session.stop_requested = True
            session.invalidate_generation(f"control:{action.value}")
            return GateEvent(
                action=action,
                stop_synthesis=True,
                flush_audio_buffer=action in {ControlAction.REPLAN, ControlAction.HARD_STOP},
                state_changed=True,
                stale_generation_id=old_generation,
            )
        if action == ControlAction.BACKCHANNEL:
            return GateEvent(action=action, trigger_backchannel=True)
        return GateEvent(action=action)


__all__ = [
    "CallablePolicy", "ControlAction", "GateEvent", "NoOpPolicy", "PolicyDecision",
    "PolicyInput", "ScriptedPolicy", "SpeakerGate", "TurnPolicy",
]
