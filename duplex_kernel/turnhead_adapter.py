"""Compatibility adapter for the temporary TurnDetectorHead policy."""
from __future__ import annotations

from typing import Any

from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput
from duplex_kernel.turnhead import TurnHeadRuntime


class TurnHeadPolicy:
    """Maps legacy TurnHead labels onto the stable control-plane actions."""

    def __init__(self, *, device: str = "cuda", checkpoint: str | None = None) -> None:
        self.runtime = TurnHeadRuntime(device=device)
        self.loaded = bool(checkpoint and self.runtime.load(checkpoint))

    def step(self, state: PolicyInput) -> PolicyDecision:
        if state.listener_hidden is not None:
            self.runtime.set_thinker_hidden(state.listener_hidden)
        if state.user_pcm is not None:
            self.runtime.feed_pcm(state.user_pcm)
        if not self.loaded:
            action = ControlAction.CONTINUE if state.speaker_active else ControlAction.LISTEN
            return PolicyDecision(action=action, frame_id=state.frame_id, confidence=1.0, reason="turnhead_unavailable")
        raw = self.runtime.decide()
        name = raw.get("pred_name", "SPEAK")
        if name == "BARGE_IN":
            action = ControlAction.HARD_STOP
        elif name == "YIELD":
            action = ControlAction.YIELD
        elif name == "EMIT_BACKCHANNEL":
            action = ControlAction.BACKCHANNEL
        elif name == "USER_BACKCHANNEL":
            action = ControlAction.CONTINUE if state.speaker_active else ControlAction.LISTEN
        else:
            action = ControlAction.CONTINUE if state.speaker_active else ControlAction.SPEAK
        probs = raw.get("probs") or []
        confidence = float(max(probs)) if probs else 1.0
        return PolicyDecision(action=action, frame_id=state.frame_id, confidence=confidence, reason=f"turnhead:{name}")

    def update_hidden(self, hidden: Any | None) -> None:
        self.runtime.set_thinker_hidden(hidden)

    def reset(self) -> None:
        self.runtime.reset()


__all__ = ["TurnHeadPolicy"]
