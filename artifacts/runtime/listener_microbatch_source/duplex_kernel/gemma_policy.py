"""Gemma learned turn policy contract for continuous full-duplex.

This policy replaces TurnHeadPolicy after QLoRA fine-tuning without
requiring any changes to the underlying kernel, scheduler, or session loop.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput


# Pre-allocated Gemma control tokens reserved in vocabulary expansion for full-duplex
GEMMA_CONTROL_TOKENS: Mapping[str, int] = {
    ControlAction.LISTEN.value: 256000,
    ControlAction.SPEAK.value: 256001,
    ControlAction.CONTINUE.value: 256002,
    ControlAction.YIELD.value: 256003,
    ControlAction.BACKCHANNEL.value: 256004,
    ControlAction.REPLAN.value: 256005,
    ControlAction.HARD_STOP.value: 256006,
}
GEMMA_TOKEN_TO_ACTION: Mapping[int, ControlAction] = {
    token_id: ControlAction(action_name)
    for action_name, token_id in GEMMA_CONTROL_TOKENS.items()
}


class GemmaPolicy:
    """TurnPolicy implementation driven by Gemma listener states and control tokens.

    Adheres strictly to the TurnPolicy protocol:
      - step(PolicyInput) -> PolicyDecision
      - update_hidden(hidden) -> None
      - reset() -> None
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        default_action: ControlAction = ControlAction.LISTEN,
        min_confidence: float = 0.50,
    ) -> None:
        self.model = model
        self.default_action = default_action
        self.min_confidence = float(min_confidence)
        self._last_hidden: Any | None = None
        self._step_count: int = 0

    def update_hidden(self, hidden: Any | None) -> None:
        self._last_hidden = hidden

    def reset(self) -> None:
        self._last_hidden = None
        self._step_count = 0

    def step(self, state: PolicyInput) -> PolicyDecision:
        """Map listener state, user audio, and model predictions to a PolicyDecision."""
        self._step_count += 1
        if state.listener_hidden is None and self._last_hidden is not None:
            state = replace(state, listener_hidden=self._last_hidden)

        # If a live model or callable decision hook is provided, query it
        if callable(self.model):
            decision = self.model(state)
            if isinstance(decision, PolicyDecision):
                return decision
            if isinstance(decision, ControlAction):
                return PolicyDecision(
                    action=decision,
                    frame_id=state.frame_id,
                    confidence=1.0,
                    reason="gemma_callable_action",
                    generation_id=state.generation_id,
                )
            if isinstance(decision, int) and decision in GEMMA_TOKEN_TO_ACTION:
                return PolicyDecision(
                    action=GEMMA_TOKEN_TO_ACTION[decision],
                    frame_id=state.frame_id,
                    confidence=1.0,
                    reason=f"gemma_control_token:{decision}",
                    generation_id=state.generation_id,
                )

        # Baseline causal state-machine behavior when model is in stub/pre-QLoRA mode
        if state.speaker_active:
            action = ControlAction.CONTINUE
            reason = "gemma_speaking_default"
        else:
            action = self.default_action
            reason = "gemma_listening_default"

        return PolicyDecision(
            action=action,
            frame_id=state.frame_id,
            confidence=1.0,
            reason=reason,
            generation_id=state.generation_id,
        )


__all__ = [
    "GemmaPolicy",
    "GEMMA_CONTROL_TOKENS",
    "GEMMA_TOKEN_TO_ACTION",
]
