"""Compatibility adapter for the trained TurnDynamics policy."""
from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput
from duplex_kernel.turnhead import TurnHeadRuntime

SILENCE_AMPLITUDE = 0.0


class TurnHeadPolicy:
    """Map the trained TurnHead labels onto the stable control-plane actions.

    The v1 checkpoint contract is ``h_t(2048) || mel(128) || delta_mel(128)``.
    A raw 1536-wide Gemma listener state is not a substitute for the trained
    2048-wide post-V9 feature and must never be padded/truncated into one.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint: str | None = None,
        min_speak_confidence: float = 0.40,
    ) -> None:
        if isinstance(min_speak_confidence, bool) or not math.isfinite(float(min_speak_confidence)):
            raise ValueError("min_speak_confidence deve essere finito e compreso tra 0 e 1")
        if not 0.0 <= float(min_speak_confidence) <= 1.0:
            raise ValueError("min_speak_confidence deve essere compreso tra 0 e 1")
        self.runtime = TurnHeadRuntime(device=device)
        self.loaded = bool(checkpoint and self.runtime.load(checkpoint))
        self.min_speak_confidence = float(min_speak_confidence)
        self._runtime_lock = threading.RLock()
        self._non_silent_input_seen = False

    @staticmethod
    def _passive_action(state: PolicyInput) -> ControlAction:
        return ControlAction.CONTINUE if state.speaker_active else ControlAction.LISTEN

    def step(self, state: PolicyInput) -> PolicyDecision:
        # Hidden updates can arrive from the Thinker worker while the scheduler
        # is classifying a microphone tick. Serialize mutations of the streaming
        # runtime without changing the session/server policy contract.
        with self._runtime_lock:
            return self._step_locked(state)

    def _step_locked(self, state: PolicyInput) -> PolicyDecision:
        if not self.loaded:
            return PolicyDecision(
                action=self._passive_action(state),
                frame_id=state.frame_id,
                confidence=0.0,
                reason="turnhead_unavailable",
                generation_id=state.generation_id,
            )

        hidden = state.listener_hidden
        # A missing/invalid live projection must evict the previous tick's
        # hidden; otherwise a stale semantic state can authorize SPEAK.
        self.runtime.set_thinker_hidden(hidden)
        if state.user_pcm is not None:
            pcm = np.asarray(state.user_pcm, dtype=np.float32).reshape(-1)
            if pcm.size:
                self._non_silent_input_seen = self._non_silent_input_seen or bool(
                    np.any(np.abs(pcm) > SILENCE_AMPLITUDE)
                )
            self.runtime.feed_pcm(pcm, finalize_frame=True)

        raw = self.runtime.decide()
        name = raw.get("pred_name")
        if not raw.get("active") or not isinstance(name, str):
            return PolicyDecision(
                action=self._passive_action(state),
                frame_id=state.frame_id,
                confidence=0.0,
                reason="turnhead_inactive",
                generation_id=state.generation_id,
            )

        probs_value = raw.get("probs")
        try:
            probs = [float(value) for value in probs_value] if probs_value is not None else []
            confidence = max(probs) if probs and all(math.isfinite(value) for value in probs) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        hidden_ready = bool(getattr(self.runtime, "has_thinker_hidden", False))
        reason = f"turnhead:{name}"

        if name == "BARGE_IN":
            if not state.speaker_active:
                action = ControlAction.LISTEN
            elif not hidden_ready:
                action = ControlAction.CONTINUE
                reason += ":missing_2048_hidden"
            else:
                action = ControlAction.HARD_STOP
        elif name == "YIELD":
            if not state.speaker_active:
                action = ControlAction.LISTEN
            elif not hidden_ready:
                action = ControlAction.CONTINUE
                reason += ":missing_2048_hidden"
            else:
                action = ControlAction.YIELD
        elif name == "EMIT_BACKCHANNEL":
            if state.speaker_active:
                action = ControlAction.CONTINUE
            elif hidden_ready:
                action = ControlAction.BACKCHANNEL
            else:
                action = ControlAction.LISTEN
                reason += ":missing_2048_hidden"
        elif name == "USER_BACKCHANNEL":
            action = self._passive_action(state)
        elif name == "SPEAK":
            if state.speaker_active:
                action = ControlAction.CONTINUE
            elif not hidden_ready:
                action = ControlAction.LISTEN
                reason += ":missing_2048_hidden"
            elif confidence < self.min_speak_confidence:
                action = ControlAction.LISTEN
                reason += ":low_confidence"
            else:
                if not self._non_silent_input_seen:
                    action = ControlAction.LISTEN
                    reason += ":silent_input_only_this_turn"
                else:
                    action = ControlAction.SPEAK
        else:
            action = self._passive_action(state)
            reason = f"turnhead_unknown:{name}"

        return PolicyDecision(
            action=action,
            frame_id=state.frame_id,
            confidence=confidence,
            reason=reason,
            generation_id=state.generation_id,
        )

    def update_hidden(self, hidden: Any | None) -> None:
        with self._runtime_lock:
            self.runtime.set_thinker_hidden(hidden)

    def reset(self) -> None:
        with self._runtime_lock:
            self.runtime.reset()
            self._non_silent_input_seen = False


__all__ = ["TurnHeadPolicy"]
