"""Policy/control-plane contracts for the pre-QLoRA kernel.

The speaker/listener core depends only on these actions.  Model-specific turn
recognition belongs in ``turnhead_adapter.py`` and can later be replaced by a
GemmaPolicy without changing the server or speaker loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np


class ControlState(str, Enum):
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"
    BACKCHANNELING = "BACKCHANNELING"
    STOPPING = "STOPPING"


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
    """One causal tick; SPEAK commits captured PCM through this frame ID.

    ``user_pcm`` is only the current 80 ms frame.  The server supplies the
    bounded accumulated turn audio separately when materializing SPEAK.
    """

    session_id: str
    frame_id: int
    user_pcm: np.ndarray | None
    speaker_active: bool
    generation_id: int
    peer_turn_active: bool = False
    listener_hidden: Any | None = None
    speaker_segment_id: str | None = None
    listener_audio_queries: Any | None = None
    listener_frame_id: int | None = None
    listener_query_reused: bool = False
    listener_hidden_frame_id: int | None = None
    listener_hidden_at_ns: int | None = None
    # Optional causal PCM context for the 1540-wide control head. History ends
    # at this tick; assistant_reference_pcm is the already-emitted t-1 frame.
    user_pcm_history: Any | None = None
    assistant_reference_pcm: Any | None = None
    # Runtime legality for the six-class Control6 head. None preserves
    # non-realtime/test policy behavior; serving supplies a non-empty set.
    legal_actions: tuple[str, ...] | None = None
    # Fresh final-normalized native Gemma hidden, never the V9 hidden.
    native_control_hidden: Any | None = None
    native_control_hidden_frame_id: int | None = None
    native_control_hidden_at_ns: int | None = None
    # Native Control6 phase_t is constructed from causal assistant t-1/t-2
    # references plus the current user 80 ms frame. It feeds only the LoRA delta.
    native_control_phase_t: Any | None = None
    # Exact assistant reference used by the native1540 phase helper. The policy
    # uses it only to validate whether the already-built log-ratio is usable.
    native_control_assistant_pcm_prior: Any | None = None
    native_control_assistant_prior_frame_id: int | None = None
    # Optional immutable six-logit snapshot scored once for a native listener
    # microbatch. Native Control6 still applies this row's live legal mask.
    native_control6_raw_scores: Any | None = None
    # Stable shared-listener row generation, distinct from the websocket/turn
    # generation used for assistant response cancellation.
    native_listener_generation: int | None = None
    # True only after the scheduler verifies exact-zero user and self-reference PCM.
    native_control_no_input_tick: bool = False
    # Stable identity for the currently unanswered user turn. Unlike the
    # response generation, this stays constant while a response is prepared.
    user_turn_onset_frame_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.peer_turn_active, bool):
            raise TypeError("peer_turn_active must be bool")
        if self.user_turn_onset_frame_id is not None and (
            isinstance(self.user_turn_onset_frame_id, bool)
            or not isinstance(self.user_turn_onset_frame_id, int)
            or self.user_turn_onset_frame_id < 0
        ):
            raise ValueError("user_turn_onset_frame_id must be a non-negative integer or None")


@dataclass(frozen=True)
class PolicyDecision:
    action: ControlAction
    frame_id: int
    confidence: float = 1.0
    reason: str = ""
    generation_id: int | None = None
    listener_frame_id: int | None = None
    source: str = "turn_policy"
    session: str | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class GateEvent:
    action: ControlAction
    accepted: bool = True
    previous_state: ControlState = ControlState.LISTENING
    next_state: ControlState = ControlState.LISTENING
    stop_synthesis: bool = False
    flush_audio_buffer: bool = False
    trigger_backchannel: bool = False
    open_segment: bool = False
    close_segment: bool = False
    advance_segment: bool = False
    safe_boundary: bool = False
    state_changed: bool = False
    stale_generation_id: int | None = None
    generation_id: int | None = None
    speaker_segment_id: str | None = None


@runtime_checkable
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
        return PolicyDecision(action=action, frame_id=state.frame_id, reason="noop", generation_id=state.generation_id)

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
        return PolicyDecision(action=action, frame_id=state.frame_id, reason="scripted", generation_id=state.generation_id)

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        self._index = 0


class ProvisionalFramePolicy:
    """Fail-closed pre-QLoRA fallback; deliberately does not infer speech.

    Without a loaded learned policy there is no VAD-free evidence on which to
    start synthesis.  ``window_frames`` remains accepted for configuration
    compatibility, but never schedules a SPEAK action.
    """

    def __init__(self, *, window_frames: int = 25) -> None:
        if isinstance(window_frames, bool) or int(window_frames) < 1:
            raise ValueError("window_frames deve essere un intero positivo")
        self.window_frames = int(window_frames)

    def step(self, state: PolicyInput) -> PolicyDecision:
        action = ControlAction.CONTINUE if state.speaker_active else ControlAction.LISTEN
        return PolicyDecision(
            action=action,
            frame_id=state.frame_id,
            reason="provisional_fail_closed_no_turn_model",
            generation_id=state.generation_id,
        )

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        pass


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
    """Explicit control-plane FSM plus generation invalidation."""

    TRANSITIONS: dict[ControlState, dict[ControlAction, tuple[bool, ControlState]]] = {
        ControlState.LISTENING: {
            ControlAction.LISTEN: (True, ControlState.LISTENING),
            ControlAction.SPEAK: (True, ControlState.SPEAKING),
            ControlAction.BACKCHANNEL: (True, ControlState.BACKCHANNELING),
            ControlAction.CONTINUE: (False, ControlState.LISTENING),
            ControlAction.YIELD: (False, ControlState.LISTENING),
            ControlAction.REPLAN: (False, ControlState.LISTENING),
            ControlAction.HARD_STOP: (True, ControlState.LISTENING),
        },
        ControlState.SPEAKING: {
            ControlAction.CONTINUE: (True, ControlState.SPEAKING),
            ControlAction.LISTEN: (True, ControlState.SPEAKING),
            ControlAction.YIELD: (True, ControlState.STOPPING),
            ControlAction.REPLAN: (True, ControlState.STOPPING),
            ControlAction.HARD_STOP: (True, ControlState.STOPPING),
            ControlAction.SPEAK: (False, ControlState.SPEAKING),
            ControlAction.BACKCHANNEL: (False, ControlState.SPEAKING),
        },
        ControlState.BACKCHANNELING: {
            ControlAction.CONTINUE: (True, ControlState.BACKCHANNELING),
            ControlAction.LISTEN: (True, ControlState.BACKCHANNELING),
            ControlAction.YIELD: (True, ControlState.STOPPING),
            ControlAction.REPLAN: (True, ControlState.STOPPING),
            ControlAction.HARD_STOP: (True, ControlState.STOPPING),
            ControlAction.SPEAK: (False, ControlState.BACKCHANNELING),
            ControlAction.BACKCHANNEL: (False, ControlState.BACKCHANNELING),
        },
        ControlState.STOPPING: {
            ControlAction.LISTEN: (True, ControlState.STOPPING),
            ControlAction.SPEAK: (False, ControlState.STOPPING),
            ControlAction.CONTINUE: (False, ControlState.STOPPING),
            ControlAction.YIELD: (False, ControlState.STOPPING),
            ControlAction.BACKCHANNEL: (False, ControlState.STOPPING),
            ControlAction.REPLAN: (False, ControlState.STOPPING),
            # HARD_STOP is idempotent and remains destructive even when a
            # cooperative stop is already waiting for cleanup.
            ControlAction.HARD_STOP: (True, ControlState.STOPPING),
        },
    }

    def apply(self, decision: PolicyDecision, session: Any) -> GateEvent:
        action = decision.action
        if not isinstance(action, ControlAction):
            try:
                action = ControlAction(str(action.value if hasattr(action, 'value') else action))
            except Exception:
                raise TypeError("SpeakerGate riceve solo ControlAction")

        # The gate and generation transition share the session lock.  This is
        # deliberately a control-plane lock only; listener/speaker execution
        # remains owned by the scheduler and response worker respectively.
        lock = getattr(session, "_generation_lock", None)
        if lock is None:
            return self._apply_unlocked(action, session)
        with lock:
            decision_generation = getattr(decision, "generation_id", None)
            if decision_generation is not None and decision_generation != session.session_generation_id:
                return GateEvent(
                    action=action,
                    accepted=False,
                    previous_state=getattr(session, "control_state", ControlState.LISTENING),
                    next_state=getattr(session, "control_state", ControlState.LISTENING),
                    generation_id=int(session.session_generation_id),
                    speaker_segment_id=getattr(session, "speaker_segment_id", None),
                )
            return self._apply_unlocked(action, session)

    def _apply_unlocked(self, action: ControlAction, session: Any) -> GateEvent:
        previous = getattr(session, "control_state", ControlState.LISTENING)
        if not isinstance(previous, ControlState):
            previous = ControlState(str(previous))
        accepted, next_state = self.TRANSITIONS[previous][action]
        if not accepted:
            return GateEvent(action=action, accepted=False, previous_state=previous, next_state=previous)

        old_generation = int(getattr(session, "session_generation_id", 0))
        session.control_state = next_state
        active_speaker = previous in {
            ControlState.SPEAKING,
            ControlState.BACKCHANNELING,
            ControlState.STOPPING,
        }
        hard_stop = action == ControlAction.HARD_STOP
        stopping = hard_stop or (active_speaker and action in {
            ControlAction.YIELD,
            ControlAction.REPLAN,
        })
        # LISTEN is observational while a producer is active. YIELD stops at a
        # cooperative boundary and drains accepted PCM; only REPLAN/HARD_STOP
        # invalidate the current generation.
        destructive = hard_stop or (stopping and action == ControlAction.REPLAN)
        already_hard_stopped = (
            previous == ControlState.STOPPING
            and hard_stop
            and getattr(session, "stop_reason", None) == ControlAction.HARD_STOP.value
        )
        if stopping:
            session.stop_requested = True
            session.stop_reason = action.value
            if destructive and not already_hard_stopped:
                session.invalidate_generation(f"control:{action.value}")
        if action in {ControlAction.SPEAK, ControlAction.BACKCHANNEL}:
            open_segment = getattr(session, "open_control_segment", None)
            if callable(open_segment):
                open_segment(action.value)
        if action == ControlAction.CONTINUE and active_speaker:
            advance_segment = getattr(session, "advance_control_segment", None)
            if callable(advance_segment):
                advance_segment()
        if stopping:
            close_segment = getattr(session, "request_control_segment_close", None)
            if callable(close_segment):
                close_segment()

        return GateEvent(
            action=action,
            accepted=True,
            previous_state=previous,
            next_state=next_state,
            stop_synthesis=stopping,
            flush_audio_buffer=destructive,
            trigger_backchannel=action == ControlAction.BACKCHANNEL,
            open_segment=action in {ControlAction.SPEAK, ControlAction.BACKCHANNEL},
            close_segment=stopping,
            advance_segment=action == ControlAction.CONTINUE and active_speaker,
            safe_boundary=action == ControlAction.YIELD and stopping,
            state_changed=previous != next_state,
            stale_generation_id=old_generation if destructive and not already_hard_stopped else None,
            generation_id=int(getattr(session, "session_generation_id", old_generation)),
            speaker_segment_id=getattr(session, "speaker_segment_id", None),
        )


__all__ = [
    "CallablePolicy", "ControlAction", "ControlState", "GateEvent", "NoOpPolicy", "PolicyDecision",
    "ProvisionalFramePolicy",
    "PolicyInput", "ScriptedPolicy", "SpeakerGate", "TurnPolicy",
]
