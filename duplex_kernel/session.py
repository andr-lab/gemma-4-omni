# -*- coding: utf-8 -*-
"""
duplex_kernel/session.py — SessionManager (sessioni duplex con KV ownership).

Pattern DwarfStar adattato alla nostra architettura: ogni sessione duplex
mantiene la propria KV cache (Thinker rolling + Talker prefix persistente) e
il proprio stato turnhead. Il kernel orchestra il micro-batching delle
sessioni decode-ready.

Sessione duplex (per utente):
  - ThinkerContext: rolling window degli ultimi K turni.
  - TalkerPrefixKV: KV stabile del prefisso [ICL|h_pfx|BOS], calcolata 1 volta.
  - TurnKVState: KV del turno corrente (troncata a fine turno).
  - TurnHeadRuntime: ascolta u_mel + h_t thinker, governa listen/speak/barge-in.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from duplex_kernel.config import FROZEN_QWEN_PLAYOUT_DELAY_FRAMES
from duplex_kernel.future_prefix import ConditioningFrameBuffer
from duplex_kernel.kv_cache import TalkerPrefixKV, ThinkerContext, TurnKVState
from duplex_kernel.listener import AssistantReferenceBuffer, GemmaListenerState
from duplex_kernel.scheduler import AudioIngressQueue
from duplex_kernel.turn_policy import ControlState

logger = logging.getLogger("duplex_kernel.session")


@dataclass(frozen=True)
class GenerationToken:
    session_generation_id: int
    response_id: str
    speaker_segment_id: str


@dataclass
class SpeakerSegmentState:
    segment_id: str
    generation_id: int
    planned_text: str = ""
    synthesized_text: str = ""
    emitted_text: str = ""
    emitted_audio_frames: int = 0


@dataclass
class DuplexSession:
    """Stato di una sessione duplex (KV ownership + turnhead + history)."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    # KV ownership
    thinker_ctx: ThinkerContext = field(default_factory=ThinkerContext)
    talker_prefix: TalkerPrefixKV = field(default_factory=TalkerPrefixKV)
    turn_kv: TurnKVState = field(default_factory=TurnKVState)
    talker_prefix_text: str | None = None
    talker_prefix_h: np.ndarray | None = None
    thinker_kv: Any | None = None
    thinker_cache_ids: list[int] = field(default_factory=list)
    thinker_hidden: Any | None = None
    # Control plane (TurnHead is only an adapter implementation).
    turn_policy: Any | None = None
    control_state: ControlState = ControlState.LISTENING
    # Generation fencing
    session_generation_id: int = 0
    response_id: str | None = None
    speaker_segment_id: str | None = None
    frame_id: int = 0
    response_cancel_notified: bool = False
    speaker_segment: SpeakerSegmentState | None = None
    stale_audio_events: int = 0
    stale_text_events: int = 0
    stale_history_writes: int = 0
    stale_kv_writes: int = 0
    stale_session_mutations: int = 0
    stale_write_attempts: dict[str, int] = field(default_factory=dict)
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    # Stato conversazione
    history: list[dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    # Audio ingress indipendente dal ciclo request/response (P2). P3 collega
    # questo flusso al listener: pending_audio resta solo per il commit API.
    audio_ingress: AudioIngressQueue = field(default_factory=AudioIngressQueue)
    listener: Any | None = field(default=None, repr=False, compare=False)
    listener_state: GemmaListenerState = field(default_factory=GemmaListenerState, repr=False)
    assistant_reference_buffer: AssistantReferenceBuffer = field(
        default_factory=AssistantReferenceBuffer, repr=False, compare=False
    )
    listener_hidden: Any | None = field(default=None, repr=False, compare=False)
    last_listener_output: Any | None = field(default=None, repr=False, compare=False)
    v9_state: Any | None = field(default=None, repr=False, compare=False)
    v9_text_anchor: Any | None = field(default=None, repr=False, compare=False)
    v9_state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    conditioning_buffer: ConditioningFrameBuffer = field(default_factory=ConditioningFrameBuffer, repr=False, compare=False)
    assistant_reference_next_frame_id: int = 0
    last_listener_frame_id: int | None = field(default=None, repr=False)
    # Micro-turn in corso (compatibilità commit/ASR)
    pending_audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    pending_mel: np.ndarray = field(default_factory=lambda: np.empty((0, 128), dtype=np.float32))
    last_activity: float = field(default_factory=time.time)
    active_response: bool = False
    stop_requested: bool = False
    voice: str = "gemma"
    voice_style: str | None = None
    ref_audio: np.ndarray | None = None
    ref_text: str | None = None
    language: str = "Italian"
    tools: list[dict[str, Any]] = field(default_factory=list)
    auto_response: bool = False
    # Speaker-only P4D delay.  The scheduler/listener clock is independent.
    speaker_playout_delay_frames: int = FROZEN_QWEN_PLAYOUT_DELAY_FRAMES
    scheduler: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # The scheduler owns consumption of audio_ingress.  Server adapters
        # inject callbacks but never call get()/policy.step() themselves.
        from duplex_kernel.scheduler import DuplexScheduler
        self.scheduler = DuplexScheduler(self.audio_ingress)

    def touch(self) -> None:
        self.last_activity = time.time()

    def begin_response(self, response_id: str) -> GenerationToken:
        with self._generation_lock:
            self.session_generation_id += 1
            self.response_id = response_id
            self.speaker_segment_id = f"seg_{uuid.uuid4().hex[:12]}"
            self.frame_id = 0
            self.stop_requested = False
            self.response_cancel_notified = False
            self.control_state = ControlState.SPEAKING
            self.speaker_segment = SpeakerSegmentState(self.speaker_segment_id, self.session_generation_id)
            base_frame_id = self.last_listener_frame_id if self.last_listener_frame_id is not None else 0
            self.conditioning_buffer.begin_segment(
                generation_id=self.session_generation_id,
                segment_id=self.speaker_segment_id,
                base_frame_id=base_frame_id,
            )
            return GenerationToken(self.session_generation_id, response_id, self.speaker_segment_id)

    def next_frame(self) -> int:
        with self._generation_lock:
            self.frame_id += 1
            return self.frame_id

    def mark_listener_frame(self, frame_id: int) -> None:
        """Anchor emitted assistant references to the listener timeline."""
        with self._generation_lock:
            frame_id = int(frame_id)
            if self.last_listener_frame_id is None or frame_id > self.last_listener_frame_id:
                self.last_listener_frame_id = frame_id

    def reserve_assistant_reference_frames(self, count: int) -> int:
        """Reserve ids starting at the latest listener frame already consumed."""
        if count <= 0:
            raise ValueError("count deve essere positivo")
        with self._generation_lock:
            anchor = self.last_listener_frame_id
            first = self.assistant_reference_next_frame_id
            if anchor is not None:
                first = max(first, anchor)
            self.assistant_reference_next_frame_id = first + int(count)
            return first

    def end_response(self, token: GenerationToken) -> None:
        with self._generation_lock:
            # A stopped worker may be stale by generation, but it must not be
            # allowed to clear a newer response's active flag.
            if self.response_id == token.response_id:
                self.active_response = False
                self.control_state = ControlState.LISTENING

    def _segment_current(self, token: GenerationToken) -> bool:
        return self.is_generation_current(token) and self.speaker_segment is not None

    def record_planned_text(self, token: GenerationToken, text: str) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["text"] = self.stale_write_attempts.get("text", 0) + 1
                return False
            self.speaker_segment.planned_text = text
            return True

    def record_synthesized_text(self, token: GenerationToken, text: str) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["text"] = self.stale_write_attempts.get("text", 0) + 1
                return False
            self.speaker_segment.synthesized_text += text
            return True

    def record_emitted_text(self, token: GenerationToken, text: str) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["text"] = self.stale_write_attempts.get("text", 0) + 1
                return False
            self.speaker_segment.emitted_text += text
            return True

    def record_emitted_audio(self, token: GenerationToken, frames: int) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["audio"] = self.stale_write_attempts.get("audio", 0) + 1
                return False
            self.speaker_segment.emitted_audio_frames += int(frames)
            return True

    def invalidate_generation(self, reason: str = "") -> int:
        with self._generation_lock:
            self.session_generation_id += 1
            self.stop_requested = True
            self.conditioning_buffer.invalidate()
            logger.debug("[session] generation invalidata %s: %s", self.session_id, reason)
            return self.session_generation_id

    def is_generation_current(self, token: GenerationToken | None = None, *, generation_id: int | None = None,
                              response_id: str | None = None, speaker_segment_id: str | None = None) -> bool:
        with self._generation_lock:
            if token is not None:
                generation_id = token.session_generation_id
                response_id = token.response_id
                speaker_segment_id = token.speaker_segment_id
            return (
                generation_id == self.session_generation_id
                and (response_id is None or response_id == self.response_id)
                and (speaker_segment_id is None or speaker_segment_id == self.speaker_segment_id)
            )

    def guarded_write(self, token: GenerationToken, kind: str, writer: Any) -> bool:
        """Esegue una mutazione solo se il generation token è ancora attivo."""
        with self._generation_lock:
            if not self.is_generation_current(token):
                self.stale_write_attempts[kind] = self.stale_write_attempts.get(kind, 0) + 1
                return False
            writer()
            return True

    def commit_history(self, token: GenerationToken, base_messages: list[dict[str, Any]], text: str) -> bool:
        with self._generation_lock:
            if not self.is_generation_current(token):
                self.stale_write_attempts["history"] = self.stale_write_attempts.get("history", 0) + 1
                return False
            committed_text = self.speaker_segment.emitted_text if self.speaker_segment is not None else text
            if not committed_text:
                return False
            self.history = [dict(m) for m in base_messages[-16:]] + [{"role": "assistant", "content": committed_text}]
            return True

    def reset_turn(self) -> None:
        """Azzera il turno corrente (buffer audio, mel, KV turno)."""
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.pending_mel = np.empty((0, 128), dtype=np.float32)
        self.turn_kv.clear()
        self.conditioning_buffer.invalidate()
        # V9 state is turn-owned.  Keep the listener clock/history alive, but
        # force the next frame to create a fresh state and first-raw-frame
        # anchor.  Resetting in place while clearing the anchor leaves a live
        # state that cannot satisfy IncrementalV9's fixed-anchor contract.
        with self.v9_state_lock:
            self.v9_state = None
            self.v9_text_anchor = None
        self.control_state = ControlState.LISTENING
        if self.turn_policy is not None and hasattr(self.turn_policy, "reset"):
            self.turn_policy.reset()


class SessionManager:
    """Gestore delle sessioni duplex (multi-sessione, micro-batching)."""

    def __init__(self, *, max_sessions: int = 4, session_idle_timeout_s: float = 600.0) -> None:
        self.sessions: dict[str, DuplexSession] = {}
        self.max_sessions = max_sessions
        self.idle_timeout_s = session_idle_timeout_s
        self._lock = threading.Lock()

    def create_session(self, *, instructions: str = "", voice: str = "gemma", language: str = "Italian") -> DuplexSession:
        with self._lock:
            self._reap_idle()
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(f"Limite sessioni raggiunto ({self.max_sessions})")
            sess = DuplexSession(
                session_id=uuid.uuid4().hex,
                instructions=instructions,
                voice=voice,
                language=language,
            )
            self.sessions[sess.session_id] = sess
            logger.info("[session] creata %s (totale %d)", sess.session_id, len(self.sessions))
            return sess

    def get(self, session_id: str) -> DuplexSession | None:
        with self._lock:
            sess = self.sessions.get(session_id)
            if sess is not None:
                sess.touch()
            return sess

    def close(self, session_id: str) -> None:
        with self._lock:
            sess = self.sessions.pop(session_id, None)
        if sess is not None:
            sess.scheduler.stop()
            sess.turn_kv.clear()
            sess.talker_prefix.clear()
            sess.listener_state.close()
            sess.assistant_reference_buffer.clear()
            sess.conditioning_buffer.close()
            sess.audio_ingress.close()
            with sess.v9_state_lock:
                sess.v9_state = None
                sess.v9_text_anchor = None
            logger.info("[session] chiusa %s", session_id)

    def _reap_idle(self) -> None:
        # Chiamato mentre _lock è acquisito: non richiamare close(), che
        # tenterebbe di acquisire lo stesso Lock e bloccherebbe il server.
        now = time.time()
        expired = [sid for sid, sess in self.sessions.items()
                   if now - sess.last_activity > self.idle_timeout_s]
        for sid in expired:
            sess = self.sessions.pop(sid)
            sess.scheduler.stop()
            sess.turn_kv.clear()
            sess.talker_prefix.clear()
            sess.listener_state.close()
            sess.assistant_reference_buffer.clear()
            sess.conditioning_buffer.close()
            sess.audio_ingress.close()
            with sess.v9_state_lock:
                sess.v9_state = None
                sess.v9_text_anchor = None
            logger.info("[session] reap idle %s", sid)

    def decode_ready(self) -> list[DuplexSession]:
        """Sessioni con audio pendente o risposta attiva (per il micro-batching)."""
        with self._lock:
            return [s for s in self.sessions.values() if s.pending_audio.size or s.active_response]

    def all(self) -> list[DuplexSession]:
        with self._lock:
            return list(self.sessions.values())


__all__ = ["DuplexSession", "GenerationToken", "SessionManager", "SpeakerSegmentState"]
