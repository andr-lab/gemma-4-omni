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

from duplex_kernel.kv_cache import TalkerPrefixKV, ThinkerContext, TurnKVState

logger = logging.getLogger("duplex_kernel.session")


@dataclass(frozen=True)
class GenerationToken:
    session_generation_id: int
    response_id: str
    speaker_segment_id: str


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
    # Generation fencing
    session_generation_id: int = 0
    response_id: str | None = None
    speaker_segment_id: str | None = None
    frame_id: int = 0
    response_cancel_notified: bool = False
    stale_audio_events: int = 0
    stale_text_events: int = 0
    stale_history_writes: int = 0
    stale_kv_writes: int = 0
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    # Stato conversazione
    history: list[dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    # Micro-turn in corso
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
            return GenerationToken(self.session_generation_id, response_id, self.speaker_segment_id)

    def next_frame(self) -> int:
        with self._generation_lock:
            self.frame_id += 1
            return self.frame_id

    def end_response(self, token: GenerationToken) -> None:
        with self._generation_lock:
            # A stopped worker may be stale by generation, but it must not be
            # allowed to clear a newer response's active flag.
            if self.response_id == token.response_id:
                self.active_response = False

    def invalidate_generation(self, reason: str = "") -> int:
        with self._generation_lock:
            self.session_generation_id += 1
            self.stop_requested = True
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

    def commit_history(self, token: GenerationToken, base_messages: list[dict[str, Any]], text: str) -> bool:
        with self._generation_lock:
            if not self.is_generation_current(token):
                self.stale_history_writes += 1
                return False
            self.history = [dict(m) for m in base_messages[-16:]] + [{"role": "assistant", "content": text}]
            return True

    def reset_turn(self) -> None:
        """Azzera il turno corrente (buffer audio, mel, KV turno)."""
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.pending_mel = np.empty((0, 128), dtype=np.float32)
        self.turn_kv.clear()
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
            sess.turn_kv.clear()
            sess.talker_prefix.clear()
            logger.info("[session] chiusa %s", session_id)

    def _reap_idle(self) -> None:
        # Chiamato mentre _lock è acquisito: non richiamare close(), che
        # tenterebbe di acquisire lo stesso Lock e bloccherebbe il server.
        now = time.time()
        expired = [sid for sid, sess in self.sessions.items()
                   if now - sess.last_activity > self.idle_timeout_s]
        for sid in expired:
            sess = self.sessions.pop(sid)
            sess.turn_kv.clear()
            sess.talker_prefix.clear()
            logger.info("[session] reap idle %s", sid)

    def decode_ready(self) -> list[DuplexSession]:
        """Sessioni con audio pendente o risposta attiva (per il micro-batching)."""
        with self._lock:
            return [s for s in self.sessions.values() if s.pending_audio.size or s.active_response]

    def all(self) -> list[DuplexSession]:
        with self._lock:
            return list(self.sessions.values())


__all__ = ["DuplexSession", "GenerationToken", "SessionManager"]
