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
from duplex_kernel.turnhead import TurnHeadRuntime

logger = logging.getLogger("duplex_kernel.session")


@dataclass
class DuplexSession:
    """Stato di una sessione duplex (KV ownership + turnhead + history)."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    # KV ownership
    thinker_ctx: ThinkerContext = field(default_factory=ThinkerContext)
    talker_prefix: TalkerPrefixKV = field(default_factory=TalkerPrefixKV)
    turn_kv: TurnKVState = field(default_factory=TurnKVState)
    # Turndynamics
    turnhead: TurnHeadRuntime | None = None
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

    def touch(self) -> None:
        self.last_activity = time.time()

    def reset_turn(self) -> None:
        """Azzera il turno corrente (buffer audio, mel, KV turno)."""
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.pending_mel = np.empty((0, 128), dtype=np.float32)
        self.turn_kv.clear()
        if self.turnhead is not None:
            self.turnhead.reset()


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


__all__ = ["DuplexSession", "SessionManager"]
