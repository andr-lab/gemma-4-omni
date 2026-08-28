# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/serving_adapter.py — Serving Runtime Adapter per OpenAI Realtime WebSocket.

Gestisce:
1. Sessioni multiple concorrenti e stato del microfono continuo (PCM a 24kHz o 16kHz).
2. Cancellazione atomica del flusso audio in uscita (Barge-In) preservando la KV-Cache (Zero-Reset).
3. Emissione e proiezione degli eventi conforme a `/v1/realtime?duplex=1`.

Riferimenti ufficiali SDD:
- vllm_omni.experimental.fullduplex.serving_adapter
- PR #3907 (Normative Realtime event contract & Zero-Reset KV-Cache)
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
from vllm_omni_plugin.contracts import DuplexRealtimeEvent, DuplexState


class _SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_agent_speaking: bool = False
        self.kv_cache_retained: bool = True
        self.epoch: int = 1
        self.turn_id: int = 1
        self.pcm_buffer: bytearray = bytearray()
        self.last_decision: Optional[DuplexState] = None
        self.last_activity_time: float = time.time()


class Gemma4OmniServingRuntimeAdapter:
    """Adapter di serving Realtime WebSocket per Gemma-4-Omni Full-Duplex."""

    def __init__(self):
        self._sessions: Dict[str, _SessionState] = {}

    def open_session(self, session_id: str) -> None:
        """Registra una nuova sessione duplex con fencing iniziale."""
        self._sessions[session_id] = _SessionState(session_id)

    def close_session(self, session_id: str) -> None:
        """Rilascia le risorse della sessione."""
        self._sessions.pop(session_id, None)

    def reap_expired_sessions(self, idle_ttl_s: float = 300.0) -> List[str]:
        """Elimina sessioni inattive oltre il tempo massimo specificato nel profilo deploy."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if (now - s.last_activity_time) > idle_ttl_s]
        for sid in expired:
            self.close_session(sid)
        return expired

    def set_agent_speaking(self, session_id: str, is_speaking: bool) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].is_agent_speaking = is_speaking

    def is_agent_speaking(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        return session.is_agent_speaking if session else False

    def is_kv_cache_retained(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        return session.kv_cache_retained if session else False

    def process_incoming_pcm(
        self,
        session_id: str,
        pcm_data: np.ndarray,
        forced_barge_in: bool = False,
    ) -> DuplexRealtimeEvent:
        """Processa il flusso continuo di campioni PCM in arrivo dal microfono del client.

        Se viene rilevato un Barge-In (interruzione vocale dell'utente mentre l'agente parla),
        attiva istantaneamente il blocco dell'audio uscente senza invalidare la KV-cache prefillata.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Sessione non trovata: {session_id}")

        session.last_activity_time = time.time()

        # Calcolo energia RMS del blocco audio utente
        if np.issubdtype(pcm_data.dtype, np.integer):
            pcm_float = pcm_data.astype(np.float32) / 32768.0
        else:
            pcm_float = pcm_data.astype(np.float32)
        rms_energy = float(np.sqrt(np.mean(pcm_float ** 2))) if len(pcm_float) > 0 else 0.0

        # Rilevamento Barge-In: utente con voce attiva (RMS > 0.05) mentre l'agente stava parlando
        is_barge_in = forced_barge_in or (session.is_agent_speaking and rms_energy > 0.05)

        if is_barge_in:
            # 1. Ferma la riproduzione dell'agente
            session.is_agent_speaking = False
            # 2. Incrementa l'epoca di fencing per scartare frame audio residui in coda
            session.epoch += 1
            # 3. Preserva la KV-cache (Zero-Reset policy) per rispondere coerentemente al contesto
            session.kv_cache_retained = True
            session.last_decision = DuplexState.BARGE_IN

            return DuplexRealtimeEvent(
                event_type="audio.cancelled",
                session_id=session_id,
                metadata={
                    "reason": "barge_in_detected",
                    "rms_energy": rms_energy,
                    "new_epoch": session.epoch,
                    "kv_retained": True,
                },
            )

        session.last_decision = DuplexState.SPEAK if session.is_agent_speaking else DuplexState.YIELD
        return DuplexRealtimeEvent(
            event_type="input_audio_buffer.appended",
            session_id=session_id,
            metadata={"samples": len(pcm_data), "rms_energy": rms_energy},
        )
