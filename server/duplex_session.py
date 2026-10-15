# -*- coding: utf-8 -*-
"""
server/duplex_session.py — Gestore di Sessione Full-Duplex per WebSocket OpenAI Realtime.

Supporta:
1. Streaming parallelo di trascrizione testuale (`response.audio_transcript.delta`) e audio PCM (`response.audio.delta`).
2. Fencing atomico per cancellazione istantanea su Barge-In.
3. Chiamate a funzione live sul canale WebSocket.
"""

import base64
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
import numpy as np


class DuplexSession:
    """Rappresenta una singola sessione WebSocket conversazionale Full-Duplex."""

    def __init__(self, session_id: str, voice: str = "Vivian"):
        self.session_id = session_id
        self.voice = voice
        self.response_id: str = f"resp_{uuid.uuid4().hex[:8]}"
        self.item_id: str = f"item_{uuid.uuid4().hex[:8]}"
        self.is_agent_speaking: bool = False
        self.kv_cache_retained: bool = True
        self.epoch: int = 1
        self.transcript_buffer: List[str] = []
        self.tools: List[Dict[str, Any]] = []
        self.conversation_items: List[Dict[str, Any]] = []

    def add_conversation_item(self, item: Dict[str, Any]):
        """Aggiunge un item alla cronologia conversazionale della sessione."""
        self.conversation_items.append(item)

    def get_last_user_text(self) -> str:
        """Estrae l'ultimo testo utente dalla cronologia conversazionale."""
        for item in reversed(self.conversation_items):
            if item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("input_text", "text")]
                    if texts:
                        return " ".join(texts)
        return ""

    def start_response(self, on_event: Callable[[Dict[str, Any]], None]):
        """Inizializza una nuova risposta visibile dell'assistente."""
        self.response_id = f"resp_{uuid.uuid4().hex[:8]}"
        self.item_id = f"item_{uuid.uuid4().hex[:8]}"
        self.transcript_buffer.clear()
        self.is_agent_speaking = True

        on_event({
            "type": "response.created",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response": {
                "id": self.response_id,
                "status": "in_progress",
                "output": [],
            },
        })

    def emit_text_delta(self, delta_text: str, on_event: Callable[[Dict[str, Any]], None]):
        """Emette uno snippet di testo durante la sintesi (streaming parallelo)."""
        if not delta_text:
            return
        self.transcript_buffer.append(delta_text)
        on_event({
            "type": "response.audio_transcript.delta",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response_id": self.response_id,
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta_text,
        })

    def emit_audio_chunk(self, pcm_bytes: bytes, on_event: Callable[[Dict[str, Any]], None]):
        """Emette un chunk audio PCM (24kHz 16-bit mono) codificato in Base64."""
        if not pcm_bytes:
            return
        b64_chunk = base64.b64encode(pcm_bytes).decode("ascii")
        on_event({
            "type": "response.audio.delta",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response_id": self.response_id,
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": b64_chunk,
        })

    def emit_function_call(
        self,
        call_id: str,
        name: str,
        arguments_json: str,
        on_event: Callable[[Dict[str, Any]], None],
    ):
        """Emette una chiamata a funzione rilevata durante la conversazione."""
        on_event({
            "type": "response.function_call_arguments.done",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response_id": self.response_id,
            "item_id": self.item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments_json,
        })

    def finish_response(self, on_event: Callable[[Dict[str, Any]], None]):
        """Conclude la risposta corrente."""
        full_transcript = "".join(self.transcript_buffer)
        if full_transcript:
            on_event({
                "type": "response.audio_transcript.done",
                "event_id": f"evt_{uuid.uuid4().hex[:8]}",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "transcript": full_transcript,
            })
        on_event({
            "type": "response.audio.done",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response_id": self.response_id,
            "item_id": self.item_id,
        })
        on_event({
            "type": "response.done",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response": {
                "id": self.response_id,
                "status": "completed",
            },
        })
        self.is_agent_speaking = False

    def handle_barge_in(self, on_event: Callable[[Dict[str, Any]], None]):
        """Gestisce l'interruzione vocale dell'utente con cancellazione atomica e Zero-Reset."""
        self.is_agent_speaking = False
        self.epoch += 1
        self.kv_cache_retained = True
        on_event({
            "type": "audio.cancelled",
            "event_id": f"evt_{uuid.uuid4().hex[:8]}",
            "response_id": self.response_id,
            "reason": "user_barge_in",
            "epoch": self.epoch,
        })


class DuplexSessionManager:
    """Registro delle sessioni attive."""

    def __init__(self):
        self._sessions: Dict[str, DuplexSession] = {}

    def create_session(self, session_id: Optional[str] = None, voice: str = "Vivian") -> DuplexSession:
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        session = DuplexSession(sid, voice=voice)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[DuplexSession]:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str):
        self._sessions.pop(session_id, None)
