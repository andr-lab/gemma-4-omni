# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/contracts.py — Contratti e DTO per il Plugin vLLM-Omni Gemma-4-Omni.

Riferimenti ufficiali SDD:
- vllm_omni.experimental.fullduplex.engine.contracts
- PR #3907 (MiniCPM-o 4.5 & Duplex Realtime architecture)
- arXiv:2602.02204 (vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DuplexState(str, Enum):
    """5 Stati del Turno Conversazionale (ADR-027 / TurnDetectorHead)."""
    SPEAK = "SPEAK"                          # 0: L'agente parla / continua a parlare
    YIELD = "YIELD"                          # 1: L'agente cede la parola all'utente
    BARGE_IN = "BARGE_IN"                    # 2: L'utente interrompe attivamente l'agente
    USER_BACKCHANNEL = "USER_BACKCHANNEL"    # 3: L'utente emette un feedback breve (mhm, ok) senza interrompere
    EMIT_BACKCHANNEL = "EMIT_BACKCHANNEL"    # 4: L'agente emette un micro-feedback acustico


@dataclass(frozen=True)
class DuplexOutputDecision:
    """Decisione tipizzata emessa dal DuplexRuntimeExtension per ogni frame o chunk."""
    turn_state: DuplexState
    should_interrupt_agent: bool
    should_yield_to_user: bool
    is_user_backchannel: bool
    is_emit_backchannel: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplexRealtimeEvent:
    """Evento del protocollo Realtime WebSocket (/v1/realtime?duplex=1)."""
    event_type: str
    session_id: str
    data: Optional[bytes] = None
    transcript: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
