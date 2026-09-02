# -*- coding: utf-8 -*-
"""
duplex_kernel/__init__.py — Custom Duplex Kernel (ADR-029).

Runtime narrow ispirato a DwarfStar (antirez/ds4), adattato all'architettura
Gemma-4-Omni full-duplex (Thinker→V9→Talker→Code2Wav + TurnDetectorHead).

Autocontenuto: nessuna dipendenza da kaggle_pipeline (training), dal plugin
vLLM-Omni, né da componenti legacy (ht_adapter, LoRA su Gemma).
"""

__version__ = "0.4.0"

from duplex_kernel.config import *  # noqa: F401,F403
from duplex_kernel.engine import PipelineEngine  # noqa: F401
from duplex_kernel.listener import (  # noqa: F401
    AssistantReferenceBuffer,
    AssistantReferenceFrame,
    GemmaAudioTowerBackend,
    ListenerBackendOutput,
    GemmaListener,
    GemmaListenerState,
    ListenerOutput,
)
from duplex_kernel.kv_cache import (  # noqa: F401
    TalkerPrefixKV,
    ThinkerContext,
    TurnKVState,
    kv_memory_bytes,
)
from duplex_kernel.scheduler import (  # noqa: F401
    AudioFrame,
    AudioIngressQueue,
    DuplexScheduler,
    DuplexTick,
    FrameClock,
    MonotonicFrameClock,
    VirtualFrameClock,
)
from duplex_kernel.session import DuplexSession, SessionManager  # noqa: F401
from duplex_kernel.voices import load_voice_profiles, resolve_voice_conditioning  # noqa: F401
# TurnHead is intentionally not imported into the core package namespace.
# Use duplex_kernel.turnhead_adapter for the TurnPolicy integration.
