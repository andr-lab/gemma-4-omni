# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/duplex_adapter.py — Gemma4OmniServingRuntimeAdapter.

Adapter minimale per il protocollo OpenAI Realtime WebSocket (/v1/realtime)
sul modello Gemma-4-Omni (chat fallback: input text, output text+audio).

Protocolli di riferimento (SDD):
- vllm_omni/experimental/fullduplex/openai/runtime_adapter.py
  (ServingRuntimeAdapter, ServingRuntimeSessionState, RuntimeDataPlane)
- vllm_omni/experimental/fullduplex/openai/chat_fallback.py
  (ChatFallbackProjectorMixin — le sessioni generiche cadono sul chat)

Il fallback chat usa session.history (messaggi di testo) e la risposta è
proiettata in eventi duplex (response.output_item.added con testo e audio).
Il barge-in è gestito dal framework (epoch). Il TurnDetectorHead nativo
(data plane audio→scheduler) è il livello 2 (futuro).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any

from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    PcmAppendBuffer,
    RuntimeDataPlane,
    ServingRuntimeAdapter,
    ServingRuntimeSessionState,
)

from vllm_omni_plugin.input import Gemma4OmniPcmAppendBuffer as _L2PcmAppendBuffer


class Gemma4OmniPcmAppendBuffer(PcmAppendBuffer):
    """Buffer PCM 24kHz monofonico per le sessioni duplex (fallback chat).

    Delegato al buffer L2 (vllm_omni_plugin.input): accumula i chunk
    ``input_audio_buffer.append`` e alla commit produce un payload ``audio_url``
    con WAV valido per il media connector (STT upstream Thinker nativo).
    """

    def __init__(self) -> None:
        self._delegate = _L2PcmAppendBuffer()

    def pending_byte_count(self) -> int:
        return self._delegate.pending_byte_count

    def clear(self) -> None:
        self._delegate.clear()

    def clear_force_listen(self) -> None:
        self._delegate.clear_force_listen()

    def has_pending(self) -> bool:
        return self._delegate.has_pending()

    def has_reserved(self) -> bool:
        return self._delegate.has_reserved()

    def prepare_append(self, payload: Any = None, **kwargs: Any) -> Any:
        # Firma compat con il session_runner (payload posizionale + opzionali).
        return self._delegate.prepare_append(
            payload if isinstance(payload, dict) else (kwargs.get("payload") or {}),
            operation_id=kwargs.get("operation_id", ""),
            chunk_period_ms=kwargs.get("chunk_period_ms", 1000),
            allow_emit=kwargs.get("allow_emit", True),
        )

    def prepare_commit(self, *, chunk_period_ms: int = 1000, operation_id: str | None = None, **kwargs: Any) -> Any:
        return self._delegate.prepare_commit(operation_id=operation_id or "", chunk_period_ms=chunk_period_ms)

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        return self._delegate.flush(chunk_period_ms=chunk_period_ms)


class Gemma4OmniSessionState(ServingRuntimeSessionState):
    """Stato sessione duplex minimale."""

    def __init__(self) -> None:
        self.audio_buffer: PcmAppendBuffer = Gemma4OmniPcmAppendBuffer()
        self.input_since_commit: bool = False
        self.speech_since_commit: bool = False
        self.native_context_locked: bool = False
        self.committed_audio_payload: dict[str, object] | None = None
        self.committed_audio_operation_id: str | None = None
        self.committed_audio_reserved_bytes: int = 0
        self.deferred_response_create: bool = False
        self.deferred_precreate_response: bool = False
        self.data_plane_task: asyncio.Task[None] | None = None
        self.data_plane_restart_requested: bool = False
        self.continuation_owner_id: str | None = None
        self.continuation_units: int = 0
        self.pending_silence_task: asyncio.Task[bool] | None = None
        self.pending_silence_owner_id: str | None = None
        self.silence_continuation_scheduler: Callable[..., Any] | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes = reserved_bytes

    def clear_committed_audio(self) -> int:
        n = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        return n

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0


class Gemma4OmniRuntimeDataPlane(RuntimeDataPlane):
    """Data plane no-op: il fallback chat non usa il data plane."""

    def begin_request(self, request_id: str) -> None:
        pass

    def is_terminal(self, request_id: str | None) -> bool:
        return False

    def mark_terminal(self, request_id: str) -> None:
        pass

    def close_stream(self, request_id: str) -> None:
        pass

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        pass

    def project(self, result: object, *, context: object | None = None) -> Iterable[dict[str, object]]:
        return ()


class Gemma4OmniServingRuntimeAdapter(ServingRuntimeAdapter):
    """Adapter di serving per il protocollo Realtime (chat fallback)."""

    adapter_id = "gemma4_omni_duplex"
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys: frozenset[str] = frozenset()
    collect_outputs_on_append = False

    def __init__(self, encode_audio: Callable[..., str | None] | None = None) -> None:
        self._encode_audio = encode_audio
        self.session_states: MutableMapping[str, ServingRuntimeSessionState] = {}
        self.data_plane: RuntimeDataPlane = Gemma4OmniRuntimeDataPlane()

    def create_session_state(self) -> ServingRuntimeSessionState:
        return Gemma4OmniSessionState()

    def session_state(self, session_id: str) -> ServingRuntimeSessionState:
        return self.session_states[session_id]

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    def is_enabled(self, config: object) -> bool:
        return True

    def capabilities(self, *, max_sessions: int) -> object:
        # Livello 2 (porting data plane audio-native): l'input audio del
        # client viene accettato (supports_input_append), bufferizzato e
        # committato al Thinker per lo STT upstream (encoder audio nativo
        # Gemma4). Il barge-in è client-side via epoch (zero-reset KV).
        return DuplexCapabilities(
            supports_model_native_turn_policy=False,
            supports_input_append=True,
            supports_replace_latest_chunk=True,
            supports_reencode_context=True,
            supports_turn_commit_only=True,
            supports_barge_in=True,
            supports_kv_lease=False,
            chunk_period_ms=100,
            input_modes=["append_audio_chunk", "turn_commit_only"],
        )

    def validate_client_extra_body(self, extra_body: object) -> None:
        pass

    async def prepare_runtime_config(self, config: object, *, model_config: Any) -> dict[str, object]:
        extra = getattr(config, "extra_body", None)
        if isinstance(extra, dict):
            return {"realtime_extra_body": dict(extra)}
        return {}

    def runtime_config_for_update(
        self,
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        return dict(current)

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> object:
        return {}
